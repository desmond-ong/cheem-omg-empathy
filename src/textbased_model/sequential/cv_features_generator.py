import time
from gensim.models import KeyedVectors
import sys
sys.path.append("/raid/omgempathy/")
from src.textbased_model import utils, config
import re
from nltk.corpus import stopwords
import argparse
import torch
import math
import os
from src.textbased_model.models import LSTMEmo_utter_sequence
from src.textbased_model import text_features_generator as feature_utils
import numpy as np

fps = 25

regex = re.compile("(Story_\d+)")
en_stopwords = None
if not config.use_stopwords:
    en_stopwords = set(stopwords.words('english'))

def get_stories(dirs):
    stories = {}
    for data_source in dirs:
        for name in data_source.keys():  # name: Subject_10_Story_1
            results = regex.search(name)
            story = results.group(1)
            if story not in stories:
                stories[story] = [name]
            else:
                stories[story].append(name)
    return stories


def combine_inputs(inputs):
    combined = {}
    for data_source in inputs:
        for key, value in data_source.items():
            combined[key] = value

    return combined


def add_files_for_a_story(files, source_tensors, target_tensors):
    for file in files:
        target_tensors[file] = source_tensors[file]


def get_output_layer_features(trained_model, validation_data):
    trained_model.eval()
    features = {}  # {filename: [output_layer_features]}
    for fname, utter_tensors in validation_data.items():
        X = utter_tensors[0].to(device)
        X = X.view(X.shape[0], 1, -1)
        valid_indices = utter_tensors[2]
        with torch.no_grad():
            pred, output_layer_features = trained_model(X, use_dropout=False)
            # output_layer_features = output_layer_features.view(output_layer_features.shape[0], -1)
            # print(output_layer_features.shape)
            output_layer_features = output_layer_features.squeeze(1)
            # print(output_layer_features.shape)
        features[fname] = (output_layer_features, valid_indices)
    return features


def get_utterances_with_timing(data_source):
    utterances = {}
    for source in data_source:
        input_dir = "data/{}/transcripts_with_scores".format(source)
        for filename in os.listdir(input_dir):
            if filename.startswith(".") or not filename.endswith("json"):
                continue
            tmp = utils.load_utterances_with_timing(os.path.join(input_dir, filename))  # {id: (text, start time, end time)}
            file_id = filename[:-5]
            utterances[file_id] = tmp
    return utterances


def write_features(features, output_dir, gt_sequences, utterances):
    utils.mkdir(output_dir)
    for fname, value in features.items():
        output_file = os.path.join(output_dir, "{}.npy".format(fname))
        seconds = math.ceil(len(gt_sequences[fname])/fps)
        output_layer_features = value[0]
        valid_indices = value[1]
        file_utterances = utterances[fname]
        vectors = []
        index = 0  # index of valid indices

        for second in range(seconds):
            index = feature_utils.get_utter_index_for_second(second, index, valid_indices, file_utterances)
            vec = output_layer_features[index].cpu().numpy()
            vectors.append(vec)
        vectorsnp = np.stack(vectors)
        # np.savetxt(output_file, vectorsnp)
        np.save(output_file, vectorsnp)


parser = argparse.ArgumentParser("Training model")
parser.add_argument("-g", "--gpu", help="gpu", required=False, default=0, type=int)
parser.add_argument("-m", "--model", help="model's id, e.g., model_1542296294", required=False, default="model_1543400926")
parser.add_argument("--epoch", default=4999, type=int)
args = vars(parser.parse_args())
gpu = int(args['gpu'])

model_name = args['model']
model_timestamp = model_name[model_name.index("_") + 1:]

if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

device = config.device

# load config
config_file = 'data/Training/training_logs/{}.config'.format(model_timestamp)
configs = utils.load_model_config(config_file)
keywords = config.config_keywords
input_dim = config.input_dim
lstm_dim = config.lstm_dim
ln1_dim = config.ln1_output_dim

dropout = 0  # no drop out

if keywords['lstm_dim'] in configs:
    lstm_dim = int(configs[keywords['lstm_dim']])

if keywords['ln1_output_dim'] in configs:
    ln1_dim = int(configs[keywords['ln1_output_dim']])

print("lstm_dim: {}\tln1_dim: {}".format(lstm_dim, ln1_dim))

# ---------------------------------------------------------------
# Prepare cross validation data

input_dir_name = "transcripts_with_scores_difference"

training_dir = "data/Training/{}".format(input_dir_name)
val_dir = "data/Validation/{}".format(input_dir_name)

print("Loading input data")
training_data = utils.load_data_dir(training_dir)  # {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}
val_data = utils.load_data_dir(val_dir)
combined_data = combine_inputs([training_data, val_data])
stories = get_stories([training_data, val_data])

utterances_with_time = get_utterances_with_timing(["Training", "Validation"])

print("Loading groundtruth sequences...")
train_gt_sequences = utils.load_groundtruth_sequences(config.training['labels'])  # {filename: [scores]}
val_gt_sequences = utils.load_groundtruth_sequences(config.validation['labels'])
combined_gt = combine_inputs([train_gt_sequences, val_gt_sequences])

print("Loading word embeddings...")
word_embeddings_file = config.glove_word2vec_file_filtered
word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
vocabs = word_embeddings.wv.vocab

print("Get data tensors...")
# {fname: [list of utterances' embeddings (i.e., avg words' embeddings)]}
training_tensor = utils.get_average_word_embeddings_for_utters(training_data, vocabs, word_embeddings, en_stopwords)
val_tensor = utils.get_average_word_embeddings_for_utters(val_data, vocabs, word_embeddings, en_stopwords)
combined_tensor = combine_inputs([training_tensor, val_tensor])

cv_train = {}  # {Story: {train tensor}}
cv_val = {}  # {Story: {val tensor}}
num_stories = len(stories)
keys = list(stories.keys())

for i in range(num_stories):
    tmp_train = {}
    tmp_val = {}
    for j in range(num_stories):
        if j != i:
            add_files_for_a_story(stories[keys[j]], combined_tensor, tmp_train)
        else:
            add_files_for_a_story(stories[keys[j]], combined_tensor, tmp_val)

    cv_train[keys[i]] = tmp_train
    cv_val[keys[i]] = tmp_val

# -------------------------------------------------------------------------
# cross validation prediction
start = time.time()

output_root_dir = "data/Training/text_output_layer_features_cross_validation/"
utils.mkdir(output_root_dir)
output_root_dir = os.path.join(output_root_dir, model_name)
utils.mkdir(output_root_dir)

for story in keys:
    print("Story: ", story)
    train_data_tensors = cv_train[story]
    val_data_tensors = cv_val[story]

    output_dir = os.path.join(output_root_dir, story)
    utils.mkdir(output_dir)

    trained_model_path = "data/Training/models/{}_{}_{}.pt".format(model_name, story, args['epoch'])
    print("Using model: {}".format(trained_model_path))
    trained_model = LSTMEmo_utter_sequence(input_dim, lstm_dim, ln1_dim).to(device)
    trained_model.load_state_dict(torch.load(trained_model_path))
    trained_model.to(device)

    print("Generating features for train/val...")

    val_features = get_output_layer_features(trained_model, val_data_tensors)
    train_features = get_output_layer_features(trained_model, train_data_tensors)

    write_features(val_features, os.path.join(output_dir, "val"), combined_gt, utterances_with_time)
    write_features(train_features, os.path.join(output_dir, "train"), combined_gt, utterances_with_time)

utils.print_time(start)

#
# input_file = "/Users/sonnguyen/raid/data/Training/text_output_layer_features_cross_validation/model_1543400926/Story_2/train/Subject_1_Story_1.npy"
# a = np.load(input_file)
#
# print(a.shape)
import time
from gensim.models import KeyedVectors
import sys
sys.path.append("/raid/omgempathy/")
from src.textbased_model import utils, config
from src.textbased_model.datasets import Utterances_Chunk
import re
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
import argparse
import torch
import math
import os
from tensorboardX import SummaryWriter
from src.textbased_model.models import LSTMEmo_utter_sequence
from torch import nn
from torch import optim
from src.sample import calculateCCC
import numpy as np

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


def validate(trained_model, validation_data, combined_data_info, gt_sequences):
    trained_model.eval()
    scores = {}  # {filename: (ccc, pearson)}
    sequences = {}
    for fname, utter_tensors in validation_data.items():
        predicted_sequence = []  # store the predicted sequence
        prev_predicted = 0.0  # in case the "utterance" tensor is None --> use the previous score (to keep continuity)
        tmp_info = combined_data_info[fname]

        X = utter_tensors[0].to(device)
        X = X.view(X.shape[0], 1, -1)
        valid_indices = utter_tensors[2]
        with torch.no_grad():
            pred = trained_model(X, use_dropout=False)
        pred = pred.view(-1, 1)

        indices = tmp_info.keys()
        for utter_index in indices:
            tmp = tmp_info[utter_index]
            start_index = tmp[2]  # start frame
            end_index = tmp[3]  # end frame
            if utter_index not in valid_indices:
                utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)
                continue
            tmp_indx = valid_indices.index(utter_index)
            pred_difference = pred[tmp_indx].item()
            prev_predicted = prev_predicted + pred_difference  # trained on difference
            utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)

        # after finish for 1 file, stores the predicted sequence
        gt_sequence = gt_sequences[fname]
        predicted_sequence = utils.refine_predicted_sequence(predicted_sequence, len(gt_sequence))
        # compute ccc
        ccc, pearson = calculateCCC.ccc(gt_sequence, predicted_sequence)
        scores[fname] = (ccc, pearson)
        sequences[fname] = predicted_sequence

    return scores, sequences


def write_details_and_record(scores, detail_writer, pearsons, cccs):
    detail_writer.write("@FileName\tPearson\tCCC\n")
    for fname, value in scores.items():
        detail_writer.write("{}\t{}\t{}\n".format(fname, value[1], value[0]))
        pearsons.append(value[1])
        cccs.append(value[0])


def write_eval_results(val_scores, train_scores, story, output_writer, output_details_writer):
    val_pearsons = []
    val_cccs = []
    train_pearsons = []
    train_cccs = []

    output_details_writer.write("@Story as Val: {}\n\n".format(story))
    output_details_writer.write("Results on Validation set:\n\n")
    write_details_and_record(val_scores, output_details_writer, val_pearsons, val_cccs)

    output_details_writer.write("Results on Training set:\n\n")
    write_details_and_record(train_scores, output_details_writer, train_pearsons, train_cccs)
    output_details_writer.write("\n--------------------------------------------\n")

    output_writer.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(story,
                                                                      np.mean(train_pearsons), np.mean(train_cccs),
                                                                      np.mean(val_pearsons), np.mean(val_cccs)))


def write_sequences(sequences, model_pred_output):
    for fname, sequence in sequences.items():
        pred_output_file = os.path.join(model_pred_output, "{}.csv".format(fname))
        utils.write_predicted_sequences(sequence, pred_output_file)


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

model_pred_output = "data/Training/crossValidation_text_predictions/"
utils.mkdir(model_pred_output)
model_pred_output = os.path.join(model_pred_output, "{}_{}_cv".format(model_timestamp, args['epoch']))
utils.mkdir(model_pred_output)

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
log_dir = config.log_dir
utils.mkdir(log_dir)
results_output = os.path.join(log_dir, "{}_eval_only_cv_results.txt".format(model_timestamp))
results_details_output = os.path.join(log_dir, "{}_eval_only_cv_results_details.txt".format(model_timestamp))

result_writer = open(results_output, 'w')
result_writer.write("\tTraining\t\tValidation\n")
result_writer.write("@Story_as_Val\tPearson\tCCC\tPearson\tCCC\n")
result_detail_writer = open(results_details_output, 'w')

for story in keys:
    train_data_tensors = cv_train[story]
    val_data_tensors = cv_val[story]

    # model_file = 'data/Training/models/{}.pt'.format(model_name)
    # model_1543400926_Story_5_4999.pt

    trained_model_path = "data/Training/models/{}_{}_{}.pt".format(model_name, story, args['epoch'])
    print("Using model: {}".format(trained_model_path))
    trained_model = LSTMEmo_utter_sequence(input_dim, lstm_dim, ln1_dim).to(device)
    trained_model.load_state_dict(torch.load(trained_model_path))
    trained_model.to(device)

    print("Evaluating the trained model on train/val...")

    val_scores, val_sequences = validate(trained_model, val_data_tensors, combined_data, combined_gt)
    train_scores, _ = validate(trained_model, train_data_tensors, combined_data, combined_gt)

    write_eval_results(val_scores, train_scores, story, result_writer, result_detail_writer)
    result_writer.flush()
    result_detail_writer.flush()
    write_sequences(val_sequences, model_pred_output)

result_writer.flush()
result_writer.close()
result_detail_writer.flush()
result_detail_writer.close()

utils.print_time(start)
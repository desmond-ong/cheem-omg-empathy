'''
run model prediction for the test data
'''

import sys
from gensim.models import KeyedVectors
import torch
import os
import time
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.sample import calculateCCC
from src.textbased_model.models import LSTMEmo_utter_sequence
import argparse
from nltk.corpus import stopwords

import re
regex = re.compile("(Subject_\d+)")

def get_subjects(dirs):
    subjects = {}
    for data_source in dirs:
        for name in data_source.keys():  # name: Subject_10_Story_1
            results = regex.search(name)
            subject = results.group(1)
            if subject not in subjects:
                subjects[subject] = [name]
            else:
                subjects[subject].append(name)
    return subjects


def run_prediction(model, testing_data):
    model.eval()
    for fname, utter_tensors in testing_data.items():
        predicted_sequence = []  # store the predicted sequence
        prev_predicted = 0.0  # in case the "utterance" tensor is None --> use the previous score (to keep continuity)
        tmp_info = data[fname]

        X = utter_tensors[0].to(device)
        X = X.view(X.shape[0], 1, -1)
        valid_indices = utter_tensors[2]
        with torch.no_grad():
            pred = model(X, use_dropout=False)
        pred = pred.view(-1, 1)

        indices = data[fname].keys()
        for utter_index in indices:
            tmp = tmp_info[utter_index]
            start_index = tmp[2]  # start frame
            end_index = tmp[3]  # end frame
            if utter_index not in valid_indices:
                utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)
                continue
            tmp_indx = valid_indices.index(utter_index)
            new_value = pred[tmp_indx].item()
            if trained_on_difference:
                prev_predicted = prev_predicted + new_value
            else:
                prev_predicted = new_value
            utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)

        # after finish for 1 file, stores the predicted sequence
        predicted_sequence = utils.refine_predicted_sequence(predicted_sequence, frames_count[fname])
        # write result
        pred_output_file = os.path.join(model_pred_output, "{}.csv".format(fname))
        utils.write_predicted_sequences(predicted_sequence, pred_output_file)


en_stopwords = set(stopwords.words('english'))

parser = argparse.ArgumentParser("Perform trained model prediction")
parser.add_argument("-m", "--model", help="Saved models' dir", default="data/Training/text_models_t1/personalized/", required=False)
parser.add_argument('-d', '--difference', help="trained on different?: 0: No (actuall value), 1: Yes", default=1, type=int)
parser.add_argument("--data", help="Path to the to-be-evaluated dataset, e.g., data/Testing/transcripts_with_scores/",
                    default="data/Testing/transcripts_with_scores/", required=False)
parser.add_argument("--output", help="Prediction output dir, e.g., data/Testing/text_personalized_predictions/",
                    default="data/Testing/text_personalized_predictions/", required=False)

parser.add_argument("-f", "--frames_count", help="Frame counts file", default="data/Testing/Frames_Count.txt")

parser.add_argument("--input_dim", help="Input dimension", default=300, type=int, required=False)
parser.add_argument("--lstm_dim", help="LSTM hidden's dimension", default=256, type=int)
parser.add_argument("--linear1_dim", help="FC 1's output dimension", default=128, type=int)
parser.add_argument("--embeddings", help="Word embeddings' file", default='data/glove.840B.300d.filtered.word2vec',
                    type=str)

args = vars(parser.parse_args())
trained_on_difference = (args['difference'] == 1)
model_dir = args['model']

print("Models dir: {}".format(model_dir))

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

# from importlib import reload
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load config
input_dim = int(args['input_dim'])
lstm_dim = int(args['lstm_dim'])
ln1_dim = int(args['linear1_dim'])
print("lstm_dim: {}\tln1_dim: {}".format(lstm_dim, ln1_dim))
dropout = 0  # no drop out

model_pred_output = args['output']
utils.mkdir(model_pred_output)

print("Loading data...")
start = time.time()
data = utils.load_data_dir(args['data'])  # {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}
subjects = get_subjects([data])

# load frame counts
frame_count_file = args['frames_count']
frames_count = utils.load_num_frames(frame_count_file)

# word_embeddings_file = config.glove_word2vec_file
word_embeddings_file = args['embeddings']
print("Loading word embeddings...")
we_model = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
vocabs = we_model.wv.vocab

print("Compute avg word embeddings and convert to tensor")
data_tensor = utils.get_average_word_embeddings_for_utters(data, vocabs, we_model, en_stopwords)  # {fname: [list of utterances' embeddings (i.e., avg words' embeddings)]}
utils.print_time(start)

# generate prediction for each subject
start = time.time()
print("Run prediction, results are saved to {}".format(model_pred_output))

for subject, files in subjects.items():
    model_file = os.path.join(model_dir, "{}.pt".format(subject))
    print("Model file: {}".format(model_file))
    model = LSTMEmo_utter_sequence(input_dim, lstm_dim, ln1_dim)
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    testing_data = {}
    for filename in files:
        testing_data[filename] = data_tensor[filename]

    run_prediction(model, testing_data)

utils.print_time(start)



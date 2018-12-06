'''
Testing with sequence, take the average as the final output
'''

import sys
from gensim.models import KeyedVectors
import torch
import os
import time
import numpy as np

sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.sample import calculateCCC
from src.textbased_model.models import LSTMEmo_duo_sequence
import argparse

parser = argparse.ArgumentParser("Run trained model on train/val (validation)")
parser.add_argument("-m", "--model", help="model's id, e.g., model_1542296294_999", required=True)
parser.add_argument('-s', '--source', help="data source: train/val", default="val")
parser.add_argument('-t', '--threshold', help="threshold for changes, < 0: no threshold", default=-1.0, type=float)

args = vars(parser.parse_args())
model_name = args['model']
model_file = 'data/Training/models/{}.pt'.format(model_name)
model_id = model_name[model_name.index("_") + 1:]
model_timestamp = model_id[:model_id.index("_")]

threshold = float(args['threshold'])

# print(model_id)
data_source_name = "Validation"
data_source = config.validation
if args['source'] == "train":
    data_source = config.training
    data_source_name = "Training"

if torch.cuda.is_available():
    torch.cuda.set_device(config.gpu_for_testing)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

# from importlib import reload
device = config.device

# load model settings
config_file = 'data/Training/training_logs/{}.config'.format(model_timestamp)
configs = utils.load_model_config(config_file)
keywords = config.config_keywords
input_dim = config.input_dim
lstm_dim = config.lstm_dim
ln1_dim = config.ln1_output_dim
dropout = 0  # no drop out
seq_size = 5 # default

if keywords['sequence'] in configs:
    seq_size = int(configs[keywords['sequence']])

if keywords['lstm_dim'] in configs:
    lstm_dim = int(configs[keywords['lstm_dim']])

if keywords['ln1_output_dim'] in configs:
    ln1_dim = int(configs[keywords['ln1_output_dim']])

# if keywords['dropout'] in configs:
#     dropout = float(configs[keywords['dropout']])

print("lstm_dim: {}\tln1_dim: {}\tdropout: {}\tSequence Size: {}\tThreshold: {}\t".format(lstm_dim, ln1_dim,
                                                                                          dropout, seq_size, threshold))

model_pred_output = data_source['prediction_output']
utils.mkdir(model_pred_output)
model_pred_output = os.path.join(model_pred_output, "{}_seq_diff_{}".format(model_id, threshold))
utils.mkdir(model_pred_output)

print("Loading validation data...")
start = time.time()
data = utils.load_data_dir(data_source['data'])
utils.print_time(start)

print("Generate sequences")
sampled_data = utils.apply_data_sampling(data, seq_size)

print("Loading groundtruth sequences...")
start = time.time()
gt_sequences = utils.load_groundtruth_sequences(data_source['labels'])
utils.print_time(start)

# word_embeddings_file = config.glove_word2vec_file
word_embeddings_file = config.glove_word2vec_file_filtered
start = time.time()
print("Loading word embeddings...")
we_model = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
vocabs = we_model.wv.vocab
utils.print_time(start)

model = LSTMEmo_duo_sequence(input_dim, lstm_dim, ln1_dim, dropout=dropout)
model.load_state_dict(torch.load(model_file))
model.to(device)
data_tensor = utils.prepare_data(data, vocabs, we_model)

start = time.time()

print("Run validation, results are saved to {}".format(model_pred_output))
model.eval()

for fname, sequences in sampled_data.items():
    tensors = data_tensor[fname]
    tmp_info = data[fname]
    predictions = {}  # {id: [predicted values in different sequences]} (later take the mean as the final predicted value)
    for sequence in sequences:
        hidden = model.init_hidden()  # initialize hidden (zeros) for a new sequence
        for utter_id in sequence:
            utter = tensors[utter_id]
            if utter[0] is None:
                # no valid words
                continue
            X = utter[0].to(device)
            X = X.view(X.shape[0], 1, -1)
            with torch.no_grad():
                pred, hidden = model(X, hidden)
                last_output = pred[-1].view(1, -1)
            # add the predicted value to the set for later computation
            if utter_id in predictions:
                predictions[utter_id].append(last_output.cpu().item())
            else:
                predictions[utter_id] = [last_output.cpu().item()]
    # after going through all the sequences, compute the final prediction score (average)
    indices = tmp_info.keys()
    predicted_sequence = utils.get_predicted_sequence_difference(predictions, tmp_info, len(gt_sequences[fname]),
                                                                 threshold=threshold)

    # write result
    pred_output_file = os.path.join(model_pred_output, "{}.csv".format(fname))
    utils.write_predicted_sequences(predicted_sequence, pred_output_file)

utils.print_time(start)
# after finishing the running, run the evaluation
gt_dir = "data/{}/Annotations".format(data_source_name)
# print("GT: {}".format(gt_dir))
# print("pred: {}".format(model_pred_output))
calculateCCC.calculateCCC(gt_dir, model_pred_output)






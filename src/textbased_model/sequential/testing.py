import sys
from gensim.models import KeyedVectors
import torch
import os
import time
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.sample import calculateCCC
from src.textbased_model.models import LSTMEmo_duo_sequence
import argparse

parser = argparse.ArgumentParser("Run trained model on train/val (validation)")
parser.add_argument("-m", "--model", help="model's id, e.g., model_1542296294_999", required=True)
parser.add_argument('-s', '--source', help="data source: train/val", default="val")

args = vars(parser.parse_args())
model_name = args['model']
model_file = 'data/Training/models/{}.pt'.format(model_name)
model_id = model_name[model_name.index("_") + 1:]
model_timestamp = model_id[:model_id.index("_")]

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

if keywords['lstm_dim'] in configs:
    lstm_dim = int(configs[keywords['lstm_dim']])

if keywords['ln1_output_dim'] in configs:
    ln1_dim = int(configs[keywords['ln1_output_dim']])

# if keywords['dropout'] in configs:
#     dropout = float(configs[keywords['dropout']])

print("lstm_dim: {}\tln1_dim: {}\tdropout: {}".format(lstm_dim, ln1_dim, dropout))

model_pred_output = data_source['prediction_output']
utils.mkdir(model_pred_output)
model_pred_output = os.path.join(model_pred_output, model_id)
utils.mkdir(model_pred_output)

print("Loading validation data...")
start = time.time()
data = utils.load_data_dir(data_source['data'])
utils.print_time(start)

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
for fname, utters in data_tensor.items():
    predicted_sequence = []  # store the predicted sequence
    prev_predicted = 0.0  # in case the "utterance" tensor is None --> use the previous score (to keep continuity)
    tmp_info = data[fname]
    hidden = model.init_hidden()  # new hidden vector for each video
    for index, utter in utters.items():
        tmp = tmp_info[index]
        start_index = tmp[2]  # start frame
        end_index = tmp[3]  # end frame

        if utter[0] is None:
            # in case the "utterance" tensor is None --> use the previous score (to keep continuity)
            utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)
            continue
        X = utter[0].to(device)
        X = X.view(X.shape[0], 1, -1)
        with torch.no_grad():
            pred, hidden = model(X, hidden)
            last_output = pred[-1].view(1, -1)

        # update sequence
        prev_predicted = last_output[0].cpu().item()
        utils.add_value_to_sequence(prev_predicted, predicted_sequence, start_index, end_index)

    # after finish for 1 file, stores the predicted sequence
    gt_sequence = gt_sequences[fname]
    predicted_sequence = utils.refine_predicted_sequence(predicted_sequence, len(gt_sequence))
    # write result
    pred_output_file = os.path.join(model_pred_output, "{}.csv".format(fname))
    utils.write_predicted_sequences(predicted_sequence, pred_output_file)


# after finishing the running, run the evaluation
gt_dir = "data/{}/Annotations".format(data_source_name)
# print("GT: {}".format(gt_dir))
# print("pred: {}".format(model_pred_output))
calculateCCC.calculateCCC(gt_dir, model_pred_output)






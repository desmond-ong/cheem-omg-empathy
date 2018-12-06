from gensim.models import KeyedVectors
import torch
import os
import time
import argparse
from nltk.corpus import stopwords
import re

import sys
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.sample import calculateCCC
from src.text_based_submit import utils
from src.text_based_submit.train import LSTMUtterChunk


pattern = ".*(model_.*)\.pt"
p = re.compile(pattern)

en_stopwords = set(stopwords.words('english'))

if __name__ == "__main__":
    en_stopwords = set(stopwords.words('english'))

    parser = argparse.ArgumentParser("Model Evaluation")
    parser.add_argument("-m", "--model", help="Saved model's path, e.g., d",
                        required=True)
    parser.add_argument('-d', '--diff', help="Predict differences?: False: No (actuall value), True: Yes", default=True,
                        type=bool)

    parser.add_argument("--data", help="Path to the to-be-evaluated dataset, e.g., data/Validation/transcripts_with_scores/", required=True)
    parser.add_argument("--output", help="Prediction output dir, e.g., data/Validation/model_predictions/", required=True)
    parser.add_argument("--groundtruth", help="Groundtruth's dir path, e.g., data/Validation/Annotations/", required=True)

    parser.add_argument("--gpu", help="gpu", required=False, default=0, type=int)
    parser.add_argument("--input_dim", help="Input dimension", default=300, type=int)
    parser.add_argument("--lstm_dim", help="LSTM hidden's dimension", default=256, type=int)
    parser.add_argument("--linear1_dim", help="FC 1's output dimension", default=128, type=int)
    parser.add_argument("--embeddings", help="Word embeddings' file", default='data/glove.840B.300d.filtered.word2vec',
                        type=str)

    args = vars(parser.parse_args())

    trained_on_difference = args['diff']
    model_file = args['model']
    results = p.search(model_file)
    model_id = results.group(1)

    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        print("#GPUs: {}".format(torch.cuda.device_count()))
        print("Current GPU: {}".format(torch.cuda.current_device()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = int(args['input_dim'])
    lstm_dim = int(args['lstm_dim'])
    ln1_dim = int(args['linear1_dim'])
    print("lstm_dim: {}\tln1_dim: {}".format(lstm_dim, ln1_dim))

    model_pred_output = args['output']
    utils.mkdir(model_pred_output)
    model_pred_output = os.path.join(model_pred_output, "{}_predictions".format(model_id))
    utils.mkdir(model_pred_output)

    print("Loading data...")
    start = time.time()
    data = utils.load_data_dir(args['data'])  # {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}

    print("Loading groundtruth sequences...")
    gt_sequences = utils.load_groundtruth_sequences(args['groundtruth'])

    # word_embeddings_file = config.glove_word2vec_file
    word_embeddings_file = args['embeddings']
    print("Loading word embeddings...")
    we_model = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
    vocabs = we_model.wv.vocab

    model = LSTMUtterChunk(input_dim, lstm_dim, ln1_dim, device=device)
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    print("Compute avg word embeddings and convert to tensor")
    data_tensor = utils.get_average_word_embeddings_for_utters(data, vocabs, we_model, en_stopwords)  # {fname: [list of utterances' embeddings (i.e., avg words' embeddings)]}
    # data_tensor = utils.prepare_data(data, vocabs, word_embeddings, en_stopwords)
    utils.print_time(start)

    start = time.time()

    print("Run validation, results are saved to {}".format(model_pred_output))
    model.eval()
    for fname, utter_tensors in data_tensor.items():
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
        gt_sequence = gt_sequences[fname]
        predicted_sequence = utils.refine_predicted_sequence(predicted_sequence, len(gt_sequence))
        # write result
        pred_output_file = os.path.join(model_pred_output, "{}.csv".format(fname))
        utils.write_predicted_sequences(predicted_sequence, pred_output_file)


    # after finishing the running, run the evaluation
    gt_dir = args['groundtruth']
    # print("GT: {}".format(gt_dir))
    # print("pred: {}".format(model_pred_output))
    calculateCCC.calculateCCC(gt_dir, model_pred_output)






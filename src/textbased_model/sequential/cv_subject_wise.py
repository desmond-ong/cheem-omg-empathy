'''
perform cross validation for personalized track (train and test for the same subject)
'''

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

regex = re.compile("(Subject_\d+)")
en_stopwords = None
if not config.use_stopwords:
    en_stopwords = set(stopwords.words('english'))

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


def combine_inputs(inputs):
    combined = {}
    for data_source in inputs:
        for key, value in data_source.items():
            combined[key] = value

    return combined


def add_files_for_a_subject(files, source_tensors, target_tensors):
    for file in files:
        target_tensors[file] = source_tensors[file]


def train(story, dataloader):
    model = LSTMEmo_utter_sequence(input_dim, lstm_dim, ln1_dim, batch_size).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), config.learning_rate)
    opt_text = 'SGD'
    if args['optimizer'].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        opt_text = 'adam'

    log_writer.write("Optimizer: {}\n".format(opt_text.upper()))
    print("Optimizer: {}".format(opt_text.upper()))

    print("Start training!")

    writer = SummaryWriter()
    iter_counter = 0
    for epoch in range(config.epoch_num):
        counter = 0
        total_loss = 0

        for data_inst in dataloader:
            X = data_inst[0].to(device)
            y = data_inst[1].to(device)

            if X.shape[0] != batch_size:
                continue
            X = X.view(X.shape[1], batch_size, -1)
            iter_counter += 1
            counter += 1
            model.zero_grad()  # zero the grad for a new sequence

            pred = model(X)
            pred = pred.view(batch_size, -1, 1)
            loss = loss_func(pred, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().item()
            writer.add_scalar("chunk_level", loss.cpu().item(), iter_counter)
        writer.add_scalar("epoch_level", total_loss / counter, epoch)
        if epoch > 0 and epoch % iter_print == 0:
            print("AvgLoss: {:.4f}. Epoch: {}. Time: {}.".format(total_loss / counter, epoch, utils.time_since(start)))
            log_writer.write(
                "AvgLoss: {:.4f}. Epoch: {}. Time: {}.\n".format(total_loss / counter, epoch, utils.time_since(start)))

        # save model after config.epoch_save
        if epoch > 0 and epoch % config.model_save_epoch_num == 0:
            output_file = os.path.join(model_output, "model_{}_{}_{}.pt".format(time_label, story, epoch))
            print("Saving model to {}".format(output_file))
            log_writer.write("Saving model to {}\n".format(output_file))
            torch.save(model.state_dict(), output_file)
            log_writer.flush()

    # make sure to save the last epoch
    output_file = os.path.join(model_output, "model_{}_{}_{}.pt".format(time_label, story, config.epoch_num - 1))
    if not os.path.isfile(output_file):
        print("Saving model to {}".format(output_file))
        log_writer.write("Saving model to {}\n".format(output_file))
        torch.save(model.state_dict(), output_file)

    writer.export_scalars_to_json(os.path.join(log_dir, "{}_{}_tensorboard.json".format(time_label, story)))
    writer.close()

    return model


def validate(trained_model, validation_data, combined_data_info, gt_sequences):
    trained_model.eval()
    scores = {}  # {filename: (ccc, pearson)}
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

    return scores


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


def cv_for_subject(input_files, subj_cv):
    train_data = subj_cv[0]
    val_data = subj_cv[1]

    for video in input_files:
        train_data_tensors = train_data[video]
        val_data_tensors = val_data[video]

        chunks = Utterances_Chunk(train_data_tensors, chunk_size, chunk_step=chunk_step)
        dataloader = DataLoader(chunks, batch_size=batch_size, shuffle=True)
        print("Validation video: {}".format(video))
        print("Chunk size: {}".format(chunk_size))
        print("Number of chunks: {}".format(len(chunks)))
        log_writer.write("Validation video: {}\n".format(video))
        log_writer.write("Chunk size: {}\n".format(chunk_size))
        log_writer.write("Number of chunks: {}\n    ".format(len(chunks)))

        trained_model = train(video, dataloader)

        trained_model.batch_size = 1  # test on one each
        print("Evaluating the trained model on train/val...")
        log_writer.write("Evaluating the trained model on train/val...\n")

        val_scores = validate(trained_model, val_data_tensors, combined_data, combined_gt)
        train_scores = validate(trained_model, train_data_tensors, combined_data, combined_gt)

        write_eval_results(val_scores, train_scores, video, result_writer, result_detail_writer)
        result_writer.flush()
        result_detail_writer.flush()
        log_writer.flush()
    result_writer.write("\n\n\n")
    result_detail_writer.write("\n\n\n")

parser = argparse.ArgumentParser("Training model")
parser.add_argument("-g", "--gpu", help="gpu", required=False, default=0, type=int)
parser.add_argument("-c", "--chunk", help="Chunk length", default=10, type=int)
parser.add_argument("-s", "--step", help="Chunk step", default=5, type=int)
parser.add_argument("-b", "--batch", help="Batch size", default=config.batch_size, type=int)
parser.add_argument("-opt", "--optimizer", help="optimizer (SGD/Adam)", required=False, default='SGD', type=str)

args = vars(parser.parse_args())
gpu = int(args['gpu'])

if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

# chunk_size = config.chunk_size
chunk_size = int(args['chunk'])
chunk_step = int(args['step'])

batch_size = int(args['batch'])
print("Batch size: {}".format(batch_size))
# from importlib import reload
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_label = "{:.0f}".format(time.time())

model_output = config.models_dir
utils.mkdir(model_output)
log_dir = config.log_dir
utils.mkdir(log_dir)
log_writer = open(os.path.join(log_dir, "{}_train_log.txt".format(time_label)), 'w')

iter_print = config.iter_print

input_dim = config.input_dim
lstm_dim = config.lstm_dim
ln1_dim = config.ln1_output_dim

with open(os.path.join(log_dir, "{}.config".format(time_label)), 'w') as config_writer:
    config_writer.write("dropout={}\n"
                        "learning_rate={}\n"
                        "epoch_num={}\n"
                        "lstm_dim={}\n"
                        "ln1_output_dim={}\n"
                        "batch_size={}\n"
                        "chunk_size={}\n"
                        "chunk_step={}\n".format(config.dropout, config.learning_rate, config.epoch_num,
                                                            config.lstm_dim, config.ln1_output_dim, config.batch_size,
                                                            chunk_size, chunk_step))
    config_writer.write("Use stopwords: {}\n".format(en_stopwords is None))
    config_writer.write("Train on average's differences\n")

# ---------------------------------------------------------------
# Prepare cross validation data

input_dir_name = "transcripts_with_scores_difference"

training_dir = "data/Training/{}".format(input_dir_name)
val_dir = "data/Validation/{}".format(input_dir_name)

print("Loading input data")
training_data = utils.load_data_dir(training_dir)  # {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}
val_data = utils.load_data_dir(val_dir)
combined_data = combine_inputs([training_data, val_data])
subjects = get_subjects([training_data, val_data])

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

# for each subject, run cv
# prepare data
subj_cv = {}

for subject, files in subjects.items():
    cv_train = {}  # {Subject: {train tensor}}
    cv_val = {}  # {Subject: {val tensor}}
    num_files = len(files)

    for i in range(num_files):
        tmp_train = {}
        tmp_val = {}
        for j in range(num_files):
            if j != i:
                tmp_train[files[j]] = combined_tensor[files[j]]
            else:
                tmp_val[files[j]] = combined_tensor[files[j]]
        cv_train[files[i]] = tmp_train
        cv_val[files[i]] = tmp_val
    subj_cv[subject] = (cv_train, cv_val)

# -------------------------------------------------------------------------
# cross validation train and test
start = time.time()
results_output = os.path.join(log_dir, "{}_cv_personalized_results.txt".format(time_label))
results_details_output = os.path.join(log_dir, "{}_cv_personalized_results_details.txt".format(time_label))

result_writer = open(results_output, 'w')
result_detail_writer = open(results_details_output, 'w')

for subject, files in subjects.items():
    result_writer.write("@Subject: {}\n\n".format(subject))
    result_detail_writer.write("@Subject: {}\n\n".format(subject))

    result_writer.write("\tTraining\t\tValidation\n")
    result_writer.write("@Video_as_Val\tPearson\tCCC\tPearson\tCCC\n")
    cv_for_subject(files, subj_cv[subject])

log_writer.flush()
log_writer.close()
result_writer.flush()
result_writer.close()
result_detail_writer.flush()
result_detail_writer.close()

utils.print_time(start)
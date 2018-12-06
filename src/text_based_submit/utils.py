import time
import math
import os
import pandas
import json
import torch
import csv
from nltk import word_tokenize
from collections import OrderedDict
import numpy as np


def time_since(start):
    s = time.time() - start
    h = math.floor(s / 3600)
    s = s - h * 3600
    m = math.floor(s / 60)
    s = s - m * 60
    if h > 0:
        return "{}h:{}m:{:.0f}s".format(h, m, s)
    return "{}m:{:.0f}s".format(m, s)


def print_time(start):
    print("Time: {}".format(time_since(start)))


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def load_subtitles(input_file):
    index_tmp = 0
    starting_time = ""
    ending_time = ""
    text = None

    subtitles = []
    flag = 0  # 0: start new, 1: time, 2: text
    with open(input_file, 'r') as reader:
        for line in reader.readlines():
            line = line.strip()
            if line == "":
                # empty line, write the old one
                subtitles.append((index_tmp, starting_time, ending_time, text))
                flag = 0
            elif flag == 0:
                # new sub, this line is index
                index_tmp = line
                flag = 1
            elif flag == 1:
                # timing line
                ps = line.split("-->")
                starting_time = ps[0].strip()
                ending_time = ps[1].strip()
                flag = 2
            elif flag == 2:
                text = line
        if flag == 2:
            # need to add the last one
            subtitles.append((index_tmp, starting_time, ending_time, text))
    return subtitles


def load_annotations(csv_file):
    data = pandas.read_csv(csv_file)
    scores = []
    for score in data['valence']:
        scores.append(score)
    return scores












def get_number(text):
    if "," in text:
        text = text.replace(",", ".")
    return float(text)

def time_to_frame(subtitle_time, fps=25):
    # 00:00:04,259 or 00:04:46,710
    total_seconds = get_seconds(subtitle_time)
    return int(total_seconds * fps)


def get_seconds(time_string):
    parts = time_string.split(":")
    hour = get_number(parts[0])
    min = get_number(parts[1])
    second = get_number(parts[2])

    total_seconds = hour * 60 * 60 + min * 60 + second

    return total_seconds

def load_data_dir(input_dir):
    '''

    :param input_dir:
    :return: {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}
    '''
    data = {}
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        if fname.startswith("."):
            continue
        # print("Loading file {}".format(fname))
        utters = load_data_for_file_dict(os.path.join(input_dir, fname))
        data[fname[:-5]] = utters
    return data


def load_data_for_file_array(input_file):
    # load transcripts with scores file, sorted by id
    utters = []  # {id: (string, avg score)}
    with open(input_file, 'r') as reader:
        data = json.load(reader)
        for utter in data:
            utters.append((utter['text'], utter['average'], utter['start_frame'], utter['end_frame']))

    return utters


def load_data_for_file_dict(input_file):
    # load transcripts with scores file, sorted by id
    utters = OrderedDict()  # {id: (string, avg score)}
    with open(input_file, 'r') as reader:
        data = json.load(reader)
        for utter in data:
            utters[utter['id']] = (utter['text'], utter['average'], utter['start_frame'], utter['end_frame'])
    return utters


def get_tensor_for_utter(utter, vocabs, word_embeddings, stopwords=None):
    '''
    :param utter:
    :param vocabs:
    :param word_embeddings:
    :param stopwords: if stopwords is not None --> remove stopwords
    :return:
    '''
    words = word_tokenize(utter.lower())
    tmp = []
    for word in words:
        if word in vocabs:
            if stopwords is not None and word in stopwords:
                continue
            tmp.append(torch.from_numpy(word_embeddings[word]))
        # else:
        #     print("Cannot find embeddings for:", word)
    if tmp:
        tmp = torch.stack(tmp)
        return tmp
    else:
        return None


def prepare_data_for_a_file(data_one_file, vocabs, word_embeddings, stopwords):
    '''
    :param data_one_file: list of (utterance, score)
    :param vocabs:
    :param word_embeddings:
    :return: ordereddict of list of tensors for each utterance
    '''
    tmp = OrderedDict()
    for index, utter in data_one_file.items():
        torch_tensor = get_tensor_for_utter(utter[0], vocabs, word_embeddings, stopwords)
        if torch_tensor is not None:
            tmp[index] = (torch_tensor, torch.Tensor([utter[1]]))
        else:
            tmp[index] = (None, None)  # to keep continuity
    return tmp


def prepare_data(input_data, vocabs, word_embeddings, stopwords=None):
    '''
    :param input_data: {file: [(text, average score)]}
    :param stopwords: if stopwords is not None --> remove stopwords
    :return: {file: OrderedDict{}} (number in torch)
    '''
    tensor_data = {}
    for fname, data in input_data.items():
        tensor_data[fname] = prepare_data_for_a_file(data, vocabs, word_embeddings, stopwords)
    return tensor_data


def load_groundtruth_sequences(csv_dir):
    gts = {}
    for filename in os.listdir(csv_dir):
        if not filename.endswith(".csv"):
            continue
        if filename.startswith("."):
            continue
        id = filename[:-4]
        data = load_annotations(os.path.join(csv_dir, filename))
        gts[id] = data
    return gts


def add_value_to_sequence(value, sequence, start, end):
    sequence[start:end] = [value for _ in range(start, end)]


def refine_predicted_sequence(predicted, groundtruth_len):
    # make sure len(predicted) == len(groundtruth)
    lp = len(predicted)
    # groundtruth_len = len(groundtruth)
    if lp == groundtruth_len:
        return predicted
    if lp > groundtruth_len:
        return predicted[:groundtruth_len]
    if lp < groundtruth_len:
        tmp = predicted[-1]  # fill the missing value with the last value
        predicted[lp:groundtruth_len] = [tmp for _ in range(lp, groundtruth_len)]
        return predicted


def write_predicted_sequences(sequence, output_file):
    with open(output_file, 'w') as writer:
        wr = csv.writer(writer, lineterminator='\n')
        wr.writerow(['valence'])
        for num in sequence:
            wr.writerow([num])


def apply_data_sampling_for_file(odict, sequence_size):
    '''
    data sampling, trim utterances sequence into sequences of continuous utterances
    :param odict: OrderedDict{} (number in torch)
    :return: [indices size config.sequence_size]
    '''
    if sequence_size < 1 or len(odict) < sequence_size:
        return [list(odict.keys())]

    sequences = []
    tmplist = list(odict.keys())
    for i in range(len(odict) - sequence_size + 1):
        tmp = []
        for j in range(i, i + sequence_size):
            tmp.append(tmplist[j])
        sequences.append(tmp)
    return sequences


def apply_data_sampling(data, sequence_size):
    '''
    data sampling, trim utterances sequence into sequences of continuous utterances
    :param data:  {filename: OrderDict {utter_id: (text, score, startframe, endframe)}}
    :return:  {file: [indices size config.sequence_size]}
    '''
    sampled_data = {}
    for filename, odict in data.items():
        seqs = apply_data_sampling_for_file(odict, sequence_size)
        sampled_data[filename] = seqs
    return sampled_data


def get_num_sequences(sampled_data):
    counter = 0
    for fname, seqs in sampled_data.items():
        counter += len(seqs)
    return counter


def load_model_config(config_file):
    settings = {}
    with open(config_file, 'r') as reader:
        for line in reader.readlines():
            parts = line.strip().split("=")
            if len(parts) == 2:
                settings[parts[0]] = parts[1]
    return settings


def get_predicted_sequence(predictions, data_info, groundtruth_len, threshold=-1):
    '''
    :param predictions: {id: [predicted values in different sequences]}
    :param data_info: OrderDict {utter_id: (text, score, startframe, endframe)}
    :param threshold: minimum changes to make a change (< 0: always change)
    :return: predicted sequence
    '''
    sorted_indices = data_info.keys()
    prev_predicted = 0.0  # previous predicted value

    sequence = []
    first_value = True
    for index in sorted_indices:
        tmp = data_info[index]
        start_index = tmp[2]  # start frame
        end_index = tmp[3]  # end frame
        if index not in predictions:
            # none
            add_value_to_sequence(prev_predicted, sequence, start_index, end_index)
            continue
        # use average value
        # avg = np.mean(predictions[index])
        new_score = np.mean(predictions[index])
        if first_value:
            first_value = False
        elif threshold > 0:
            # require threshold for changing
            if abs(new_score - prev_predicted) < threshold:
                # make no change
                new_score = prev_predicted
        add_value_to_sequence(new_score, sequence, start_index, end_index)
        prev_predicted = new_score

    # fill the sequence to match with the groundtruth's
    sequence = refine_predicted_sequence(sequence, groundtruth_len)

    return sequence


def get_predicted_sequence_difference(predictions, data_info, groundtruth_len, threshold=-1):
    '''
    get predicted sequence for the output from diff-model, i.e., training on differences
    :param predictions: {id: [predicted values in different sequences]}
    :param data_info: OrderDict {utter_id: (text, score, startframe, endframe)}
    :param threshold: minimum changes to make a change (< 0: always change)
    :return: predicted sequence
    '''
    sorted_indices = data_info.keys()
    prev_predicted = 0.0  # previous predicted value
    actual_value = 0.0  # the actual value when there's no threshold
    sequence = []
    first_value = True
    for index in sorted_indices:
        tmp = data_info[index]
        start_index = tmp[2]  # start frame
        end_index = tmp[3]  # end frame
        if index not in predictions:
            # none
            add_value_to_sequence(prev_predicted, sequence, start_index, end_index)
            continue
        # use average value
        diff = np.mean(predictions[index])
        actual_value += diff  # keep track of the original predicted value

        new_score = actual_value

        if first_value:
            first_value = False
        elif threshold > 0:
            # require threshold for changing
            if abs(actual_value - prev_predicted) < threshold:
                # make no change
                new_score = prev_predicted

        # new_score = prev_predicted + diff
        add_value_to_sequence(new_score, sequence, start_index, end_index)
        prev_predicted = new_score

    # fill the sequence to match with the groundtruth's
    sequence = refine_predicted_sequence(sequence, groundtruth_len)

    return sequence


# for creating raw average word embeddings for utterances
def load_utterances_with_timing(input_file):
    utters = {}
    with open(input_file, 'r') as reader:
        data = json.load(reader)
        for utter in data:
            utters[utter['id']] = (utter['text'], utter['start_time'], utter['end_time'])
    return utters


def get_avg_embeddings(utterances, vocabs, word_embeddings, stopWords=None):
    '''
    :param utterances: list of [utterance] <- [(id, text, start time, end time)]
    :param vocabs:
    :param word_embeddings:
    :return: dictionary of list of tensors for each utterance
    '''
    tmp = {}
    valid_indices = []
    for key, utter in utterances.items():
        utter_vec = get_avg_embeddings_for_utter(utter[0], vocabs, word_embeddings, stopWords)
        if utter_vec is not None:
            tmp[key] = utter_vec
            valid_indices.append(key)
    return tmp, valid_indices


def get_avg_embeddings_for_utter(utter, vocabs, word_embeddings, stopWords):
    words = word_tokenize(utter.lower())
    tmp = []
    for word in words:
        if word in vocabs:
            if stopWords is not None and word in stopWords:
                continue
            tmp.append(word_embeddings[word])
        # else:
        #     print("Cannot find embeddings for:", word)
    if tmp:
        tmp = np.average(np.stack(tmp), axis=0)
        return tmp
    else:
        return None


# preprocess the input tensor, one file is one sequence of all utterances

def get_video_tensor_for_file(data_one_file, vocabs, word_embeddings, stopwords):
    '''
    :param data_one_file: list of (utterance, score)
    :param vocabs:
    :param word_embeddings:
    :return: ordereddict of list of tensors for each utterance
    '''
    tmp = OrderedDict()
    for index, utter in data_one_file.items():
        torch_tensor = get_tensor_for_utter(utter[0], vocabs, word_embeddings, stopwords)
        if torch_tensor is not None:
            tmp[index] = (torch_tensor, torch.Tensor([utter[1]]))
        else:
            tmp[index] = (None, None)  # to keep continuity
    return tmp


def get_video_tensor(input_data, vocabs, word_embeddings, stopwords=None):
    '''
    :param input_data: {file: [(text, average score)]}
    :param stopwords: if stopwords is not None --> remove stopwords
    :return: {file: (input_tensor, output_tensor)}
    '''
    tensor_data = {}
    for fname, data in input_data.items():
        tensor_data[fname] = get_video_tensor_for_file(data, vocabs, word_embeddings, stopwords)
    return tensor_data


# each utterance = average embeddings of the (non-stop) words in the utterance
def get_utter_list(utters_dict):
    '''

    :param utters_dict: OrderedDict
    :return:
    '''
    utters = []
    for key, value in utters_dict.items():
        utters.append((key, value[0], value[1]))
    return utters


def get_average_word_embeddings_for_utters(data, vocabs, word_embeddings, en_stopwords=None):
    '''

    :param data: {filename: (features, values, valid_indices)}
    :param vocabs:
    :param word_embeddings:
    :param en_stopwords:
    :return:
    '''
    avg_we = {}
    for fname, data in data.items():
        utters = get_utter_list(data)
        features = []
        values = []
        valid_indices = []  # utterance index that has embeddings
        for utter in utters:
            utter_vec = get_avg_embeddings_for_utter(utter[1], vocabs, word_embeddings, en_stopwords)
            if utter_vec is not None:
                features.append(torch.from_numpy(utter_vec))
                values.append(torch.Tensor([utter[2]]))
                valid_indices.append(utter[0])

        features = torch.stack(features)
        values = torch.stack(values)
        avg_we[fname] = (features, values, valid_indices)
    return avg_we

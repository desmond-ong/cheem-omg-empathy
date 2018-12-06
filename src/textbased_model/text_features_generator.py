'''
Generate raw text features for fusion network
'''

from torch import optim, nn
import sys
from gensim.models import KeyedVectors
import numpy as np
import torch
import os
import time
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import utils, config
from src.sample import calculateCCC
from src.textbased_model.models import LSTMEmo_duo_sequence
# from importlib import reload
import argparse
import math
import re
from nltk.corpus import stopwords

word_embedding_size = 300
fps = 25


def get_utter_index_for_second(second, valid_utter_index, valid_indices, untterances):
    if valid_utter_index >= len(valid_indices):
        # last valid utterance already
        return len(valid_indices) - 1

    utter = untterances[valid_indices[valid_utter_index]]
    end_second = utils.get_seconds(utter[2])  # seconds of the ending
    if end_second > second:
        return valid_utter_index  # satisfy
    else:
        return get_utter_index_for_second(second, valid_utter_index + 1, valid_indices, untterances)


def generate_features_for(data_source, word_embeddings, output_dir_name, stopWords):
    print("PROCESSING FOR ", data_source)
    root = "data/{}/".format(data_source)
    output_dir = "{}{}/".format(root, output_dir_name)
    utils.mkdir(output_dir)
    input_dir = '{}transcripts_with_scores/'.format(root)
    vocabs = word_embeddings.wv.vocab
    # print("Loading groundtruth sequences...")
    gt_sequences = utils.load_groundtruth_sequences("{}Annotations/".format(root))  # (only to know the length of the sequence )

    for filename in os.listdir(input_dir):
        if filename.startswith(".") or not filename.endswith("json"):
            continue
        # if filename != "Subject_6_Story_2.json":
        #     continue
        print("Processing for", filename)
        utterances = utils.load_utterances_with_timing(os.path.join(input_dir, filename))  # {id: (text, start time, end time)}
        file_id = filename[:-5]
        output_file = "{}{}.npy".format(output_dir, file_id)

        avg_embeddings, valid_indices = utils.get_avg_embeddings(utterances, vocabs, word_embeddings, stopWords)  # {id: avg_word_embeddings}

        seconds = math.ceil(len(gt_sequences[file_id])/fps)

        index = 0  # index of valid indices

        vectors = []
        for second in range(seconds):
            index = get_utter_index_for_second(second, index, valid_indices, utterances)
            vec = avg_embeddings[valid_indices[index]]
            vectors.append(vec)
        vectorsnp = np.stack(vectors)
        # np.savetxt(output_file, vectorsnp)
        np.save(output_file, vectorsnp)


def generate_features_with_len_info(data_source, word_embeddings, output_dir_name, stopWords, num_frames):
    print("PROCESSING FOR ", data_source)
    root = "data/{}/".format(data_source)
    output_dir = "{}{}/".format(root, output_dir_name)
    utils.mkdir(output_dir)
    input_dir = '{}transcripts_with_scores/'.format(root)
    vocabs = word_embeddings.wv.vocab
    # print("Loading groundtruth sequences...")

    for filename in os.listdir(input_dir):
        if filename.startswith(".") or not filename.endswith("json"):
            continue
        # if filename != "Subject_6_Story_2.json":
        #     continue
        print("Processing for", filename)
        utterances = utils.load_utterances_with_timing(os.path.join(input_dir, filename))  # {id: (text, start time, end time)}
        file_id = filename[:-5]
        output_file = "{}{}.npy".format(output_dir, file_id)

        avg_embeddings, valid_indices = utils.get_avg_embeddings(utterances, vocabs, word_embeddings, stopWords)  # {id: avg_word_embeddings}

        seconds = math.ceil(num_frames[file_id]/fps)

        index = 0  # index of valid indices

        vectors = []
        for second in range(seconds):
            index = get_utter_index_for_second(second, index, valid_indices, utterances)
            vec = avg_embeddings[valid_indices[index]]
            vectors.append(vec)
        vectorsnp = np.stack(vectors)
        # np.savetxt(output_file, vectorsnp)
        np.save(output_file, vectorsnp)


def load_num_frames(frame_count_file):
    pattern = "(.*).csv - (\d*) Frames"
    parser = re.compile(pattern)
    frames = {}
    with open(frame_count_file, 'r') as reader:
        lines = reader.read().split("\n")
    for line in lines:
        line = line.strip()
        if line != "":
            results = parser.search(line)
            frames[results.group(1)] = int(results.group(2))
    return frames


if __name__ == "__main__":
    stopWords = set(stopwords.words('english'))

    output_dir_name = "Avg_Text_Raw_Features"

    word_embeddings_file = config.glove_word2vec_file_filtered
    start = time.time()
    print("Loading word embeddings...")
    word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
    vocabs = word_embeddings.wv.vocab
    utils.print_time(start)

    # training_start = time.time()
    # generate_features_for("Traing", word_embeddings, output_dir_name, stopWords)
    # print("Time: {}".format(utils.time_since(training_start)))

    # val_start = time.time()
    # generate_features_for("Validation", word_embeddings, output_dir_name, stopWords)
    # print("Time for validation: {}".format(utils.time_since(val_start)))

    frame_count_file = "data/Testing/Frames_Count.txt"
    frames = load_num_frames(frame_count_file)
    for key, value in frames.items():
        print("{}\t{}".format(key, value))
    generate_features_with_len_info("Testing", word_embeddings, output_dir_name, stopWords, frames)
    #
    # f = '/Users/sonnguyen/raid/data/Training/Avg_Visual_Features/Subject_6_Story_2.npy'
    # a = np.load(f)
    # print(a.shape)
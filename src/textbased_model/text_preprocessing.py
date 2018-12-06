import os
import sys
import pandas
sys.path.append("")  # append the path to this project!
from src.textbased_model import utils
from src.textbased_model.obj_utterance import Utterance, UtteranceDiff
import numpy as np
import json


root_dir = "data/"

diff = 1e-6

def analyze_timing(subtitles, annotations):
    utterances = []
    for sub in subtitles:
        id = sub[0]
        start_time = sub[1]
        end_time = sub[2]
        text = sub[3]
        start_frame = utils.time_to_frame(start_time)
        end_frame = utils.time_to_frame(end_time)
        if end_frame > len(annotations):
            end_frame = len(annotations)
        scores = annotations[start_frame:end_frame]
        mean = np.mean(scores)

        utter = Utterance(id, start_time, end_time, start_frame, end_frame, text, scores, mean)
        utterances.append(utter)
    return utterances


def transcript_scores_mapping(data_source):
    subtitle_dir = os.path.join(root_dir, data_source, "Transcripts_Youtube")
    annotation_dir = os.path.join(root_dir, data_source, "Annotations")
    output_dir = os.path.join(root_dir, data_source, "transcripts_with_scores")
    utils.mkdir(output_dir)

    for fname in os.listdir(subtitle_dir):
        id = fname[:-4]
        print("Processing for ", id)
        subtitle_file = os.path.join(subtitle_dir, fname)
        anno_file = os.path.join(annotation_dir, "{}.csv".format(id))
        output_file = os.path.join(output_dir, "{}.json".format(id))
        subtitles = utils.load_subtitles(subtitle_file)
        annotations = utils.load_annotations(anno_file)
        utters = analyze_timing(subtitles, annotations)
        with open(output_file, 'w') as writer:
            json.dump([obj.__dict__ for obj in utters], writer, indent=3)


def compute_score_difference(data_source):
    scores = os.path.join(root_dir, data_source, "transcripts_with_scores")
    output = os.path.join(root_dir, data_source, "transcripts_with_scores_difference")
    utils.mkdir(output)

    for filename in os.listdir(scores):
        if filename.startswith("."):
            continue
        if not filename.endswith(".json"):
            continue
        input_file = os.path.join(scores, filename)
        output_file = os.path.join(output, filename)
        utters = []
        with open(input_file, 'r') as reader:
            prev = 0.0
            data = json.load(reader)
            for utter in data:
                difference = utter['average'] - prev
                if abs(difference) < diff:
                    difference = 0
                new_utter = UtteranceDiff(utter['id'], utter['start_time'], utter['end_time'], utter['start_frame'],
                                          utter['end_frame'], utter['text'], difference)

                prev = utter['average']
                utters.append(new_utter)

        with open(output_file, 'w') as writer:
            json.dump([utt.__dict__ for utt in utters], writer, indent=3)


def analyze_timing_subtitles_only(subtitles):
    utterances = []
    for sub in subtitles:
        id = sub[0]
        start_time = sub[1]
        end_time = sub[2]
        text = sub[3]
        start_frame = utils.time_to_frame(start_time)
        end_frame = utils.time_to_frame(end_time)

        utter = Utterance(id, start_time, end_time, start_frame, end_frame, text, None, None)
        utterances.append(utter)
    return utterances


def transcript_to_json(data_source):
    '''
    for test set that doesn't have annotations
    :param data_source:
    :return:
    '''
    subtitle_dir = os.path.join(root_dir, data_source, "Transcripts_Youtube")
    output_dir = os.path.join(root_dir, data_source, "transcripts_with_scores")
    utils.mkdir(output_dir)

    for fname in os.listdir(subtitle_dir):
        id = fname[:-4]
        print("Processing for ", id)
        subtitle_file = os.path.join(subtitle_dir, fname)
        output_file = os.path.join(output_dir, "{}.json".format(id))
        subtitles = utils.load_subtitles(subtitle_file)
        utters = analyze_timing_subtitles_only(subtitles)
        with open(output_file, 'w') as writer:
            json.dump([obj.__dict__ for obj in utters], writer, indent=3)


# transcript_scores_mapping("Training")
# transcript_scores_mapping("Validation")
compute_score_difference("Validation")
transcript_to_json("Testing")
'''
preprocessing steps
'''

import os
import numpy as np
import json
from src.text_based_submit import utils

root_dir = "data/"
diff = 1e-6


class Utterance():
    def __init__(self, id, start_time, end_time, start_frame, end_frame, text, scores, average):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.text = text
        self.scores = scores
        self.average = average


class UtteranceDiff():
    def __init__(self, id, start_time, end_time, start_frame, end_frame, text, average):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.text = text
        self.average = average  # this is indeed difference, keep same name --> no need to change utils functions


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


def transcript_scores_mapping(data_source, transcript_dirname, annotation_dirname, output_dirname):
    subtitle_dir = os.path.join(root_dir, data_source, transcript_dirname)
    annotation_dir = os.path.join(root_dir, data_source, annotation_dirname)
    output_dir = os.path.join(root_dir, data_source, output_dirname)
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


def transcript_to_json(data_source, transcript_dirname, output_dir_name):
    '''
    for test set that doesn't have annotations
    :param data_source:
    :return:
    '''
    subtitle_dir = os.path.join(root_dir, data_source, transcript_dirname)
    output_dir = os.path.join(root_dir, data_source, output_dir_name)
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


def compute_score_difference(data_source, scores_dirname, difference_dirname):
    scores = os.path.join(root_dir, data_source, scores_dirname)
    output = os.path.join(root_dir, data_source, difference_dirname)
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


def map_transcripts_to_scores():
    sources_with_annotations = ["Training", "Validation"]
    for data_source in sources_with_annotations:
        transcript_scores_mapping(data_source, youtube_transcript_dirname, annotation_dirname, output_dirname)


def transcripts_to_json_no_scores():
    transcript_to_json("Testing", youtube_transcript_dirname, output_dirname)


def scores_to_differences():
    scores_dirname = "transcripts_with_scores"
    difference_dirname = "transcripts_with_scores_difference"
    compute_score_difference("Training", scores_dirname, difference_dirname)
    compute_score_difference("Validation", scores_dirname, difference_dirname)


if __name__ == "__main__":
    # generate json files: mapping youtube transcripts (utterances) with annotations (Training, Validation)
    youtube_transcript_dirname = "Transcripts_Youtube"
    annotation_dirname = "Annotations"
    output_dirname = "transcripts_with_scores"

    # map the transcripts with annotations scores
    # map_transcripts_to_scores()

    # generate json files: when annotations are not available (Testing)
    # transcripts_to_json_no_scores()

    # generate json files: scores are the changes of continuous utterances' averages
    # scores_to_differences()



'''
Compute pairwise ccc for different models' predictions
'''
import os
from src.textbased_model import utils
import pandas
from src.sample import calculateCCC
import numpy as np

def load_valence(valence_file):
    data = pandas.read_csv(valence_file)
    return list(data['valence'])


def compute_pairwise_ccc(dir1, dir2):
    ccc_scores = {}
    pearson_scores = {}
    for filename in os.listdir(dir1):
        if filename.startswith("."):
            continue
        fileid = filename[:-4]
        list1 = load_valence(os.path.join(dir1, filename))
        list2 = load_valence(os.path.join(dir2, filename))
        ccc, pearson = calculateCCC.ccc(list1, list2)
        ccc_scores[fileid] = ccc
        pearson_scores[fileid] = pearson
    return ccc_scores, pearson_scores


def write_pairwise_ccc(scores, output_file):
    writer = open(output_file, 'w')
    writer.write("Model1\tModel2\tCCC\tPearson\n")
    # write average
    for model1, tmp in scores.items():
        for model2, tmp2 in tmp.items():
            ccc_scores = tmp2[0]
            pearson_scores = tmp2[1]
            writer.write("{}\t{}\t{}\t{}\n".format(prediction_dirs[model1], prediction_dirs[model2],
                                                   np.mean(list(ccc_scores.values())), np.mean(list(pearson_scores.values()))))

    writer.write("\n\n@Details\n")
    for model1, tmp in scores.items():
        for model2, tmp2 in tmp.items():
            writer.write("\n\n\n@\t{}\t{}\n\n".format(prediction_dirs[model1], prediction_dirs[model2]))
            writer.write("Video\tCCC\tPearson\n")
            for filename, ccc_score in tmp2[0].items():
                writer.write("{}\t{}\t{}\n".format(filename, ccc_score, tmp2[1][filename]))

    writer.flush()
    writer.close()

prediction_dirs = ["(G1) Predictions_AT", "(G2) Predictions_T1", "(G3) Predictions_ATV", "Predictions_Visual"]
root_dir = "/Users/sonnguyen/raid/data/Testing/"
output_file = "/Users/sonnguyen/raid/data/Testing/pairwise_ccc.txt"

num = len(prediction_dirs)

scores = {}  # {file1: {file2: {video_name: pearson, ccc}}}

for i in range(num):
    tmp = {}
    for j in range(i + 1, num):
        dir1 = os.path.join(root_dir, prediction_dirs[i])
        dir2 = os.path.join(root_dir, prediction_dirs[j])
        ccc_scores, pearson_scores = compute_pairwise_ccc(dir1, dir2)
        tmp[j] = (ccc_scores, pearson_scores)
    scores[i] = tmp

# write scores
write_pairwise_ccc(scores, output_file)
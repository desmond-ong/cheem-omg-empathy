'''
Compute average changes for the groundtruth
'''

import pandas
import os
anno_dir = "/Users/sonnguyen/raid/data/Training/Annotations/"
t = 1e-6

total_jump = 0.0
count = 0
total_frames = 0

for filename in os.listdir(anno_dir):
    total_jump = 0.0
    count = 0
    total_frames = 0
    if filename.startswith(".") or not filename.endswith("csv"):
        continue
    data = pandas.read_csv(os.path.join(anno_dir, filename))
    prev = 0
    first = True
    for val in data['valence']:
        total_frames += 1
        if first:
            first = False
            continue
        diff = abs(prev - val)
        prev = val
        if diff > t:
            # there is a difference
            total_jump += diff
            count += 1

    print("{}\tAvgValenceChange:\t{:.3f}\t\t#Changes:\t{}\tTotalFrames:\t{}".format(filename, total_jump/count, count, total_frames))



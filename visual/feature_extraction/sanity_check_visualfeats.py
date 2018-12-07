## Script for calculating features equal to the number of frames ##


import numpy as np
import os
from itertools import repeat
import pandas as pd
pattern = "Subject_(\d+)_Story_(\d+)(?:_\w*)?"
base_dir = './data/Testing/'   ##change the base directory for train, val and test

feats_dir = os.path.join(base_dir, 'Avg_Visual_Features/Subject')
frames_count_file = os.path.join(base_dir,'Frames_Count.txt')    ###Either load frame count from txt file or load Annotations csv to check the frame length
## Load frame counts from txt file

file_writer = open(frames_count_file,'r')
csv_file=[]
frame_count =[]
for line in file_writer:
    row = str.split(line,' ')
    print(row)
    if (row[0]!= '\n'):
        csv_file.append(row[0])
        frame_count.append(row[2])

####################################################################
## Load frame count from annotations, if annotations present

"""
frame_count= []
annotations_csv = os.path.join(base_dir,'Annotations')
csv_files = [os.path.join(annotations_csvr, fn) for fn in sorted(os.listdir(annotations_csv)) if re.match(pattern, fn) is not None]
for ann in csv_files:
	valence = ps.read_csv(ann)
	frame_count.append(len(valence))
"""
####################################################################
avg_frames = [int(frame_count[i])/25 for i in range(len(frame_count))]
npy_files = os.listdir(feats_dir)

def main():

	for file,count in zip(csv_file,avg_frames):
	    idx = [i for i , s in enumerate(npy_files) if s[0:len(s)-4] in file[0:len(file)-4]]

	    feat_dir = os.path.join(feats_dir,npy_files[idx[0]])
	    feat_file = np.load(os.path.join(feats_dir,npy_files[idx[0]]))
	    feats_shape = feat_file.shape

	    if(feats_shape[0] != count):
		feat_file = np.squeeze(feat_file,1)
		feats= feat_file[feats_shape[0]-1]
		list(feat_file).extend(repeat(feats, (count+1)-feats_shape[0]))
	    feat_file = np.delete(feat_file,feats_shape[0]-1,0)
	    ### Save final corrected features
	    np.save(feat_dir,feat_file)

if __name__ == '__main__':
    main()

## Script for averaging raw visual features over 1 second (25 frames) from extract_face_features.py ##

## Some Frames have missed face detections and therefore with less features in the dir Visual_Features/ ##
## After running this script, replicate missed features, if any, by running sanity_check_visualfeats.py ##

from __future__ import division

import numpy as np
import os, sys
base_dir = './data/Testing'    ##change the base directory for train, val and test

feats_root_dir = os.path.join(base_dir, 'Visual_Features/Subject')
feats_dirs = os.listdir(feats_root_dir)
output_dir = os.path.join(base_dir, 'Avg_Visual_Features/Subject')

def main():
	for i in range(len(feats_dirs)):
	    total_feats =[]
	    feats_video_file = feats_dirs[i]
	    feats_video_dir = os.path.join(feats_root_dir,feats_video_file)
	    feats_frame_list = os.listdir(feats_video_dir)
	    feats_frame_list.sort()
	    for k in range(len(feats_frame_list)):
		feats = np.load(os.path.join(feats_video_dir,feats_frame_list[k]))
		total_feats.append(feats)
	    final_mean_feats =[]
	    steps = len(total_feats)/25
	    if (len(total_feats) % 25 == 0):
		print('25 frames per second complete')
		for j in range(int(steps)):
		    seq_len = 25
		    mean_feats =np.mean(total_feats[j*seq_len:(seq_len*(j+1))-1], 0)

		    final_mean_feats.append(mean_feats)
	    else:
		print('Incomplete remaining frames')
		rem = len(total_feats) % 25
		for j in range(int(steps)):
		    seq_len = 25
		    mean_feats =np.mean(total_feats[j*seq_len:(seq_len*(j+1))-1], 0)
		    final_mean_feats.append(mean_feats)

		mean_feats = np.mean(total_feats[len(total_feats)-rem : len(total_feats)], 0)
	    final_mean_feats.append(mean_feats)

	    ## save average visual features 
	    feats_name = feats_video_file+'.npy'
	    np.save(os.path.join(output_dir,feats_name), final_mean_feats)

if __name__ == '__main__':
    main()

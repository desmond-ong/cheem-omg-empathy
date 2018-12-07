## DataLoader Script for loading visual features for OMG Empathy dataset ##

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import re
pattern = "Subject_(\d+)_Story_(\d+)(?:_\w*)?"

class OMGEmpathyDataset(Dataset):
    def __init__(self, csv_dir, root_dir, batch,transform=None):
        self.root_dir = root_dir
        self.csv_dir = csv_dir
        self.transform = transform
        self.batch_size = batch
        visual_feats_file = [os.path.join(self.root_dir, fn) for fn in sorted(os.listdir(self.root_dir))]
        valence_file = [os.path.join(self.csv_dir, fn) for fn in sorted(os.listdir(self.csv_dir)) if re.match(pattern, fn) is not None]
        #visual_feats_file = self.root_dir
        #valence_file = self.csv_dir
        self.visual_data = []
        self.valence_data = []
        self.n_frames=[]
        self.csv_file_name = []
        self.ratio =25
        for vi, va in zip(visual_feats_file, valence_file):
	    # Load valence ratings and visual features (avg over 1 second time frame)
            visual = np.load(vi)
            valence = pd.read_csv(va)

            self.n_frames.append(len(valence))\
            group_idx = np.arange(len(valence)) // self.ratio
            valence = np.array(valence.groupby(group_idx).mean())
            visual, valence = [visual], [valence]
            self.visual_data += visual
            self.valence_data += valence
            self.csv_file_name.append(vi)

        self.visual_data = np.concatenate(self.visual_data)
        self.valence_data = np.concatenate(self.valence_data)

        length = self.visual_data.shape[0]
        rem = self.visual_data.shape[0] % self.batch_size
        self.visual_data = self.visual_data[0: length - rem]
        self.valence_data = self.valence_data[0:length - rem]


    def __len__(self):
        return len(self.valence_data)

    def __getitem__(self, idx):
        return self.visual_data[idx], self.valence_data[idx]


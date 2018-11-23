from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def collate_fn(data):
    """Collates variable length sequences into padded batch tensor."""
    def merge(sequences):
        dims = sequences[0].shape[1]
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths), dims)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end, :] = torch.from_numpy(seq[:end,:])
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    x_seqs, y_seqs = zip(*data)
    x, x_lengths = merge(x_seqs)
    y, y_lengths = merge(y_seqs)
    if x_lengths != y_lengths:
        print("Warning: input and output have different sequence lengths.")
        lengths = [max(xl, yl) for xl, yl in zip(x_lengths, y_lengths)]
    else:
        lengths = x_lengths
    return x, y, lengths

class OMGopenSMILE(Dataset):

    def __init__(self, folder, transform=None,
                 pattern="Subject_(\d+)_Story_(\d+)(?:_\w*)?.csv",
                 valence_folder="Annotations", openSMILE_folder="OpenSMILE",
                 fps = 25.0, chunk_dur=0.5):
        self.folder = folder
        self.pattern = pattern
        self.ratio = fps * chunk_dur
        self.valence_folder = os.path.join(folder, valence_folder)
        self.openSMILE_folder = os.path.join(folder, openSMILE_folder)
        # Load files into list
        self.valence_files = [os.path.join(self.valence_folder, fn) for fn
                              in sorted(os.listdir(self.valence_folder))
                              if re.match(pattern, fn) is not None]
        self.openSMILE_files = [os.path.join(self.openSMILE_folder, fn) for fn
                                in sorted(os.listdir(self.openSMILE_folder))
                                if re.match(pattern, fn) is not None]
        # Check that number of files are equal
        if len(self.valence_files) != len(self.openSMILE_files):
            raise Exception("Number of files do not match.")
        # Load dataframes for each video
        self.valence_dfs = []
        self.openSMILE_dfs = []
        for f1, f2 in zip(self.valence_files, self.openSMILE_files):
            valence_df = pd.read_csv(f1)
            openSMILE_df = pd.read_csv(f2)
            # Average valence ratings across time chunks
            group_idx = np.arange(len(valence_df)) // self.ratio
            valence_df = valence_df.groupby(group_idx).mean()
            # Append dataframe
            self.valence_dfs.append(valence_df)
            self.openSMILE_dfs.append(openSMILE_df)

    def __len__(self):
        return len(self.valence_dfs)

    def __getitem__(self, idx):
        data = np.array(self.openSMILE_dfs[idx])
        target = np.array(self.valence_dfs[idx])
        return data, target
    
if __name__ == "__main__":
    # Test code by loading dataset
    print("Loading data...")
    dataset = OMGopenSMILE("./data/Training")
    data, target = dataset[0]
    print("Shape of first sample:")
    print(data.shape, target.shape)

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
    def merge(sequences, max_len=None):
        dims = sequences[0].shape[1]
        lengths = [len(seq) for seq in sequences]
        if max_len is None:
            max_len = max(lengths)
        padded_seqs = torch.zeros(len(sequences), max_len, dims)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end, :] = torch.from_numpy(seq[:end,:])
        return padded_seqs

    padded = []
    lengths = np.zeros(len(data), dtype=int)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data = zip(*data)
    for modality in data:
        m_lengths = [len(seq) for seq in modality]
        lengths = np.maximum(lengths, m_lengths)
    lengths = list(lengths)
    for modality in data:
        padded.append(merge(modality, max(lengths)))
    return tuple(padded + [lengths])

class OMGcombined(Dataset):
    """Dataset that loads features for each modality and valence ratings.
    
    audio_path -- Folder of OpenSMILE features in CSV format
    text_path -- Folder of text features in NPY format
    visual_path -- Folder of visual features in NPY format (extra dim=1)
    """
    
    def __init__(self, audio_path, text_path, visual_path, valence_path,
                 transform=None, pattern="Subject_(\d+)_Story_(\d+)(?:_\w*)?",
                 fps = 25.0, chunk_dur=1.0):
        self.audio_path = audio_path
        self.text_path = text_path
        self.visual_path = visual_path
        self.valence_path = valence_path
        self.pattern = pattern
        self.ratio = fps * chunk_dur

        # Load files into list
        audio_files = [os.path.join(audio_path, fn) for fn
                       in sorted(os.listdir(self.audio_path))
                       if re.match(pattern, fn) is not None]
        text_files = [os.path.join(text_path, fn) for fn
                      in sorted(os.listdir(self.text_path))
                      if re.match(pattern, fn) is not None]
        visual_files = [os.path.join(visual_path, fn) for fn
                        in sorted(os.listdir(self.visual_path))
                        if re.match(pattern, fn) is not None]
        valence_files = [os.path.join(valence_path, fn) for fn
                         in sorted(os.listdir(self.valence_path))
                         if re.match(pattern, fn) is not None]

        # Check that number of files are equal
        if not (len(audio_files) == len(text_files) and
                len(text_files) == len(visual_files) and
                len(visual_files) == len(valence_files)):
            raise Exception("Number of files do not match.")

        # Store subject and story IDs
        self.subjects, self.stories = [], []
        for fn in sorted(os.listdir(self.valence_path)):
            match = re.match(pattern, fn)
            if match:
                self.subjects.append(match.group(1))
                self.stories.append(match.group(2))
        
        # Load data for each video
        self.audio_data = []
        self.text_data = []
        self.visual_data = []
        self.valence_data = []
        for f_au, f_te, f_vi, f_va in zip(audio_files, text_files,
                                          visual_files, valence_files):
            # Load each input modality
            self.audio_data.append(np.array(pd.read_csv(f_au)))
            self.text_data.append(np.load(f_te))
            self.visual_data.append(np.load(f_vi).squeeze(1))
            # Load valence ratings and average across time chunks
            valence = pd.read_csv(f_va)
            group_idx = np.arange(len(valence)) // self.ratio
            valence = np.array(valence.groupby(group_idx).mean())
            self.valence_data.append(valence)

    def __len__(self):
        return len(self.valence_data)

    def __getitem__(self, i):
        return (self.audio_data[i], self.text_data[i],
                self.visual_data[i], self.valence_data[i])
    
if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="./data/Training",
                        help='dataset base folder')
    args = parser.parse_args()

    audio_path = os.path.join(args.folder, "CombinedAudio")
    text_path = os.path.join(args.folder, "CombinedText")
    visual_path = os.path.join(args.folder, "CombinedVisual")
    valence_path = os.path.join(args.folder, "Annotations")

    print("Loading data...")
    dataset = OMGcombined(audio_path, text_path, visual_path, valence_path)
    print("Testing batch collation...")
    data = collate_fn([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-1]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        audio, text, visual, valence = data
        print("#{}\tSubject: {}\tStory: {}:".\
              format(i+1, dataset.subjects[i], dataset.stories[i]))
        print(audio.shape, text.shape, visual.shape, valence.shape)
        if not (len(audio) == len(text) and
                len(text) == len(visual) and
                len(visual) == len(valence)):
            print("WARNING: Mismatched sequence lengths.")

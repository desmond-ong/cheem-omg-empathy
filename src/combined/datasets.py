from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def len_to_mask(lengths):
    """Converts list of sequence lengths to a mask tensor."""
    mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths))
    mask = mask < torch.tensor(lengths).unsqueeze(1)
    return mask

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
    
    def __init__(self, audio_path, text_path, v_sub_path, v_act_path,
                 val_path, pattern="Subject_(\d+)_Story_(\d+)(?:_\w*)?",
                 fps=25.0, chunk_dur=1.0, split_ratio=1, truncate=False):
        self.audio_path = audio_path
        self.text_path = text_path
        self.v_sub_path = v_sub_path
        self.v_act_path = v_act_path
        self.val_path = val_path
        self.pattern = pattern
        self.time_ratio = fps * chunk_dur
        self.split_ratio = split_ratio

        # Load files into list
        audio_files = [os.path.join(audio_path, fn) for fn
                       in sorted(os.listdir(self.audio_path))
                       if re.match(pattern, fn) is not None]
        text_files = [os.path.join(text_path, fn) for fn
                      in sorted(os.listdir(self.text_path))
                      if re.match(pattern, fn) is not None]
        v_sub_files = [os.path.join(v_sub_path, fn) for fn
                        in sorted(os.listdir(self.v_sub_path))
                        if re.match(pattern, fn) is not None]
        v_act_files = [os.path.join(v_act_path, fn) for fn
                        in sorted(os.listdir(self.v_act_path))
                        if re.match(pattern, fn) is not None]
        val_files = [os.path.join(val_path, fn) for fn
                         in sorted(os.listdir(self.val_path))
                         if re.match(pattern, fn) is not None]

        # Check that number of files are equal
        if not (len(audio_files) == len(text_files) and
                len(text_files) == len(v_sub_files) and
                len(v_sub_files) == len(v_act_files) and
                len(v_act_files) == len(val_files)):
            raise Exception("Number of files do not match.")

        # Store subject and story IDs
        self.subjects = []
        self.stories = []
        for fn in sorted(os.listdir(self.val_path)):
            match = re.match(pattern, fn)
            if match:
                self.subjects.append(match.group(1))
                self.stories.append(match.group(2))
        
        # Load data for each video
        self.audio_data = []
        self.text_data = []
        self.v_sub_data = []
        self.v_act_data = []
        self.val_data = []
        self.val_orig = []
        for f_au, f_te, f_vs, f_va, f_vl in \
            zip(audio_files, text_files, v_sub_files, v_act_files, val_files):
            # Load each input modality
            audio = np.array(pd.read_csv(f_au))
            text = np.load(f_te)
            v_sub = np.load(f_vs).squeeze(1)
            v_act = np.load(f_va).squeeze(1)
            # Load and store original valence ratings
            val = pd.read_csv(f_vl)
            self.val_orig.append(np.array(val).flatten())
            # Average valence across time chunks
            group_idx = np.arange(len(val)) // self.time_ratio
            val = np.array(val.groupby(group_idx).mean())
            # Truncate to minimum sequence length
            if truncate:
                seq_len =\
                    min([len(d) for d in [audio, text, v_sub, v_act, val]])
                audio = audio[:seq_len]
                text = text[:seq_len]
                v_sub = v_sub[:seq_len]
                v_act = v_act[:seq_len]
                val = val[:seq_len]
            # Split data to create more examples
            if split_ratio > 1:
                audio = np.array_split(audio, split_ratio, axis=0)
                text = np.array_split(text, split_ratio, axis=0)
                v_sub = np.array_split(v_sub, split_ratio, axis=0)
                v_act = np.array_split(v_act, split_ratio, axis=0)
                val = np.array_split(val, split_ratio, axis=0)
            else:
                audio, text, v_sub, v_act, val =\
                    [audio], [text], [v_sub], [v_act], [val]
            self.audio_data += audio
            self.text_data += text
            self.v_sub_data += v_sub
            self.v_act_data += v_act
            self.val_data += val

    def __len__(self):
        return len(self.val_data)

    def __getitem__(self, i):
        return (self.audio_data[i], self.text_data[i],
                self.v_sub_data[i], self.v_act_data[i], self.val_data[i])
    
if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="./data/Training",
                        help='dataset base folder')
    args = parser.parse_args()

    audio_path = os.path.join(args.folder, "CombinedAudio")
    text_path = os.path.join(args.folder, "CombinedText")
    v_sub_path = os.path.join(args.folder, "CombinedVSub")
    v_act_path = os.path.join(args.folder, "CombinedVAct")
    val_path = os.path.join(args.folder, "Annotations")

    print("Loading data...")
    dataset = OMGcombined(audio_path, text_path,
                          v_sub_path, v_act_path, val_path)
    print("Testing batch collation...")
    data = collate_fn([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-1]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        audio, text, v_sub, v_act, val = data
        print("#{}\tSubject: {}\tStory: {}:".\
              format(i+1, dataset.subjects[i], dataset.stories[i]))
        print(audio.shape, text.shape, v_sub.shape, v_act.shape, val.shape)
        if not (len(audio) == len(text) and
                len(text) == len(v_sub) and
                len(v_sub) == len(v_act) and
                len(v_act) == len(val)):
            print("WARNING: Mismatched sequence lengths.")

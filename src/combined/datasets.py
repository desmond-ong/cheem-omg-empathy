from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def len_to_mask(lengths):
    """Converts list of sequence lengths to a mask tensor."""
    mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths))
    mask = mask < torch.tensor(lengths).unsqueeze(1)
    return mask.unsqueeze(-1)

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
    mask = len_to_mask(lengths)
    return tuple(padded + [mask, lengths])

class OMGcombined(Dataset):
    """Dataset that loads features for each modality and valence ratings.
    
    audio_path -- Folder of OpenSMILE features in CSV format
    text_path -- Folder of text features in NPY format
    v_sub_path -- Folder of subject visual features in NPY format
    v_act_path -- Folder of actor visual features in NPY format
    val_path -- Folder of valence annotations in CSV format
    """
    
    def __init__(self, audio_path=None, text_path=None,
                 v_sub_path=None, v_act_path=None, val_path=None,
                 pattern="Subject_(\d+)_Story_(\d+)(?:_\w*)?",
                 fps=25.0, chunk_dur=1.0, split_ratio=1, truncate=False,
                 normalize=False, dataset=None):
        # Copy construct if dataset is provided
        if dataset is not None:
            self.copy(dataset)
            return
        
        self.time_ratio = fps * chunk_dur

        # Load files into list
        audio_files = [os.path.join(audio_path, fn) for fn
                       in sorted(os.listdir(audio_path))
                       if re.match(pattern, fn) is not None]
        text_files = [os.path.join(text_path, fn) for fn
                      in sorted(os.listdir(text_path))
                      if re.match(pattern, fn) is not None]
        v_sub_files = [os.path.join(v_sub_path, fn) for fn
                       in sorted(os.listdir(v_sub_path))
                       if re.match(pattern, fn) is not None]
        v_act_files = [os.path.join(v_act_path, fn) for fn
                       in sorted(os.listdir(v_act_path))
                       if re.match(pattern, fn) is not None]
        val_files = [os.path.join(val_path, fn) for fn
                     in sorted(os.listdir(val_path))
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
        for fn in sorted(os.listdir(val_path)):
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
            # Append data
            self.audio_data.append(audio)
            self.text_data.append(text)
            self.v_sub_data.append(v_sub)
            self.v_act_data.append(v_act)
            self.val_data.append(val)

        # Normalize inputs
        if normalize:
            self.normalize()
            
        # Split data to create more examples
        self.split(split_ratio)
            
    def __len__(self):
        return len(self.val_split)

    def __getitem__(self, i):
        return (self.audio_split[i], self.text_split[i],
                self.v_sub_split[i], self.v_act_split[i], self.val_split[i])

    def normalize(self):
        """Rescale all inputs to [-1, 1] range."""
        audio_max = np.stack([a.max(0) for a in self.audio_data]).max(0)
        text_max = np.stack([a.max(0) for a in self.text_data]).max(0)
        v_sub_max = np.stack([a.max(0) for a in self.v_sub_data]).max(0)
        v_act_max = np.stack([a.max(0) for a in self.v_act_data]).max(0)

        audio_min = np.stack([a.min(0) for a in self.audio_data]).min(0)
        text_min = np.stack([a.min(0) for a in self.text_data]).min(0)
        v_sub_min = np.stack([a.min(0) for a in self.v_sub_data]).min(0)
        v_act_min = np.stack([a.min(0) for a in self.v_act_data]).min(0)

        audio_rng = audio_max - audio_min
        audio_rng = audio_rng * (audio_rng > 0) + 1e-10 * (audio_rng <= 0)
        text_rng = text_max - text_min
        text_rng = text_rng * (text_rng > 0) + 1e-10 * (text_rng <= 0)
        v_sub_rng = v_sub_max - v_sub_min
        v_sub_rng = v_sub_rng * (v_sub_rng > 0) + 1e-10 * (v_sub_rng <= 0)
        v_act_rng = v_act_max - v_act_min
        v_act_rng = v_act_rng * (v_act_rng > 0) + 1e-10 * (v_act_rng <= 0)
        
        self.audio_data = [(a-audio_min) / audio_rng * 2 - 1 for
                           a in self.audio_data]
        self.text_data = [(a-text_min) / text_rng * 2 - 1 for
                           a in self.text_data]
        self.v_sub_data = [(a-v_sub_min) / v_sub_rng * 2 - 1 for
                           a in self.v_sub_data]
        self.v_act_data = [(a-v_act_min) / v_act_rng * 2 - 1 for
                           a in self.v_act_data]
        
    def split(self, n):
        """Splits each sequence into n chunks."""
        self.split_ratio = n
        self.audio_split = list(itertools.chain.from_iterable(
            [np.array_split(a, n, 0) for a in self.audio_data]))
        self.text_split = list(itertools.chain.from_iterable(
            [np.array_split(a, n, 0) for a in self.text_data]))
        self.v_sub_split = list(itertools.chain.from_iterable(
            [np.array_split(a, n, 0) for a in self.v_sub_data]))
        self.v_act_split = list(itertools.chain.from_iterable(
            [np.array_split(a, n, 0) for a in self.v_act_data]))
        self.val_split = list(itertools.chain.from_iterable(
            [np.array_split(a, n, 0) for a in self.val_data]))
    
    def copy(self, other):
        """Copy constructor"""
        self.stories = list(other.stories)
        self.subjects = list(other.subjects)

        self.audio_data = list(other.audio_data)
        self.text_data = list(other.text_data)
        self.v_sub_data = list(other.v_sub_data)
        self.v_act_data = list(other.v_act_data)

        self.val_data = list(other.val_data)
        self.val_orig = list(other.val_orig)

        self.time_ratio = other.time_ratio
        self.split(other.split_ratio)
    
    def join(self, other):
        """Join with another dataset."""
        if (self.time_ratio != other.time_ratio or
            self.split_ratio != other.split_ratio):
            raise Exception("Time and split ratios need to match.")
        joined = self.__class__(dataset=self)
        joined.stories += other.stories
        joined.subjects += other.subjects
        joined.audio_data += other.audio_data
        joined.text_data += other.text_data
        joined.v_sub_data += other.v_sub_data
        joined.v_act_data += other.v_act_data
        joined.val_data += other.val_data
        joined.val_orig += other.val_orig
        joined.split(self.split_ratio)
        return joined

    def extract_story(self, s):
        """Extract specified story ids to form new train-test split."""
        extract = self.__class__(dataset=self)
        remain = self.__class__(dataset=self)
        # Find indices of specified story
        idx = [i for i, story in enumerate(self.stories) if story in s]
        # Extract data for specified story
        extract.stories = [self.stories[i] for i in idx]
        extract.subjects = [self.subjects[i] for i in idx]
        extract.audio_data = [self.audio_data[i] for i in idx]
        extract.text_data = [self.text_data[i] for i in idx]
        extract.v_sub_data = [self.v_sub_data[i] for i in idx]
        extract.v_act_data = [self.v_act_data[i] for i in idx]
        extract.val_data = [self.val_data[i] for i in idx]
        extract.val_orig = [self.val_orig[i] for i in idx]
        # Delete extracted data from remainder
        for i in sorted(idx, reverse=True):
            del remain.stories[i]
            del remain.subjects[i]
            del remain.audio_data[i]
            del remain.text_data[i]
            del remain.v_sub_data[i]
            del remain.v_act_data[i]
            del remain.val_data[i]
            del remain.val_orig[i]
        extract.split(self.split_ratio)
        remain.split(self.split_ratio)
        return extract, remain
    
if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="./data/Training",
                        help='dataset base folder')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs')
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
    if args.normalize:
        print("Normalizing data.")
        dataset.normalize()

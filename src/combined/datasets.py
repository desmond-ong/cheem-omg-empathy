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

class OMGMulti(Dataset):
    """Multimodal dataset for OMG empathy challenge."""
    
    def __init__(self, in_names=None, in_paths=None, val_path=None,
                 pattern="Subject_(\d+)_Story_(\d+)(?:_\w*)?",
                 fps=25.0, chunk_dur=1.0, split_ratio=1, truncate=False,
                 normalize=False, dataset=None):
        """Loads valence ratings and features for each modality.

        in_names -- names of each input modality
        in_paths -- list of folders containing input features
        val_path -- folder containing valence ratings
        pattern -- pattern file name
        fps -- sampling rate of valence annotations
        chunk_dur -- duration of time chucks for input features
        truncate -- if true, truncate to modality with minimum length
        normalize -- if true, normalize features to [-1,1]
        dataset -- if provided, copy construct
        """
        # Copy construct if dataset is provided
        if dataset is not None:
            self.copy(dataset)
            return

        # Store modality names, ratio of sampling rates
        self.in_names = in_names
        self.time_ratio = fps * chunk_dur

        # Load filenames into lists
        in_files = dict()
        in_paths = dict(zip(in_names, in_paths))
        for n in self.in_names:
            in_files[n] = [os.path.join(in_paths[n], fn) for fn
                           in sorted(os.listdir(in_paths[n]))
                           if re.match(pattern, fn) is not None]
        val_files = [os.path.join(val_path, fn) for fn
                     in sorted(os.listdir(val_path))
                     if re.match(pattern, fn) is not None]

        # Check that number of files are equal
        for n in self.in_names:
            if len(in_files[n]) != len(val_files):
                raise Exception("Number of files do not match.")

        # Store subject and story IDs
        self.subjects = []
        self.stories = []
        for fn in sorted(os.listdir(val_path)):
            match = re.match(pattern, fn)
            if match:
                self.subjects.append(match.group(1))
                self.stories.append(match.group(2))
        
        # Load data from files
        self.val_data = []
        self.val_orig = []
        self.in_data = {n: [] for n in self.in_names}
        for i in range(len(val_files)):
            # Load and store original valence ratings
            val = pd.read_csv(val_files[i])
            self.val_orig.append(np.array(val).flatten())
            # Average valence across time chunks
            group_idx = np.arange(len(val)) // self.time_ratio
            val = np.array(val.groupby(group_idx).mean())
            seq_len = len(val)
            self.val_data.append(val)
            # Load each input modality
            for n, in_data in self.in_data.iteritems():
                fp = in_files[n][i]
                if re.match("^.*\.csv", fp):
                    d = np.array(pd.read_csv(fp))
                else: 
                    d = np.load(fp)
                if len(d.shape) > 2:
                    d = d.reshape(d.shape[0], -1)
                in_data.append(d)
                if len(d) < seq_len:
                    seq_len = len(d)
            # Truncate to minimum sequence length
            if truncate:
                self.val_data[-1] = self.val_data[-1][:seq_len]
                for n in self.in_names:
                    self.in_data[n][-1] = self.in_data[n][-1][:seq_len]

        # Normalize inputs
        if normalize:
            self.normalize()
            
        # Split data to create more examples
        self.split(split_ratio)
            
    def __len__(self):
        return len(self.val_split)

    def __getitem__(self, i):
        in_data = [self.in_split[n][i] for n in self.in_names]
        return tuple(in_data + [self.val_split[i]])

    def normalize(self):
        """Rescale all inputs to [-1, 1] range."""
        # Find max and min for each dimension of each modality
        in_max = {n: np.stack([a.max(0) for a in self.in_data[n]]).max(0)
                  for n in self.in_names}
        in_min = {n: np.stack([a.min(0) for a in self.in_data[n]]).min(0)
                  for n in self.in_names}
        # Compute range per dim and add constant to ensure it is non-zero
        in_rng = {n: (in_max[n]-in_min[n]) for n in self.in_names}
        in_rng = {n: in_rng[n] * (in_rng[n] > 0) + 1e-10 * (in_rng[n] <= 0)
                  for n in self.in_names}

        # Actually rescale the data
        for n in self.in_names:
            self.in_data[n] = [(a-in_min[n]) / in_rng[n] * 2 - 1 for
                               a in self.in_data[n]]
        
    def split(self, r):
        """Splits each sequence into n chunks."""
        self.split_ratio = r
        self.in_split = dict()
        for n in self.in_names:
            self.in_split[n] = list(itertools.chain.from_iterable(
                [np.array_split(a, r, 0) for a in self.in_data[n]]))
        self.val_split = list(itertools.chain.from_iterable(
            [np.array_split(a, r, 0) for a in self.val_data]))
    
    def copy(self, other):
        """Copy constructor."""
        self.stories = list(other.stories)
        self.subjects = list(other.subjects)

        self.in_names = other.in_names
        self.in_data = {k: list(v) for k, v in other.in_data.iteritems()}

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
        joined.val_data += other.val_data
        joined.val_orig += other.val_orig
        for n in joined.in_names:
            joined.in_data[n] += other.in_data[n]
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
        extract.val_data = [self.val_data[i] for i in idx]
        extract.val_orig = [self.val_orig[i] for i in idx]
        for n in extract.in_names:
            extract.in_data[n] = [self.in_data[n][i] for i in idx]
        # Delete extracted data from remainder
        for i in sorted(idx, reverse=True):
            del remain.stories[i]
            del remain.subjects[i]
            del remain.val_data[i]
            del remain.val_orig[i]
            for n in remain.in_names:
                del remain.in_data[n][i]
        extract.split(self.split_ratio)
        remain.split(self.split_ratio)
        return extract, remain

class OMGFusion(OMGMulti):
    """Variant of OMGMulti which returns concatenated input features."""
    def __getitem__(self, i):
        in_data = [self.in_split[n][i] for n in self.in_names]
        return np.concatenate(in_data, axis=1), self.val_split[i]
    
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
    
if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="./data/Training",
                        help='dataset base folder')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs')
    args = parser.parse_args()

    in_names = ['audio', 'text', 'v_sub', 'v_act']
    in_paths = dict()
    in_paths['audio'] = os.path.join(args.folder, "CombinedAudio")
    in_paths['text'] = os.path.join(args.folder, "CombinedText")
    in_paths['v_sub'] = os.path.join(args.folder, "CombinedVSub")
    in_paths['v_act'] = os.path.join(args.folder, "CombinedVAct")
    val_path = os.path.join(args.folder, "Annotations")

    print("Loading data...")
    dataset = OMGMulti(in_names, [in_paths[n] for n in in_names], val_path)
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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CombinedLSTM(nn.Module):
    """Basic audio-text-visual LSTM model with feature level fusion."""
    
    def __init__(self, mods=('audio', 'text', 'v_sub', 'v_act'),
                 dims=(990, 300, 4096, 4096), embed_dim=128,
                 hidden_dim=512, n_layers=1, attn_len=3, use_cuda=False):
        super(CombinedLSTM, self).__init__()
        self.mods = mods
        self.n_mods = len(mods)
        self.dims = dict(zip(mods, dims))
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attn_len = attn_len
        
        # Create raw-to-embed FC+Dropout layer for each modality
        self.embed = dict()
        dropouts = {'audio': 0.1, 'text': 0.1, 'v_sub': 0.5, 'v_act': 0.0}
        for m in self.mods:
            self.embed[m] = nn.Sequential(nn.Dropout(p=dropouts[m]),
                                          nn.Linear(self.dims[m], embed_dim),
                                          nn.ReLU())
            self.add_module('embed_{}'.format(m), self.embed[m])
        # Layer that computes attention from embeddings
        self.attn = nn.Sequential(nn.Linear(len(mods)*embed_dim, embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim, attn_len),
                                  nn.Softmax(dim=1))
        # LSTM computes hidden states from embeddings for each modality
        self.lstm = nn.LSTM(len(mods) * embed_dim, hidden_dim,
                            n_layers, batch_first=True)
        # Regression network from LSTM hidden states to predicted valence
        self.h_to_out = nn.Sequential(nn.Linear(hidden_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, 1))
        # Enable CUDA if flag is set
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, inputs, lengths):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Flatten temporal dimension
        for m in self.mods:
            inputs[m] = inputs[m].view(-1, self.dims[m])
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.embed[m](inputs[m]) for m in self.mods], 1)
        # Compute attention weights
        attn = self.attn(embed)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, self.n_mods*self.embed_dim)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        # Forward propagate LSTM
        h, _ = self.lstm(embed, (h0, c0))
        # Undo the packing
        h, _ = pad_packed_sequence(h, batch_first=True)
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        stacked = torch.stack([self.pad_shift(h, i) for
                               i in range(self.attn_len)], dim=-1)
        context = torch.sum(attn.unsqueeze(2) * stacked, dim=-1)
        # Flatten temporal dimension
        out = context.reshape(-1, self.hidden_dim)
        # Decode the context for each time step
        out = self.h_to_out(out)
        # Unflatten temporal dimension
        out = out.view(batch_size, seq_len, 1)
        return out

    def pad_shift(self, x, shift):
        """Shift 3D tensor forwards in time with zero padding."""
        if shift > 0:
            padding = torch.zeros(x.size(0), shift, x.size(2))
            if self.use_cuda:
                padding = padding.cuda()
            return torch.cat((padding, x[:, :-shift, :]), dim=1)
        else:
            return x
    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os
    from datasets import OMGcombined
    
    base_folder = "./data/Training"
    audio_path = os.path.join(base_folder, "CombinedAudio")
    text_path = os.path.join(base_folder, "CombinedText")
    v_sub_path = os.path.join(base_folder, "CombinedVSub")
    v_act_path = os.path.join(base_folder, "CombinedVAct")
    valence_path = os.path.join(base_folder, "Annotations")

    print("Loading data...")
    dataset = OMGcombined(audio_path, text_path,
                          v_sub_path, v_act_path, valence_path,
                          truncate=True)
    print("Building model...")
    model = CombinedLSTM()
    model.eval()
    print("Passing a sample through the model...")
    audio, text, v_sub, v_act, valence = dataset[0]
    lengths = [audio.shape[0]]
    inputs = {'audio': audio, 'text': text, 'v_sub': v_sub, 'v_act': v_act}
    for m in inputs.keys():
        inputs[m] = torch.tensor(inputs[m]).unsqueeze(0).float()
    out = model(inputs, lengths).view(-1)
    print("Predicted valences:")
    for o in out:
        print("{:+0.3f}".format(o.item()))

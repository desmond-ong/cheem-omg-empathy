from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CombinedLSTM(nn.Module):
    """Basic audio-text-visual LSTM model with feature level fusion."""
    
    def __init__(self, mods=('audio', 'text', 'v_sub', 'v_act'),
                 dims=(990, 300, 4096, 4096), embed_dim=128, hidden_dim=512,
                 n_layers=1, attn_len=3, reconstruct=False, use_cuda=False):
        super(CombinedLSTM, self).__init__()
        self.mods = mods
        self.n_mods = len(mods)
        self.dims = dict(zip(mods, dims))
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attn_len = attn_len
        self.reconstruct = reconstruct
        
        # Create raw-to-embed FC+Dropout layer for each modality
        self.embed = dict()
        dropouts = {'audio': 0.1, 'text': 0.1, 'v_sub': 0.2, 'v_act': 0.2}
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
        self.dec_target = nn.Sequential(nn.Linear(hidden_dim, embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(embed_dim, 1))
        # Decoder networks to reconstruct each modality
        if self.reconstruct:
            self.dec = dict()
            for m in self.mods:
                self.dec[m] = nn.Sequential(nn.Linear(hidden_dim, embed_dim),
                                            nn.ReLU(),
                                            nn.Linear(embed_dim, self.dims[m]))
                self.add_module('dec_{}'.format(m), self.dec[m])
        # Enable CUDA if flag is set
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, inputs, mask, lengths, output_features=False):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.embed[m](inputs[m].view(-1, self.dims[m]))
                           for m in self.mods], 1)
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
        context = context.reshape(-1, self.hidden_dim)
        # Return features before final FC layer if flag is set
        if output_features:
            features = self.dec_target[0](context)
            return features.view(batch_size, seq_len, -1)
        # Decode the context for each time step
        target = self.dec_target(context).view(batch_size, seq_len, 1)
        # Mask target entries that exceed sequence lengths
        target = target * mask.float()
        # Reconstruct each modality if flag is set
        if self.reconstruct:
            recon = dict()
            for m in self.mods:
                recon[m] = self.dec[m](context).view(batch_size, seq_len, -1)
                recon[m] = recon[m] * mask.float()
            return target, recon
        else:
            return target, None

    def pad_shift(self, x, shift):
        """Shift 3D tensor forwards in time with zero padding."""
        if shift > 0:
            padding = torch.zeros(x.size(0), shift, x.size(2))
            if self.use_cuda:
                padding = padding.cuda()
            return torch.cat((padding, x[:, :-shift, :]), dim=1)
        elif shift < 0:
            padding = torch.zeros(x.size(0), -shift, x.size(2))
            if self.use_cuda:
                padding = padding.cuda()
            pad = torch.cat((x[:, -shift:, :], padding), dim=1)
            return pad
        else:
            return x
    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os
    from datasets import OMGMulti, collate_fn

    base_folder = "./data/Training"
    in_names = ['audio', 'text', 'v_sub', 'v_act']
    in_paths = dict()
    in_paths['audio'] = os.path.join(base_folder, "CombinedAudio")
    in_paths['text'] = os.path.join(base_folder, "CombinedText")
    in_paths['v_sub'] = os.path.join(base_folder, "CombinedVSub")
    in_paths['v_act'] = os.path.join(base_folder, "CombinedVAct")
    val_path = os.path.join(base_folder, "Annotations")
    
    print("Loading data...")
    dataset = OMGMulti(in_names, [in_paths[n] for n in in_names], val_path)
    print("Building model...")
    model = CombinedLSTM()
    model.eval()
    print("Passing a sample through the model...")
    audio, text, v_sub, v_act, val, mask, lengths = collate_fn([dataset[0]])
    inputs = {'audio': audio, 'text': text, 'v_sub': v_sub, 'v_act': v_act}
    for m in inputs.keys():
        inputs[m] = torch.tensor(inputs[m]).float()
    out, _ = model(inputs, mask, lengths)
    out = out.view(-1)
    print("Predicted valences:")
    for o in out:
        print("{:+0.3f}".format(o.item()))

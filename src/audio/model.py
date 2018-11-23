"""LSTM model that predicts valence from OpenSMILE features."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, use_cuda=False):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Dropout layer so we don't use all the raw features
        self.dropout = nn.Dropout(p=0.1)
        # LSTM computes hidden states
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        # Fully-connected layer to 128 dim embeddings
        self.fc1 = nn.Linear(hidden_size, 128)
        # Fully-connected layer computes valence
        self.fc2 = nn.Linear(128, 1)
        # Enable CUDA if flag is set
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, x, lengths):
        # Get batch size
        batch_size, max_seq_length, _ = x.size()
        # Dropout the raw features
        x = self.dropout(x)
        # Pack the input to mask padded entries
        x = pack_padded_sequence(x, lengths, batch_first=True)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, (h0, c0))
        # Undo the packing
        out, _ = pad_packed_sequence(out, batch_first=True)
        # Reshape output to (batch_size * max_seq_length, hidden_size)
        out = out.reshape(-1, self.hidden_size)
        # Decode the hidden state of each time step
        out = self.fc2(self.fc1(out))
        # Reshape back to (batch_size, max_seq_length, hidden_size)
        out = out.view(batch_size, max_seq_length, 1)
        return out


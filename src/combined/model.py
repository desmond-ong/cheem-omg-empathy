from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CombinedLSTM(nn.Module):
    """Basic audio-text-visual LSTM model with feature level fusion."""
    
    def __init__(self, audio_size=990, text_size=300, visual_size=4096,
                 embed_size=128, hidden_size=128, n_layers=1, use_cuda=False):
        super(CombinedLSTM, self).__init__()
        self.audio_size = audio_size
        self.text_size = text_size
        self.visual_size = visual_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # Create raw-to-embed FC+Dropout layer for each modality
        self.a_to_embed = nn.Sequential(nn.Dropout(p=0.1),
                                        nn.Linear(audio_size, embed_size))
        self.t_to_embed = nn.Sequential(nn.Dropout(p=0.1),
                                        nn.Linear(text_size, embed_size))
        self.v_to_embed = nn.Sequential(nn.Dropout(p=0.0),
                                        nn.Linear(visual_size, embed_size))
        # LSTM computes hidden states from embeddings for each modality
        self.lstm = nn.LSTM(3 * embed_size, hidden_size,
                            n_layers, batch_first=True)
        # Regression network from LSTM hidden states to predicted valence
        self.h_to_out = nn.Sequential(nn.Linear(hidden_size, embed_size),
                                      nn.Linear(embed_size, 1))
        # Enable CUDA if flag is set
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, audio, text, visual, lengths):
        # Get batch size
        batch_size, max_seq_length, _ = audio.size()
        # Flatten temporal dimension
        audio = audio.view(-1, self.audio_size)
        text = text.view(-1, self.text_size)
        visual = visual.view(-1, self.visual_size)
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.a_to_embed(audio),
                           self.t_to_embed(text),
                           self.v_to_embed(visual)], dim=1)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, max_seq_length, 3*self.embed_size)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        # Forward propagate LSTM
        out, (h, c) = self.lstm(embed, (h0, c0))
        # Undo the packing
        out, _ = pad_packed_sequence(out, batch_first=True)
        # Flatten temporal dimension
        out = out.reshape(-1, self.hidden_size)
        # Decode the hidden state of each time step
        out = self.h_to_out(out)
        # Unflatten temporal dimension
        out = out.view(batch_size, max_seq_length, 1)
        return out

if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os
    from datasets import OMGcombined
    
    base_folder = "./data/Training"
    audio_path = os.path.join(base_folder, "CombinedAudio")
    text_path = os.path.join(base_folder, "CombinedText")
    visual_path = os.path.join(base_folder, "CombinedVisual")
    valence_path = os.path.join(base_folder, "Annotations")

    print("Loading data...")
    dataset = OMGcombined(audio_path, text_path, visual_path, valence_path)
    print("Building model...")
    model = CombinedLSTM()
    model.eval()
    print("Passing a sample through the model...")
    audio, text, visual, valence = dataset[0]
    lengths = [audio.shape[0]]
    audio = torch.tensor(audio).unsqueeze(0).float()
    text = torch.tensor(text).unsqueeze(0).float()
    visual = torch.tensor(visual).unsqueeze(0).float()
    out = model(audio, text, visual, lengths).view(-1)
    print("Predicted valences:")
    for o in out:
        print("{:+0.3f}".format(o.item()))

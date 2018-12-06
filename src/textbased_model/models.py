import torch
from torch import nn
import sys
sys.path.append("/raid/omgempathy/")  # path to the main dir
from src.textbased_model import config
import torch.nn.functional as F

device = config.device


class LSTMEmo_noSequence(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, ln1_out_dim, batch_size=1):
        super(LSTMEmo_noSequence, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim)
        self.dropout = nn.Dropout2d(config.dropout)
        self.linear1 = nn.Linear(lstm_hidden_dim, ln1_out_dim)
        self.linear2 = nn.Linear(ln1_out_dim, 1)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(1, self.batch_size, self.hidden_dim).to(device))

    def forward(self, words):
        # input is one utterance
        hidden = self.init_hidden()
        lstm_output, hidden = self.lstm(words, hidden)
        dropout_output = self.dropout(lstm_output)
        ln1_output = self.linear1(dropout_output)
        ln1_output = F.relu(ln1_output)
        ln2_output = torch.tanh(self.linear2(ln1_output))
        return ln2_output


# model for utterances chunking (tested with cross validation, 30 Nov 2018)

class LSTMEmo_utter_sequence(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, ln1_out_dim, batch_size=1):
        super(LSTMEmo_utter_sequence, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size
        print("self batch size: ", self.batch_size)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim)
        self.dropout = nn.Dropout2d(config.dropout)
        self.linear1 = nn.Linear(lstm_hidden_dim, ln1_out_dim)
        self.linear2 = nn.Linear(ln1_out_dim, 1)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(1, self.batch_size, self.hidden_dim).to(device))

    def forward(self, utterances, use_dropout=True):
        # input is one utterance
        hidden = self.init_hidden()
        lstm_output, hidden = self.lstm(utterances, hidden)
        dropout_output = lstm_output
        if use_dropout:
            dropout_output = self.dropout(lstm_output)
        ln1_output = self.linear1(dropout_output)
        ln1_output_relu = F.relu(ln1_output)
        ln2_output = self.linear2(ln1_output_relu)
        final_output = torch.tanh(ln2_output)
        # return final_output, ln1_output
        return final_output

# extended from LSTMEmo_utter_sequence
# global attention, attention length = sequence length

class LSTM_with_Attention(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, ln1_out_dim, attn_len, batch_size=1, n_layers=1):
        super(LSTM_with_Attention, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        # Layer that computes attention from input
        self.attn = nn.Sequential(nn.Linear(input_dim, input_dim),
                                  nn.ReLU(),
                                  nn.Linear(input_dim, attn_len),
                                  nn.Softmax(dim=1))
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim)
        self.dropout = nn.Dropout2d(config.dropout)

        self.h_to_out = nn.Sequential(nn.Linear(lstm_hidden_dim, ln1_out_dim),
                                      nn.ReLU(),
                                      nn.Linear(ln1_out_dim, 1),
                                      nn.Tanh())

    def init_hidden(self):
        return (torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(device))

    def forward(self, utterances):
        '''
        :param utterances:  (seq_len, batch_size, input_dim)
        :return:
        '''
        # input is one utterance
        hidden = self.init_hidden()

        lstm_output, hidden = self.lstm(utterances, hidden)
        attn_weights = self.attn(utterances)  # [input_seq_len, batch, output_dim]
        attn_weights_batch_first = attn_weights.view(self.batch_size, attn_weights.shape[0], -1)
        lstm_batch_first = lstm_output.view(self.batch_size, lstm_output.shape[0], -1)
        # print("LSTM:", lstm_output.shape)
        # print("Attention:", attn_weights.shape)
        # print("LSTM batch first:", lstm_batch_first.shape)
        # print("Attention batch first:", attn_weights_batch_first.shape)

        output_attn_weighted = torch.bmm(attn_weights_batch_first, lstm_batch_first).view(attn_weights.shape[0],
                                                                                          self.batch_size, -1)
        # print("Output attn weighted: ", output_attn_weighted.shape)

        dropout_output = self.dropout(output_attn_weighted)
        # print(dropout_output)
        ln2_output = self.h_to_out(dropout_output)
        # print(ln2_output)
        return ln2_output



class LSTMEmo_duo_sequence(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, ln1_out_dim, dropout=config.dropout, batch_size=1):
        super(LSTMEmo_duo_sequence, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.linear1 = nn.Linear(lstm_hidden_dim, ln1_out_dim)
        self.linear2 = nn.Linear(ln1_out_dim, 1)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(1, self.batch_size, self.hidden_dim).to(device))

    def forward(self, words, hidden):
        # input is one utterance
        lstm_output, hidden = self.lstm(words, hidden)
        dropout_output = self.dropout(lstm_output)
        ln1_output = self.linear1(dropout_output)
        ln1_output = F.relu(ln1_output)
        ln2_output = torch.tanh(self.linear2(ln1_output))
        return ln2_output, hidden





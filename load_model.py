import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from copy import deepcopy


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, bsz):
        super(RNNModel, self).__init__()
        self.nhid = 256
        self.bsz = bsz
        self.drop = nn.Dropout(.05)
        self.encoder = nn.Embedding(ntoken, 128)
        self.rnn = nn.LSTM(128, 256, dropout=.05)
        self.fc_dr = nn.Linear(256, 128)
        self.decoder = nn.Linear(128, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc_dr.bias.data.fill_(0)
        self.fc_dr.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        emb = self.drop(self.encoder(x))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        dr = F.tanh(self.fc_dr(output))
        decoded = self.decoder(dr.view(dr.size(0) * dr.size(1), dr.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self):
        weight = next(self.parameters()).data
        self.hidden = (Variable(weight.new(1, self.bsz, self.nhid).zero_()), Variable(weight.new(1, self.bsz, self.nhid).zero_()))


if __name__ == '__main__':
    model = torch.load('gold_model.mdl', map_location=lambda storage, loc: storage)

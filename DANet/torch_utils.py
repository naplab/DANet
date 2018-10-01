import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should 
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, 
                                         batch_first=True, bidirectional=bidirectional)
        
        

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input, hidden):
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_())
        
        
class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.
    
    args:
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should 
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """
    
    def __init__(self, input_size, hidden_size, nonlinearity=None):
        super(FCLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.FC = nn.Linear(self.input_size, self.hidden_size)
        if nonlinearity:
            self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = None
            
        self.init_hidden()
    
    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)
        
    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size)
        self.FC.bias.data.fill_(0)
        self.FC.weight.data.uniform_(-initrange, initrange)

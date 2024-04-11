import numpy as np
import matplotlib.pyplot as plt
import torch  as th
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import pandas as pd
import seaborn as sns
import random
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, random_split
from BESTRq_classes.BESTRq import BestRqFramework, RandomProjectionQuantizer
from compute_fft import compute_spectrogram, plot_spectrogram, mask


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim=600, hidden_dim=100, nstack=2, dropout=0, codebook_size = 50, embedding_dim = 50):
        super(AttentionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nstack = nstack
        self.dropout = dropout
        # LSTM layer
        self.lstm_stack = nn.ModuleList([
            nn.LSTM(input_size=input_dim if i == 0 else 2*hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional= True,
            dropout=dropout if i < nstack - 1 else 0)  # Apply dropout only between layers
        for i in range(nstack)])


        # Linear layers for attention
        self.attention_linear = nn.Linear(2*hidden_dim, 1)
        self.context_linear = nn.Linear(2*hidden_dim, hidden_dim)

        # Final linear layer
        self.linear = nn.Linear(hidden_dim, codebook_size)

        # Dropout layer
        self.drop = nn.Dropout(p=dropout)


    def init_hidden(self, bsz):
        # Initialize hidden state for LSTM
        return (th.zeros(2, bsz, self.hidden_dim),
            th.zeros(2, bsz, self.hidden_dim))


    def forward(self, inputs, h0=None):
        if h0 is None:
            (h0, c0) = self.init_hidden(inputs.size(3))  # Use size(1) to get the batch size

        h, c  = h0[0:2, :, :].to(device), c0[0:2, :, :].to(device)


        inputs = inputs.squeeze(2).permute(0,2,1)
        # LSTM forward pass
        for i in range(self.nstack):


            inputs, (h,c) = self.lstm_stack[i](inputs, (h, c))
            inputs = self.drop(inputs)


        # Compute attention weights
        attention_weights = F.softmax(self.attention_linear(inputs), dim=0)

        # Apply attention to LSTM output
        attention_applied = th.sum(attention_weights * inputs, dim=1)

        # Compute context vector
        context = self.context_linear(attention_applied)

        context = F.relu(context)

        # Apply dropout
        out = self.drop(context)

        #Linear layer
        out = self.linear(out)


        #Softmax layer
        out = F.softmax(out, dim= 1)

        return out

class GRUPredictor(nn.Module):
    """GRUPredictor is a recurrent model. It takes as input a vector and predict
    the next one given past observations."""
    def __init__(self, input_dim=3, hidden_dim=50, nstack = 3, dropout=0):
        super(GRUPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nstack = nstack
        self.dropout = dropout
        self.gru = nn.GRU(input_size=input_dim, hidden_size = hidden_dim,  num_layers = nstack, batch_first=  True)
        self.linear = nn.Linear(in_features = 2*600*50, out_features = 6)
        self.drop = nn.Dropout(p = dropout)
    def init_hidden(self, bsz):
        # This function is given: understand it.
        self.h = th.zeros(self.nstack, bsz, self.hidden_dim)
        return self.h

    def forward(self, inputs, h0=None):
        shape = inputs.shape
        inputs = inputs.permute(0, 2, 1)
        if h0 == None:
          h0 = self.init_hidden(inputs.shape[0])
          h0 = h0.to(inputs.device)
        hidden, h0 = self.gru(inputs, h0)
        out = self.drop(hidden)
        out = out.flatten().view(shape[0], -1)
        out = self.linear(out)
        return out


class AttentionLSTM_spec(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=100, nstack=2, dropout=0, codebook_size=50,  conv_channels=128, conv_kernel_size=3):
        super(AttentionLSTM_spec, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nstack = nstack
        self.dropout = dropout

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=conv_kernel_size, padding=1)

        # Batch normalization layer
        self.batchnorm = nn.BatchNorm1d(conv_channels)

        # Max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_dim, num_layers=nstack, bidirectional=True)

        # Linear layers for attention
        self.attention_linear = nn.Linear(2 * hidden_dim, 1)
        self.context_linear = nn.Linear(2 * hidden_dim, hidden_dim)

        # Final linear layer
        self.linear = nn.Linear(2*hidden_dim, codebook_size)

        # Dropout layer
        self.drop = nn.Dropout(p=dropout)

    def init_hidden(self, bsz):
        # Initialize hidden state for LSTM
        return (th.zeros(2 * self.nstack, bsz, self.hidden_dim),
                th.zeros(2 * self.nstack, bsz, self.hidden_dim))

    def forward(self, inputs, h0=None):
        inputs = inputs.permute(0,2,1)
        if h0 is None:
            (h0, c0) = self.init_hidden(100)  # Use size(1) to get the batch size

        h0, c0 = h0.to(inputs.device), c0.to(inputs.device)

        #Convolutional layer
        conv_output = self.conv1d(inputs)

        # Apply batch normalization
        conv_output = self.batchnorm(conv_output)

        # Apply max pooling
        conv_output = self.maxpool(conv_output)


        # LSTM forward pass
        lstm_output, _ = self.lstm(conv_output.permute(0,2,1), (h0, c0))



        context = F.relu(lstm_output)

        # Apply dropout
        out = self.drop(context)

        # Linear layer
        out = self.linear(out)

        # Softmax layer
        out = F.softmax(out, dim=1)

        return out


class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 9 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(th.relu(self.conv1(x)))
        x = self.pool(th.relu(self.conv2(x)))
        x = self.pool(th.relu(self.conv3(x)))
        x = self.pool(th.relu(self.conv4(x)))
        x = x.view(-1, 128 * 9 * 2)
        x = self.dropout(th.relu(self.fc1(x)))
        x = self.fc2(x)
        return x




class CNN_multiproj(nn.Module):
    def __init__(self, codebook_size=100, quantizer_dim =  100):
        super(CNN_multiproj, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 2, 512)
        self.fc2 = nn.Linear(512, quantizer_dim)
        self.fc3 = nn.Linear(9, 500)
        self.fc4 = nn.Linear(500, codebook_size)
        self.quantizer_dim = quantizer_dim
        self.codebook_size = codebook_size
    def forward(self, x):
        x = self.pool(th.relu(self.conv1(x)))
        x = self.pool(th.relu(self.conv2(x)))
        x = self.pool(th.relu(self.conv3(x)))
        x = self.pool(th.relu(self.conv4(x)))
        x = x.view(-1, 128*2)
        x = self.dropout(th.relu(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(-1, 9)
        x = self.dropout(th.relu(self.fc3(x)))
        x = self.fc4(x)
        x = x.view(-1, self.quantizer_dim, self.codebook_size)
        return x


class BiLSTM(nn.Module):
    """GRUPredictor is a recurrent model. It takes as input a vector and predict
    the next one given past observations."""
    def __init__(self, input_dim=1, hidden_dim=50, nstack = 3, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nstack = nstack
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = nstack, bias=True, batch_first=False, dropout=0.4, bidirectional=True)
        self.drop = nn.Dropout(p = dropout)
        self.linear = nn.Linear(2*hidden_dim*100, 5)

    def init_hidden(self, bsz):
        # Initialize hidden state for LSTM
        return (th.zeros(2 * self.nstack, bsz, self.hidden_dim),
                th.zeros(2 * self.nstack, bsz, self.hidden_dim))


    def forward(self, inputs, h0=None):
        if h0 is None:
            (h0, c0) = self.init_hidden(inputs.shape[1])  # Use size(1) to get the batch size

        h0, c0 = h0.to(inputs.device), c0.to(inputs.device)
        lstm_output, _ = self.lstm(inputs, (h0, c0))

        out = self.drop(lstm_output)

        out = out.flatten(start_dim = 1)

        out = self.linear(out)
        return out

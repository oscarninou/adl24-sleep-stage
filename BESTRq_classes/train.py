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
from compute_fft import compute_spectrogram, plot_spectrogram
from models.CNN_BiLSTM_Attention import ParallelModel

def pretrain(trainloader, validloader, model , BestRQ, epochs=10, lr=1e-3, device = 'cpu', raw_signal = True):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    train_losses = []  # Pour sauvegarder la loss à chaque époch
    valid_losses = []  # Pour sauvegarder la loss de validation à chaque époch
    valid_accuracies = []    # Pour sauvegarder l'accuracy de validation à chaque époch
    train_accuracies = []    # Pour sauvegarder l'accuracy de validation à chaque époch

    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        model.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            if raw_signal:
                inputs = inputs.view(1, 600, -1).to(device)
            else:
                inputs = inputs.unsqueeze(1).permute(0,3,1,2).to(device)
            inputs = BestRQ.masking(inputs)
            encoder_outs = model(inputs)
            loss = loss_function(encoder_outs, labels.view(-1))
            loss.backward()
            optimizer.step()
            # Compute accuracy
            predicted = th.argmax(encoder_outs, dim=1)
            if raw_signal:
                total += labels.size(1)
            else:
                total += labels.size(0)
            correct += (predicted == labels.view_as(predicted)).sum().item()
            epoch_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        # Calculate validation accuracy
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        with th.no_grad():
            epoch_valid_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in validloader:
                if raw_signal:
                    inputs = inputs.view(1, 600, -1).to(device)
                else:
                    inputs = inputs.unsqueeze(1).permute(0,3,1,2).to(device)
                encoder_outs = model(inputs)
                loss = loss_function(encoder_outs, labels.view(-1))
                epoch_valid_loss += loss.item()

                # Compute accuracy
                predicted = th.argmax(encoder_outs, dim=1)
                if raw_signal:
                    total += labels.size(1)
                else:
                    total += labels.size(0)
                correct += (predicted == labels.view_as(predicted)).sum().item()

            # Calculate average validation loss for the epoch
            avg_valid_loss = epoch_valid_loss / len(validloader)
            valid_losses.append(avg_valid_loss)

            # Calculate validation accuracy
            valid_accuracy = correct / total
            valid_accuracies.append(valid_accuracy)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {valid_accuracy}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies

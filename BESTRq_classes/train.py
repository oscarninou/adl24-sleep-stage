import numpy as np
import matplotlib.pyplot as plt
import torch  as th
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, random_split
from BESTRq_classes.BESTRq import BestRqFramework, RandomProjectionQuantizer
from compute_fft import compute_spectrogram, plot_spectrogram, mask


def pretrain(training_data, valid_data, model , BestRQ, ratio_dataset = 2, epochs=10, lr=1e-3, device = 'cpu', raw_signal = True, batch_size = 200, device_for_proj = True):

    xtrain, ytrain= training_data
    xvalid, yvalid = valid_data

    if device_for_proj == True:

        # Convert tensors to device
        xtrain_device = xtrain.to(device)
        xvalid_device = xvalid.to(device)

    else:
        xtrain_device = xtrain
        xvalid_device = xvalid


    # Initialize empty lists to store the results
    ytrain_batches = []
    yvalid_batches = []

    # Process training data in batches
    for i in range(0, len(xtrain_device), batch_size):
        x_batch = xtrain_device[i:min(i+batch_size, len(xtrain_device))]
        y_batch = BestRQ(x_batch,)
        ytrain_batches.append(y_batch)

    # Process validation data in batches
    for i in range(0, len(xvalid_device), batch_size):
        x_batch = xvalid_device[i:min(i+batch_size, len(xvalid_device))]
        y_batch = BestRQ(x_batch)
        yvalid_batches.append(y_batch)

    # Concatenate the batches
    ytrain = th.cat(ytrain_batches, dim=0)
    yvalid = th.cat(yvalid_batches, dim=0)
    print('Projection done')

    dataset_t = TensorDataset(xtrain, ytrain)
    dataset_size = len(dataset_t)
    half_dataset_size = dataset_size // ratio_dataset
    dataset_t1, _ = random_split(dataset_t,  [half_dataset_size, dataset_size - half_dataset_size])
    train_loader = DataLoader(dataset_t1, batch_size= batch_size, shuffle=True)
    print('Training loader ok')

    dataset_v = TensorDataset(xvalid, yvalid)
    dataset_size = len(dataset_v)
    half_dataset_size = dataset_size // ratio_dataset
    dataset_v1, dataset_v2 = random_split(dataset_v,  [half_dataset_size, dataset_size - half_dataset_size])
    valid_loader = DataLoader(dataset_v1, batch_size= batch_size, shuffle=True)
    print('Validation loader ok')




    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    train_losses = []  # Pour sauvegarder la loss à chaque époch
    valid_losses = []  # Pour sauvegarder la loss de validation à chaque époch
    valid_accuracies = []    # Pour sauvegarder l'accuracy de validation à chaque époch
    train_accuracies = []    # Pour sauvegarder l'accuracy de validation à chaque époch


    print('Training started')
    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, _ = mask(inputs, mask_prob = BestRQ.mask_prob, mask_time = BestRQ.mask_time, number_of_mask = BestRQ.num_masks_per_signal, device = inputs.device, raw_signal = raw_signal)
            encoder_outs = model(inputs)
            loss = loss_function(encoder_outs, labels.view(-1))
            loss.backward()
            optimizer.step()
            # Compute accuracy
            predicted = th.argmax(encoder_outs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.view_as(predicted)).sum().item()
            epoch_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Calculate validation accuracy
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        print('Validation')

        # Validation phase
        model.eval()
        with th.no_grad():
            epoch_valid_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                encoder_outs = model(inputs)
                loss = loss_function(encoder_outs, labels.view(-1))
                epoch_valid_loss += loss.item()

                # Compute accuracy
                predicted = th.argmax(encoder_outs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.view_as(predicted)).sum().item()

            # Calculate average validation loss for the epoch
            avg_valid_loss = epoch_valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)

            # Calculate validation accuracy
            valid_accuracy = correct / total
            valid_accuracies.append(valid_accuracy)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {valid_accuracy}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies



def train_decoder(training_data, valid_data, decoder,encoder, epochs=10, lr=1e-3, device = 'cpu', raw_signal = True, batch_size = 500, device_for_proj = True):

    xtrain, ytrain= training_data
    xvalid, yvalid = valid_data

    if device_for_proj:

        # Convert tensors to device
        xtrain, ytrain = xtrain.to(device), ytrain.to(device)
        xvalid, yvalid = xvalid.to(device), yvalid.to(device)


    dataset_t = TensorDataset(xtrain, ytrain)
    train_loader = DataLoader(dataset_t, batch_size= batch_size, shuffle=True)
    print('Training loader ok')

    dataset_v = TensorDataset(xvalid, yvalid)
    valid_loader = DataLoader(dataset_v, batch_size= batch_size, shuffle=True)
    print('Validation loader ok')




    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    train_losses = []  # Pour sauvegarder la loss à chaque époch
    valid_losses = []  # Pour sauvegarder la loss de validation à chaque époch
    valid_accuracies = []    # Pour sauvegarder l'accuracy de validation à chaque époch
    train_accuracies = []    # Pour sauvegarder l'accuracy de validation à chaque époch


    print('Training started')
    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        decoder.train()
        encoder.eval()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            if device_for_proj:
                inputs = inputs.unsqueeze(1).to(device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            encoder_outs = encoder(inputs)
            preds = decoder(encoder_outs)
            loss = loss_function(preds, labels.view(-1))
            loss.backward()
            optimizer.step()
            # Compute accuracy
            predicted = th.argmax(preds, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.view_as(predicted)).sum().item()
            epoch_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Calculate validation accuracy
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        print('Validation')

        # Validation phase
        decoder.eval()

        with th.no_grad():
            epoch_valid_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                if device_for_proj:
                    inputs = inputs.unsqueeze(1).to(device)
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                encoder_outs = encoder(inputs)
                preds = decoder(encoder_outs)
                loss = loss_function(preds, labels.view(-1))
                epoch_valid_loss += loss.item()

                # Compute accuracy
                predicted = th.argmax(preds, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.view_as(predicted)).sum().item()

            # Calculate average validation loss for the epoch
            avg_valid_loss = epoch_valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)

            # Calculate validation accuracy
            valid_accuracy = correct / total
            valid_accuracies.append(valid_accuracy)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {valid_accuracy}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies



def pretrain_multiproj(training_data, valid_data, model , BestRQ, ratio_dataset = 2, epochs=10, lr=1e-3, device = 'cpu', raw_signal = True, batch_size = 200, device_for_proj = True):

    xtrain, ytrain= training_data
    xvalid, yvalid = valid_data

    if device_for_proj == True:

        # Convert tensors to device
        xtrain_device = xtrain.to(device)
        xvalid_device = xvalid.to(device)

    else:
        xtrain_device = xtrain
        xvalid_device = xvalid


    # Initialize empty lists to store the results
    ytrain_batches = []
    yvalid_batches = []

    # Process training data in batches
    for i in range(0, len(xtrain_device), batch_size):
        x_batch = xtrain_device[i:min(i+batch_size, len(xtrain_device))]
        y_batch = BestRQ(x_batch,)
        ytrain_batches.append(y_batch)

    # Process validation data in batches
    for i in range(0, len(xvalid_device), batch_size):
        x_batch = xvalid_device[i:min(i+batch_size, len(xvalid_device))]
        y_batch = BestRQ(x_batch)
        yvalid_batches.append(y_batch)

    # Concatenate the batches
    ytrain = th.cat(ytrain_batches, dim=0)
    yvalid = th.cat(yvalid_batches, dim=0)
    print('Projection done')

    dataset_t = TensorDataset(xtrain, ytrain)
    dataset_size = len(dataset_t)
    half_dataset_size = dataset_size // ratio_dataset
    dataset_t1, _ = random_split(dataset_t,  [half_dataset_size, dataset_size - half_dataset_size])
    train_loader = DataLoader(dataset_t1, batch_size= batch_size, shuffle=True)
    print('Training loader ok')

    dataset_v = TensorDataset(xvalid, yvalid)
    dataset_size = len(dataset_v)
    half_dataset_size = dataset_size // ratio_dataset
    dataset_v1, dataset_v2 = random_split(dataset_v,  [half_dataset_size, dataset_size - half_dataset_size])
    valid_loader = DataLoader(dataset_v1, batch_size= batch_size, shuffle=True)
    print('Validation loader ok')




    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim = 2)

    train_losses = []  # Pour sauvegarder la loss à chaque époch
    valid_losses = []  # Pour sauvegarder la loss de validation à chaque époch
    valid_wers = []    # Pour sauvegarder l'accuracy de validation à chaque époch
    train_wers = []    # Pour sauvegarder l'accuracy de validation à chaque époch


    print('Training started')
    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0.0
        wrong = 0
        total = 0

        # Training phase
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, _ = mask(inputs, mask_prob = BestRQ.mask_prob, mask_time = BestRQ.mask_time, number_of_mask = BestRQ.num_masks_per_signal, device = inputs.device, raw_signal = raw_signal)
            encoder_outs = model(inputs)
            #encoder_outs = softmax(encoder_outs)
            loss = loss_function(encoder_outs, labels)
            loss.backward()
            optimizer.step()
            # Compute accuracy
            predicted = th.argmax(encoder_outs, dim=1)
            total += labels.size(0)
            wrong += (predicted != labels.view_as(predicted)).sum().item() / labels.size(1)
            epoch_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Calculate validation accuracy
        train_wer= wrong / total
        train_wers.append(train_wer)

        print('Validation')

        # Validation phase
        model.eval()
        with th.no_grad():
            epoch_valid_loss = 0.0
            wrong = 0
            total = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                encoder_outs = model(inputs)
                #encoder_outs = softmax(encoder_outs)
                loss = loss_function(encoder_outs, labels)
                epoch_valid_loss += loss.item()

                # Compute accuracy
                predicted = th.argmax(encoder_outs, dim=1)
                total += labels.size(0)
                wrong += (predicted != labels.view_as(predicted)).sum().item() / labels.size(1)

            # Calculate average validation loss for the epoch
            avg_valid_loss = epoch_valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)

            # Calculate validation accuracy
            valid_wer = wrong / total
            valid_wers.append(valid_wer)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}, Training WER: {train_wer}, Validation WER: {valid_wer}")

    return train_losses, valid_losses, train_wers, valid_wers

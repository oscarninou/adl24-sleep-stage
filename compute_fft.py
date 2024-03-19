import argparse
import glob
import math
import ntpath
import os
import shutil
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from datetime import datetime


filepath = '/Users/constouille/Documents/GitHub/DL_ESPCI/adl24-sleep-stage/cassette-th-data.pck'
xtrain, xvalid, ytrain, yvalid = np.load(filepath, allow_pickle = True)
length = len(xtrain[0])
n_dim = 4
n_subject = len(xtrain)
n_subsample = 20
size_subsample = length // n_subsample
sliding_fft_tensor = th.ones(n_dim, n_subject, length)
stop = False

for signals in [xtrain, xvalid]:
    print('Training' if (signals[0] == xtrain[0]).sum() == length else 'Validation')
    for j, signal in enumerate(signals):
        sliding_fft = []
        for balecouilles in range(n_subsample):
            signal_to_process = signal[i*size_subsample : (i+1)*size_subsample]
            fft_signal = np.fft.fft(signal_to_process)
            sliding_fft.append(abs(fft_signal))
        sliding_fft = np.array(sliding_fft)
        sliding_fft = sliding_fft.flatten()
        sliding_fft_tensor[i,j, :] = th.tensor(sliding_fft)
        if stop:
            break

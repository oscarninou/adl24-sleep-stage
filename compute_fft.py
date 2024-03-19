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
from mne.io import read_raw_edf

filepath = '/Users/constouille/Documents/GitHub/DL_ESPCI/adl24-sleep-stage/cassette-th-data.pck'
data = np.load(filepath, allow_pickle = True)
length = len(data[0][0])
n_dim = len(data)
n_subject = len(data[0])
n_subsample = 20
size_subsample = length // n_subsample
sliding_fft_tensor = th.ones(n_dim, n_subject, length)
stop = False

for i in range(n_dim//2):
    signals = data[i]
    print('yaaa')
    for j, signal in enumerate(signals):
        sliding_fft = []
        for balecouilles in range(n_subsample):
            signal_to_process = signal[i*size_subsample : (i+1)*size_subsample]
            fft_signal = np.fft.fft(signal_to_process)
            sliding_fft.append(abs(fft_signal))
        sliding_fft = np.array(sliding_fft)
        sliding_fft = sliding_fft.flatten()
        sliding_fft_tensor[i,j, :] = th.tensor(sliding_fft)
        a = 2
        if stop:
            break

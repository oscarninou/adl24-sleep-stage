import torch as th
import numpy as np
import matplotlib.pyplot as plt

def compute_spectrogram(tensor, n_fft, hop_length = None, window = None):
    return abs(th.stft(tensor, n_fft= n_fft, hop_length = hop_length, window = window, center=True, return_complex= True))


def plot_spectrogram(spectrogram, sample_rate=1, title='Spectrogram', xlabel='Time', ylabel='Frequency'):
    """
    Plot the spectrogram.

    Args:
    - spectrogram (torch.Tensor): Spectrogram to plot. It should have shape (channels, freq_bins, time_frames).
    - sample_rate (int): Sampling rate of the signal (samples per second).
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    # Convert the spectrogram tensor to numpy array
    spectrogram_np = spectrogram.numpy()

    # Get the number of channels, frequency bins, and time frames
    num_channels, freq_bins, time_frames = spectrogram_np.shape

    # Compute time axis
    time_axis = np.arange(time_frames) / sample_rate

    # Compute frequency axis
    freq_axis = np.arange(freq_bins) * (sample_rate / 2) / (freq_bins // 2)

    # Plot the spectrogram for each channel
    for i in range(num_channels):
        plt.figure(figsize=(10, 6))
        plt.imshow(np.log(spectrogram_np[i] + 1), aspect='auto', origin='lower', cmap='viridis',
                extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
        plt.title(f'{title} - Channel {i+1}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label='Log amplitude')
        plt.tight_layout()
        plt.show()
    return None

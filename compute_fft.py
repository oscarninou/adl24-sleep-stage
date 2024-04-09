import torch as th
import numpy as np
import matplotlib.pyplot as plt

def compute_spectrogram(tensor, n_fft):
    if len(tensor.shape) > 2:
        tensor_stft = []
        for i in range(tensor.shape[1]):
            feature = tensor[:, i, :]
            feature= abs(th.stft(feature, n_fft= n_fft, hop_length = n_fft//20, window = th.hann_window(n_fft).to(tensor.device), center=True, return_complex= True, normalized=True))
            tensor_stft.append(feature)
        tensor= th.stack(tensor_stft, dim=-1)
    return tensor


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

def mask(spectrogram, mask_prob, mask_time, number_of_mask, device, raw_signal):
    # Get the dimensions of the input tensor
    if raw_signal:
        batch_size, num_channel, time = spectrogram.size()
    else:
        batch_size, num_channel , freq_bins, time = spectrogram.size()


    # Calculate the start index for the mask

    noise = th.normal(mean = 0, std = 0.1, size= spectrogram.size()).to(device)


    # Generate a binary mask tensor
    mask = th.zeros(spectrogram.size())
    mask_idx = th.rand(batch_size) < mask_prob
    for _ in range(number_of_mask):
        start_index = th.randint(0, time - mask_time + 1, (batch_size,))
        for i in range(batch_size):
            if mask_idx[i]:
                if raw_signal:
                    mask[i, :, start_index[i]:start_index[i] + mask_time] = 1
                else :
                    mask[i,:, :, start_index[i]:start_index[i] + mask_time] = 1

    mask = mask > 0.5
    mask = mask.to(device)

    # Mask and replace selected elements with random noise
    masked_tensor = spectrogram.clone().to(device)
    masked_tensor[mask] = noise[mask]
    return masked_tensor, mask

# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy.signal import convolve
from torchaudio.transforms import Spectrogram

# glue first and last moment together to give a sphere?
# first moment follows last in endless loop
# %%
# arg
parser = argparse.ArgumentParser(
    description="Perform Helmholtz decomposition on the spectrogram of an audio file."
)
parser.add_argument("input_file", type=Path, help="Path to the input audio file.")
parser.add_argument("output_directory", type=Path, help="Path to the output directory.")
args = parser.parse_args()

args.output_directory.mkdir(parents=True, exist_ok=True)

waveform, SAMPLE_RATE = torchaudio.load(args.input_file)
NUM_CHANNELS, NUM_FRAMES = waveform.shape
print(f"{waveform.shape=}")
print(f"{waveform.dtype=}")

spectrogram = Spectrogram(power=True)(waveform)
print(spectrogram.shape)  # [2, 201, 62469]
single_spec = spectrogram[0]
# XXX 11:17 using only 1 channel
print(single_spec.shape)  # [1, 201, 62469]
grad_time, grad_freq = np.gradient(single_spec)
print(grad_time.shape)
assert grad_time.shape == single_spec.shape


def spectrogram_to_waveform(spectrogram):
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=400,
        hop_length=200,
    )
    waveform = istft_transform(torch.tensor(spectrogram).unsqueeze(0))#.clamp_(-1, 1)
    return waveform


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    time_axis = torch.arange(NUM_FRAMES) / sample_rate

    figure, axes = plt.subplots(NUM_CHANNELS, 1)
    if NUM_CHANNELS == 1:
        axes = [axes]
    for c in range(NUM_CHANNELS):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if NUM_CHANNELS > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)


# TODO qt bindings don't load
# plot_waveform(waveform, SAMPLE_RATE)

# Perform Helmholtz decomposition on the spectrogram
# curl_free, div_free = helmholtz_decomposition(spectrogram)
# TODO include the harmonic component

# Convert the decomposed spectrograms back to waveforms
# curl_free_waveform = spectrogram_to_waveform(curl_free, waveform.shape[0])
# div_free_waveform = spectrogram_to_waveform(div_free, waveform.shape[0])
# split div_free_waveform into poloidal and toroidal components
# Save the resulting files
torchaudio.save(
    args.output_directory / "curl_free.wav", curl_free_waveform, SAMPLE_RATE
)
torchaudio.save(args.output_directory / "div_free.wav", div_free_waveform, SAMPLE_RATE)
torchaudio.save(
    args.output_directory / "harmonic.wav",
    waveform,
    SAMPLE_RATE,
)


# TODO helmholtz decomposition
def helmholtz_decomposition(spectrogram):
    # TODO verify time and frequency axis
    # Calculate the gradients along time (axis=1) and frequency (axis=0)
    # Bug? The gradients are flipped
    grad_t, grad_f = np.gradient(spectrogram)

    # Calculate the Laplacians along time and frequency
    # TODO axis
    d2_t = np.gradient(grad_t, axis=1)
    d2_f = np.gradient(grad_f, axis=0)

    # Create a low-pass filter
    lp_filter = np.ones((5, 5)) / 25

    # Apply low-pass filter to the Laplacians to obtain the curl-free component
    curl_free = convolve(d2_t + d2_f, lp_filter, mode="same")

    # Subtract the curl-free component from the original spectrogram to obtain the divergence-free component
    div_free = spectrogram - curl_free

    return curl_free, div_free



# TODO multiple channels

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy.signal import convolve
from torchaudio.transforms import Spectrogram


def main():
    parser = argparse.ArgumentParser(
        description="Perform Helmholtz decomposition on the spectrogram of an audio file."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "output_directory", type=Path, help="Path to the output directory."
    )
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)

    waveform, SAMPLE_RATE = torchaudio.load(args.input_file)

    def plot_waveform(waveform, sample_rate):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle("waveform")
        plt.show(block=False)

    # TODO qt bindings don't load
    # plot_waveform(waveform, SAMPLE_RATE)
    print(waveform.shape)
    NUM_CHANNELS, NUM_FRAMES = waveform.shape

    # Perform Helmholtz decomposition on the spectrogram
    curl_free, div_free = helmholtz_decomposition(spectrogram)
    # TODO include the harmonic component

    # Convert the decomposed spectrograms back to waveforms
    curl_free_waveform = spectrogram_to_waveform(curl_free, waveform.shape[0])
    div_free_waveform = spectrogram_to_waveform(div_free, waveform.shape[0])

    # Save the resulting files
    torchaudio.save(
        args.output_directory / "curl_free.wav", curl_free_waveform, SAMPLE_RATE
    )
    torchaudio.save(
        args.output_directory / "div_free.wav", div_free_waveform, SAMPLE_RATE
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


def spectrogram_to_waveform(spectrogram, num_channels):
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=400,
        n_hop=200,
    )
    waveform = istft_transform(torch.tensor(spectrogram).unsqueeze(0)).clamp_(-1, 1)
    return waveform[:num_channels]


if __name__ == "__main__":
    main()

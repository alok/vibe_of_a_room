import tyro
import argparse
import itertools
import random
import string
import urllib.request
from pathlib import Path
from typing import Final, Literal

import torch
import torch.nn.functional as F
import torchaudio
from einops import einsum, pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from jaxtyping import Float, Integer
from torch import LongTensor as LT
from torch import Tensor
from torch import Tensor as T
from torch import jit, nn, vmap


def main(input_file: Path):
    # Load audio file
    waveform, sample_rate = torchaudio.load(input_file)
    # sample rate: 48000
    # waveform shape: 2, 12493636
    # assert sample_rate * waveform.shape[-1] == len(video)
    # clip to nearest multiple of 8
    waveform = waveform[:, : waveform.shape[-1] // 8 * 8]
    print(waveform.shape, sample_rate)

    # Convert to power spectrogram
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, power=2)
    # spectrogram_transform = torchaudio.transforms.Spectrogram()
    # shape: 2 for speaker channels, freq, time
    power_spectrogram = spectrogram_transform(waveform)
    print(f"{power_spectrogram.shape = }")

    # Apply Hodge decomposition for each channel
    div_free_list, curl_free_list, harmonic_list = [], [], []
    for channel in range(power_spectrogram.shape[0]):
        vector_field = power_spectrogram[channel, :, :]
        print(f"{vector_field.shape = }")
        div_free, curl_free, harmonic = hodge_decomposition(vector_field)
        div_free_list.append(div_free)
        curl_free_list.append(curl_free)
        harmonic_list.append(harmonic)

    # Stack the results for each channel
    div_free = torch.stack(div_free_list)
    curl_free = torch.stack(curl_free_list)
    harmonic = torch.stack(harmonic_list)

    # Convert components back to waveforms
    div_free_waveform = spectrogram_to_waveform(div_free, sample_rate)
    curl_free_waveform = spectrogram_to_waveform(curl_free, sample_rate)
    harmonic_waveform = spectrogram_to_waveform(harmonic, sample_rate)

    # Save components as WAV files
    save_component_as_wav(div_free_waveform, sample_rate, "div_free")
    save_component_as_wav(curl_free_waveform, sample_rate, "curl_free")
    save_component_as_wav(harmonic_waveform, sample_rate, "harmonic")


def hodge_decomposition(
    vector_field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute divergence
    div = torch.zeros_like(vector_field)
    div[1:, :] += vector_field[1:, :] - vector_field[:-1, :]
    div[:, 1:] += vector_field[:, 1:] - vector_field[:, :-1]
    print(f'{div.shape=}')
# [1025, 12201]
    # Compute curl
    curl = torch.zeros_like(vector_field)
    print(f"{curl.shape = }")
    print(f'{vector_field[:-1, 1:].shape=}')
    print(f'{vector_field[:-1, :-1].shape=}')
    print(f'{curl[-1:, :].shape=}')
    curl[-1:, :] -= vector_field[:-1, 1:] - vector_field[:-1, :-1]
    curl[:, -1:] += vector_field[1:, :-1] - vector_field[:-1, :-1]

    # Compute Laplacian
    laplacian = torch.zeros_like(vector_field)
    laplacian[1:-1, :] += (
        vector_field[2:, :] - 2 * vector_field[1:-1, :] + vector_field[:-2, :]
    )
    laplacian[:, 1:-1] += (
        vector_field[:, 2:] - 2 * vector_field[:, 1:-1] + vector_field[:, :-2]
    )

    # Compute harmonic component
    harmonic = laplacian / 4

    # Compute divergence-free and curl-free components
    div_free = vector_field - harmonic
    curl_free = div_free - curl

    return div_free, curl_free, harmonic


def spectrogram_to_waveform(spectrogram: torch.Tensor) -> torch.Tensor:
    # Convert power spectrogram back to amplitude spectrogram
    amplitude_spectrogram = torch.sqrt(spectrogram)

    # Convert amplitude spectrogram back to waveform
    istft_transform = torchaudio.transforms.InverseSpectrogram(n_fft=2048)
    waveform = istft_transform(amplitude_spectrogram.unsqueeze(0))

    return waveform


def save_component_as_wav(
    component_waveform: torch.Tensor, sample_rate: int, prefix: str
):
    slug = random.randint(1, 1_000)
    output_file = args.output_dir / f"{prefix}_{slug}.wav"
    torchaudio.save(output_file, component_waveform, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the vibe of a room using Hodge decomposition"
    )
    parser.add_argument("input_file", type=Path, help="Path to the input audio file")
    parser.add_argument("output_dir", type=Path, help="Path to output folder.")
    args = parser.parse_args()

    main(args.input_file)

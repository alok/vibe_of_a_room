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
import tyro
from einops import einsum, pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from jaxtyping import Float, Integer
from torch import LongTensor as LT
from torch import Tensor
from torch import Tensor as T
from torch import jit, nn, vmap

default_args = ["data/24-7.wav", "data/output/"]


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
    vf = vector_field

    # TODO div = (V[r+1,c] - V[r,c] + V[r,c+1] - V[r,c])/2 where V
    div = torch.zeros_like(vf)
    div[1:, :] += vf[1:, :] - vf[:-1, :]
    div[:, 1:] += vf[:, 1:] - vf[:, :-1]
    print(f"{div.shape=}")  # [1025, 12201]

    curl = torch.zeros_like(vf)
    print(f"{curl.shape = }")
    print(f"{vf[:-1, 1:].shape=}")
    print(f"{vf[:-1, :-1].shape=}")
    print(f"{curl[-1:, :].shape=}")
    curl[-1:, :] -= vf[:-1, 1:] - vf[:-1, :-1]
    curl[:, -1:] += vf[1:, :-1] - vf[:-1, :-1]

    laplacian = torch.zeros_like(vf)
    laplacian[1:-1, :] += vf[2:, :] - 2 * vf[1:-1, :] + vf[:-2, :]
    laplacian[:, 1:-1] += vf[:, 2:] - 2 * vf[:, 1:-1] + vf[:, :-2]

    harmonic = laplacian / 4

    div_free = vf - harmonic
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


def div(vec_field: torch.Tensor) -> torch.Tensor:
    vf = vec_field
    div = (vf[1:, :] - vf[:-1, :] + vf[:, 1:] - vf[:, :-1]) / 2
    return div


def test_div() -> bool:
    vf = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert div(vf).shape == (2, 2)
    assert torch.allclose(div(vf), torch.tensor([[2, 2], [2, 2]]))
    return True

# -abc = -a -b -c
# --abc 
# abc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the vibe of a room using Hodge decomposition"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to the input audio file",
        default=default_args[0],
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to output folder.",
        default=default_args[1],
    )
    args = parser.parse_args()
    test_div()

    main(args.input_file)

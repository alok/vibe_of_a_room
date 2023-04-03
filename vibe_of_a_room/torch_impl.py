import argparse

import torch
import torchaudio


# Load WAV file and compute the STFT
def load_wav_and_compute_stft(file_path, window_size, overlap):
    waveform, samplerate = torchaudio.load(file_path)
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=window_size, hop_length=overlap, power=None
    )
    stft_data = stft_transform(waveform)
    return stft_data, samplerate


# Save the harmonic part as a new WAV file
def save_harmonic_wav(harmonic_stft, samplerate, output_file, window_size, overlap):
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=window_size, hop_length=overlap
    )
    harmonic_waveform = istft_transform(harmonic_stft)
    torchaudio.save(output_file, harmonic_waveform, samplerate)


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Compute the harmonic part of a WAV file using Helmholtz decomposition."
)
parser.add_argument("input_file", type=str, help="Input WAV file")
parser.add_argument(
    "output_file", type=str, help="Output WAV file with the harmonic part"
)
args = parser.parse_args()

# Define the window size and overlap for the STFT
window_size = 1024
overlap = window_size // 2

# Load the WAV file and compute the STFT
stft_data, samplerate = load_wav_and_compute_stft(args.input_file, window_size, overlap)

# Compute the gradient of the STFT data
_, gradient_y, gradient_x = torch.gradient(stft_data)


# Helmholtz decomposition function using torch functions
def helmholtz_decomposition(gradient_x, gradient_y):
    kx = torch.fft.fftfreq(gradient_x.shape[-1], 1.0 / gradient_x.shape[-1])
    ky = torch.fft.fftfreq(gradient_y.shape[-2], 1.0 / gradient_y.shape[-2])
    KX, KY = torch.meshgrid(kx, ky)

    F_gradient_x = torch.fft.fft2(gradient_x)
    F_gradient_y = torch.fft.fft2(gradient_y)

    denom = KX**2 + KY**2
    denom[0, 0] = 1

    F_curl_free_x = (KX**2 * F_gradient_x + KX * KY * F_gradient_y) / denom
    F_curl_free_y = (KX * KY * F_gradient_x + KY**2 * F_gradient_y) / denom

    F_div_free_x = (-KX * KY * F_gradient_x + KX**2 * F_gradient_y) / denom
    F_div_free_y = (-(KY**2) * F_gradient_x + KX * KY * F_gradient_y) / denom

    F_harmonic_x = F_gradient_x - F_curl_free_x - F_div_free_x
    F_harmonic_y = F_gradient_y - F_curl_free_y - F_div_free_y

    curl_free_x = torch.real(torch.fft.ifft2(F_curl_free_x))
    curl_free_y = torch.real(torch.fft.ifft2(F_curl_free_y))

    div_free_x = torch.real(torch.fft.ifft2(F_div_free_x))
    div_free_y = torch.real(torch.fft.ifft2(F_div_free_y))

    harmonic_x = torch.real(torch.fft.ifft2(F_harmonic_x))
    harmonic_y = torch.real(torch.fft.ifft2(F_harmonic_y))

    return curl_free_x, curl_free_y, div_free_x, div_free_y, harmonic_x, harmonic_y


# Perform the Helmholtz decomposition
(
    curl_free_x,
    curl_free_y,
    div_free_x,
    div_free_y,
    harmonic_x,
    harmonic_y,
) = helmholtz_decomposition(gradient_x, gradient_y)

# Use the harmonic components directly to reconstruct the audio
harmonic_stft = harmonic_x + 1j * harmonic_y

# Save the harmonic part as a new WAV file
save_harmonic_wav(harmonic_stft, samplerate, args.output_file, window_size, overlap)

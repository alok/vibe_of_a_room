import jax
from pathlib import Path
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from jax.numpy import fft

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Compute the harmonic part of a WAV file using Helmholtz decomposition."
)
parser.add_argument("input_file", type=Path, help="Input WAV file")
parser.add_argument(
    "output_file", type=Path, help="Output WAV file with the harmonic part"
)
args = parser.parse_args()

# Define the window size and overlap for the STFT
WINDOW_SIZE: int = 1024
OVERLAP: int = WINDOW_SIZE // 2


def load_wav_and_compute_spectrogram(file_path, window_size, overlap):
    data, samplerate = sf.read(file_path)
    data = jnp.array(data.T[0])
    print(data.shape)
    _, _, spectrogram_data = jax.scipy.signal.stft(
        data, samplerate, nperseg=window_size, noverlap=overlap
    )
    print(spectrogram_data.shape)
    return spectrogram_data, samplerate


# Load the WAV file and compute the spectrogram
spectrogram_data, samplerate = load_wav_and_compute_spectrogram(
    args.input_file, WINDOW_SIZE, OVERLAP
)

# Compute the gradient of the spectrogram data
# gradient_y, gradient_x = jnp.gradient(spectrogram_data)
gradient_time, gradient_freq = jnp.gradient(spectrogram_data)


def helmholtz_decomposition(gradient_x, gradient_y):
    kx = fft.fftfreq(gradient_x.shape[1], 1.0 / gradient_x.shape[1])
    ky = fft.fftfreq(gradient_y.shape[0], 1.0 / gradient_y.shape[0])
    KX, KY = jnp.meshgrid(kx, ky)

    F_gradient_x = fft.fft2(gradient_x)
    F_gradient_y = fft.fft2(gradient_y)

    denom = KX**2 + KY**2
    denom = denom.at[0, 0].set(1)

    F_curl_free_x = (KX**2 * F_gradient_x + KX * KY * F_gradient_y) / denom
    F_curl_free_y = (KX * KY * F_gradient_x + KY**2 * F_gradient_y) / denom

    F_div_free_x = (-KX * KY * F_gradient_x + KX**2 * F_gradient_y) / denom
    F_div_free_y = (-(KY**2) * F_gradient_x + KX * KY * F_gradient_y) / denom

    F_harmonic_x = F_gradient_x - F_curl_free_x - F_div_free_x
    F_harmonic_y = F_gradient_y - F_curl_free_y - F_div_free_y

    curl_free_x = jnp.real(fft.ifft2(F_curl_free_x))
    curl_free_y = jnp.real(fft.ifft2(F_curl_free_y))

    div_free_x = jnp.real(fft.ifft2(F_div_free_x))
    div_free_y = jnp.real(fft.ifft2(F_div_free_y))

    harmonic_x = jnp.real(fft.ifft2(F_harmonic_x))
    harmonic_y = jnp.real(fft.ifft2(F_harmonic_y))

    return curl_free_x, curl_free_y, div_free_x, div_free_y, harmonic_x, harmonic_y


# Perform the Helmholtz decomposition
(
    curl_free_x,
    curl_free_y,
    div_free_x,
    div_free_y,
    harmonic_x,
    harmonic_y,
) = helmholtz_decomposition(gradient_time, gradient_freq)


# Define a function for plotting vector fields
def plot_vector_field(ax, field_x, field_y, title):
    ax.quiver(
        np.arange(0, 100, 5),
        np.arange(0, 100, 5),
        field_x[::5, ::5],
        field_y[::5, ::5],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.set_title(title)


# # Visualize the original vector field and the decomposed components
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# plot_vector_field(axes[0, 0], gradient_time, gradient_freq, "Original Vector Field")
# plot_vector_field(axes[0, 1], curl_free_x, curl_free_y, "Curl-free Component")
# plot_vector_field(axes[1, 0], div_free_x, div_free_y, "Divergence-free Component")
# plot_vector_field(axes[1, 1], harmonic_x, harmonic_y, "Harmonic Component")
# plt.show()


# Save the harmonic part as a new WAV file
def save_harmonic_wav(
    harmonic_spectrogram, samplerate, output_file, window_size, overlap
):
    _, reconstructed_audio = jax.scipy.signal.istft(
        harmonic_spectrogram, samplerate, nperseg=window_size, noverlap=overlap
    )
    sf.write(output_file, reconstructed_audio, samplerate)


# Perform the Helmholtz decomposition
(
    curl_free_x,
    curl_free_y,
    div_free_x,
    div_free_y,
    harmonic_x,
    harmonic_y,
) = helmholtz_decomposition(gradient_time, gradient_freq)

# Combine the harmonic components to form the harmonic spectrogram
harmonic_spectrogram = jnp.sqrt(harmonic_x**2 + harmonic_y**2)

# Save the harmonic part as a new WAV file
save_harmonic_wav(
    harmonic_spectrogram, samplerate, args.output_file / 'harmonic.wav', WINDOW_SIZE, OVERLAP
)

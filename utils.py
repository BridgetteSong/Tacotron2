import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import librosa
import torch.utils.data
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

from scipy.io.wavfile import read
import hparams as hp
import sys

from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")
is_pytorch_16plus = LooseVersion(torch.__version__) >= LooseVersion("1.6")
is_pytorch_15plus = LooseVersion(torch.__version__) >= LooseVersion("1.5")


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def stream(message) :
    sys.stdout.write(f"\r{message}")


def load_wav(full_path):
    sr, data = read(full_path)
    return data, sr


def reconstruct_waveform(mel, compression="log", n_iter=60):
    """
    Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform.
    """
    if compression == "log10":
        amp_mel = dynamic_range_decompression_torch_pow10(mel)
    elif compression == "log":
        amp_mel = dynamic_range_decompression_torch_exp(mel)
    else:
        raise ValueError("not supported compression type {}".format(compression))
    amp_mel = amp_mel.data.cpu().numpy()
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=hp.sampling_rate,
        n_fft=hp.filter_length, fmin=hp.mel_fmin, fmax=hp.mel_fmax)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hp.hop_length, win_length=hp.win_length)
    return wav

mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch_log10(x, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val))


def dynamic_range_decompression_torch_pow10(x):
    return torch.pow(10.0, x)


def dynamic_range_compression_torch_log(x, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val))


def dynamic_range_decompression_torch_exp(x):
    return torch.exp(x)


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, compression="log"):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), [int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)], mode='reflect')
    y = y.squeeze(1)

    if is_pytorch_17plus:
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    else:
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-5)

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    if compression == "log":
        spec = dynamic_range_compression_torch_log(spec)
    elif compression == "log10":
        spec = dynamic_range_compression_torch_log10(spec)
    else:
        raise ValueError("not supported compression type {}".format(compression))
    return spec

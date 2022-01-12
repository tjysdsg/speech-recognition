import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import math


def segment(sig, sample_rate=16000, frame_len=0.025, frame_step=0.01):
    slen = sig.size
    frame_len1 = int(frame_len * sample_rate)
    frame_step1 = int(frame_step * sample_rate)
    num_frames = 1
    if slen > frame_len:
        num_frames = math.ceil(slen / frame_step1)

    final_len = int((num_frames - 1) * frame_step1 + frame_len1)

    pad_sig = np.concatenate([sig, np.zeros(final_len - slen)])
    frames = np.zeros((num_frames, frame_len1))
    for i in range(num_frames):
        frames[i, :] = pad_sig[i * frame_step1:i * frame_step1 + frame_len1]

    return frames


def zero_padding(frames, frame_len=None):
    width = frames.shape[1]
    height = frames.shape[0]

    # if frame_len is not specified, find the next power of 2
    if frame_len is None:
        frame_len = 1 << (width - 1).bit_length()

    pad_len = frame_len - width
    pad_len_left = pad_len // 2
    pad_len_right = pad_len - pad_len_left

    f = np.zeros((frames.shape[0], frame_len))
    for i in range(0, height):
        f[i, pad_len_left:frame_len - pad_len_right] = frames[i, :]
    return f


def mfcc_features(path_file, frame_size=0.025, frame_stride=0.01, low_freq=80, high_freq=None):
    sample_rate, signal = wavfile.read(path_file)
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frames = segment(emphasized_signal, sample_rate, frame_size, frame_stride)
    frames = zero_padding(frames)

    # hamming window
    frames *= np.hamming(frames.shape[1])

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

    nfilt = 40
    high_freq = high_freq or sample_rate / 2
    low_freq_mel = (2595 * np.log10(1 + low_freq / 700))  # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + high_freq / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = np.log10(filter_banks)

    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
    return filter_banks, mfcc


def standardize(data):
    data -= (np.mean(data, axis=0))
    data = data / np.std(data, axis=0)
    return data

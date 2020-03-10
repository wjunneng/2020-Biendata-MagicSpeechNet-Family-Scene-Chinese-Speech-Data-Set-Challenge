# source: https://pytorch.org/audio/_modules/torchaudio/

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import math


def create_fb_matrix(n_freqs, f_min, f_max, n_mels):
    # type: (int, float, float, int) -> Tensor
    r""" Create a frequency bin conversion matrix.
    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency
        f_max (float): Maximum frequency
        n_mels (int): Number of mel filterbanks
    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """
    # freq bins
    freqs = torch.linspace(f_min, f_max, n_freqs)
    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 0. if f_min == 0 else 2595. * math.log10(1. + (f_min / 700.))
    m_max = 2595. * math.log10(1. + (f_max / 700.))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700. * (10 ** (m_pts / 2595.) - 1.)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    fb = fb.clone().detach()
    del freqs, m_pts, f_pts, f_diff, slopes, zero, down_slopes, up_slopes
    return fb


def mel_scale(spectrogram, n_mels=128, sample_rate=16000, f_min=0., f_max=None, n_stft=None):
    r"""This turns a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.
    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).
    Args:
        spectrogram (torch.Tensor): A spectrogram STFT of dimension (channel, freq, time)
        n_mels (int): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int): Sample rate of audio signal. (Default: ``16000``)
        f_min (float): Minimum frequency. (Default: ``0.``)
        f_max (float, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`.
    Returns:
        torch.Tensor: Mel frequency spectrogram of size (channel, ``n_mels``, time)
    """
    spectrogram = spectrogram.clone().detach()
    f_max = f_max if f_max is not None else float(sample_rate // 2)
    assert f_min <= f_max, 'Require f_min: %f < f_max: %f' % (f_min, f_max)

    if n_stft is None:
        fb = create_fb_matrix(spectrogram.size(1), f_min, f_max, n_mels)
    else:
        fb = create_fb_matrix(n_stft, f_min, f_max, n_mels)

    # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
    # -> (channel, time, n_mels).transpose(...)
    mel = torch.matmul(spectrogram.transpose(1, 2), fb).transpose(1, 2)
    mel = mel.clone().detach()

    del fb, spectrogram
    return mel

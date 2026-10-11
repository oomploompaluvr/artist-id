"""NumPy port of the Oliva & Torralba (2001) GIST descriptor.

This follows Torralba's reference MATLAB code (``LMgist.m``):

1. Pre-filtering - log intensity, whitening (subtract a low-pass version) and
   local contrast normalisation, so the descriptor is insensitive to global
   illumination.
2. A bank of Gabor filters built directly in the frequency domain:
   4 scales x 8 orientations = 32 filters.
3. For every filter the magnitude response is averaged over a 4 x 4 grid of
   blocks, giving 32 x 16 = 512 numbers that summarise the dominant
   orientations and scales in each part of the image (the "spatial envelope"
   from which naturalness, openness, roughness, expansion and ruggedness are
   estimated).

The original C implementation (lear-gist) needs FFTW, which is painful on Windows;
this version only needs NumPy and gives the same structure of descriptor.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.fft as sfft

ORIENTATIONS_PER_SCALE = (8, 8, 8, 8)
NUMBER_BLOCKS = 4
FC_PREFILT = 4
BOUNDARY = 32
IMAGE_SIZE = 256


@lru_cache(maxsize=4)
def gabor_bank(orientations: tuple = ORIENTATIONS_PER_SCALE, n: int = IMAGE_SIZE + 2 * BOUNDARY) -> np.ndarray:
    """Transfer functions of the Gabor filters, shape (n_filters, n, n)."""
    params = []
    for i, n_or in enumerate(orientations):
        for j in range(n_or):
            params.append((0.35, 0.3 / (1.85 ** i), 16 * n_or ** 2 / 32 ** 2, np.pi / n_or * j))

    fx, fy = np.meshgrid(np.arange(-n // 2, n // 2), np.arange(-n // 2, n // 2))
    fr = np.fft.fftshift(np.sqrt(fx ** 2 + fy ** 2))
    t = np.fft.fftshift(np.angle(fx + 1j * fy))

    bank = np.empty((len(params), n, n), dtype=np.float32)
    for k, (a, f0, b, theta) in enumerate(params):
        tr = t + theta
        tr = tr + 2 * np.pi * (tr < -np.pi) - 2 * np.pi * (tr > np.pi)
        bank[k] = np.exp(-10 * a * (fr / n / f0 - 1) ** 2 - 2 * b * np.pi * tr ** 2)
    return bank


def prefilter(img: np.ndarray, fc: float = FC_PREFILT) -> np.ndarray:
    """Whitening + local contrast normalisation (``prefilt.m``)."""
    w = 5
    s1 = fc / np.sqrt(np.log(2))
    img = np.log(img.astype(np.float64) + 1)
    img = np.pad(img, w, mode="symmetric")
    sn, sm = img.shape
    n = max(sn, sm)
    n += n % 2
    img = np.pad(img, ((0, n - sn), (0, n - sm)), mode="symmetric")

    fx, fy = np.meshgrid(np.arange(-n // 2, n // 2), np.arange(-n // 2, n // 2))
    gf = np.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / s1 ** 2))

    out = img - np.real(np.fft.ifft2(np.fft.fft2(img) * gf))
    local_std = np.sqrt(np.abs(np.fft.ifft2(np.fft.fft2(out ** 2) * gf)))
    out = out / (0.2 + local_std)
    return out[w:sn - w, w:sm - w]


def _block_mean(x: np.ndarray, n_blocks: int) -> np.ndarray:
    """Average of each of the n_blocks x n_blocks tiles, for a stack (k, H, W) -> (k, n, n)."""
    k, ny, nx = x.shape
    ys = np.floor(np.linspace(0, ny, n_blocks + 1)).astype(int)
    xs = np.floor(np.linspace(0, nx, n_blocks + 1)).astype(int)
    out = np.empty((k, n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            out[:, i, j] = x[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean(axis=(1, 2))
    return out


def gabor_responses(gray: np.ndarray) -> np.ndarray:
    """Magnitude of every Gabor filter response, shape (n_filters, H, W)."""
    img = prefilter(gray)
    bank = gabor_bank(n=img.shape[0] + 2 * BOUNDARY)
    padded = np.pad(img, BOUNDARY, mode="symmetric").astype(np.float32)
    F = sfft.fft2(padded)
    resp = np.abs(sfft.ifft2(F[None] * bank, axes=(-2, -1)))
    return resp[:, BOUNDARY:-BOUNDARY, BOUNDARY:-BOUNDARY]


def gist(gray: np.ndarray, n_blocks: int = NUMBER_BLOCKS) -> np.ndarray:
    """512-d GIST descriptor of a square grayscale uint8 image (256 x 256)."""
    blocks = _block_mean(gabor_responses(gray), n_blocks)
    # Order: filter-major, then blocks column-major (matches LMgist's reshape).
    return blocks.transpose(0, 2, 1).reshape(-1).astype(np.float32)

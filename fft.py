#!/usr/bin/env python3

import argparse
import math
import time
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Optional image libraries
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image
    _HAS_CV2 = False


# ---------------------- Utility functions ----------------------

def next_power_of_two(n: int) -> int:
    return 1 if n == 0 else 2 ** ((n - 1).bit_length())


def pad_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    th, tw = target_shape
    out = np.zeros((th, tw), dtype=image.dtype)
    out[:h, :w] = image
    return out


# ---------------------- DFT / FFT implementations ----------------------

def naive_dft(x: np.ndarray) -> np.ndarray:
    """Compute the DFT of 1D complex-valued array x using the naive O(N^2) formula."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        s = 0+0j
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            s += x[n] * complex(math.cos(angle.imag), math.sin(angle.imag)) if False else x[n] * np.exp(angle)
        X[k] = s
    return X


def naive_idft(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    x = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        s = 0+0j
        for k in range(N):
            angle = 2j * math.pi * k * n / N
            s += X[k] * np.exp(angle)
        x[n] = s / N
    return x


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def cooley_tukey_fft(x: np.ndarray) -> np.ndarray:
    """Recursive Cooley-Tukey FFT. Pads input to power of two length if necessary.
    Returns X (complex numpy array) of same length as padded input.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N == 0:
        return np.array([], dtype=np.complex128)
    if N == 1:
        return x.copy()
    if not is_power_of_two(N):
        # pad to next power of two
        M = next_power_of_two(N)
        x_padded = np.zeros(M, dtype=np.complex128)
        x_padded[:N] = x
        return cooley_tukey_fft(x_padded)
    # recursive case
    even = cooley_tukey_fft(x[0::2])
    odd = cooley_tukey_fft(x[1::2])
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        twiddle = np.exp(-2j * math.pi * k / N) * odd[k]
        X[k] = even[k] + twiddle
        X[k + N // 2] = even[k] - twiddle
    return X


def inverse_fft(X: np.ndarray) -> np.ndarray:
    """Inverse FFT computed via conjugation trick: ifft(X) = conj( fft(conj(X)) ) / N"""
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    if N == 0:
        return np.array([], dtype=np.complex128)
    x = cooley_tukey_fft(np.conjugate(X))
    x = np.conjugate(x) / float(N)
    return x


# ---------------------- 2D transforms ----------------------

def fft2d(matrix: np.ndarray, use_naive: bool = False) -> np.ndarray:
    """Compute 2D DFT by applying 1D transform on rows then columns."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape
    # Transform rows
    out = np.zeros((h, w), dtype=np.complex128)
    for r in range(h):
        if use_naive:
            out[r, :] = naive_dft(matrix[r, :])
        else:
            out[r, :] = cooley_tukey_fft(matrix[r, :])
    # Transform columns
    out2 = np.zeros_like(out)
    for c in range(w):
        if use_naive:
            out2[:, c] = naive_dft(out[:, c])
        else:
            out2[:, c] = cooley_tukey_fft(out[:, c])
    return out2


def ifft2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape
    # inverse columns then inverse rows
    out = np.zeros((h, w), dtype=np.complex128)
    for c in range(w):
        out[:, c] = inverse_fft(matrix[:, c])
    out2 = np.zeros_like(out)
    for r in range(h):
        out2[r, :] = inverse_fft(out[r, :])
    return out2


# ---------------------- Image helpers ----------------------

def load_grayscale_image(path: str) -> np.ndarray:
    """Return a 2D numpy array (float64) normalized to [0,1]."""
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image '{path}' with cv2")
        return img.astype(np.float64) / 255.0
    else:
        img = Image.open(path).convert('L')
        arr = np.array(img).astype(np.float64) / 255.0
        return arr


def show_image_pair(left: np.ndarray, right: np.ndarray, titles=("Original","FFT")):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(left, cmap='gray', vmin=0, vmax=1)
    plt.title(titles[0])
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(right, cmap='gray', norm=LogNorm())
    plt.title(titles[1])
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------- Modes implementations ----------------------

def mode_display_fft(img: np.ndarray):
    h, w = img.shape
    # pad to powers of two
    target_h = next_power_of_two(h)
    target_w = next_power_of_two(w)
    padded = pad_to_shape(img, (target_h, target_w))
    F = fft2d(padded, use_naive=False)
    magnitude = np.abs(F)
    # shift zero-frequency to center for visualization
    magnitude_shifted = np.fft.fftshift(magnitude)
    show_image_pair(padded, magnitude_shifted, titles=("Padded image", "|FFT| (log scale)"))


def mode_denoise(img: np.ndarray, cutoff_ratio=0.1):
    # cutoff_ratio: fraction of smallest dimension used as radius (e.g., 0.1 -> small low-pass area)
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))
    F = fft2d(padded)
    # shift
    Fshift = np.fft.fftshift(F)
    # create low-pass mask
    cy, cx = th // 2, tw // 2
    radius = int(min(th, tw) * cutoff_ratio)
    Y, X = np.ogrid[:th, :tw]
    mask = (X - cx)**2 + (Y - cy)**2 <= radius*radius
    before_nonzero = np.count_nonzero(F)
    Fshift_filtered = Fshift * mask
    after_nonzero = np.count_nonzero(Fshift_filtered)
    print(f"DENOISE: kept {after_nonzero}/{F.size} coefficients ({after_nonzero/F.size:.4f})")
    # inverse shift
    Funshifted = np.fft.ifftshift(Fshift_filtered)
    recon = ifft2d(Funshifted).real
    # crop back to original
    recon_cropped = recon[:h, :w]
    show_image_pair(img, np.clip(recon_cropped, 0, 1), titles=("Original","Denoised (low-pass)"))


def mode_compress(img: np.ndarray):
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))
    F = fft2d(padded)
    mag = np.abs(F).flatten()
    N = mag.size
    # compression levels (fractions of coefficients kept)
    levels = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
    images = []
    nonzeros = []
    for frac in levels:
        if frac >= 1.0:
            mask = np.ones(F.shape, dtype=bool)
        else:
            k = max(1, int(N * frac))
            # keep the k largest magnitudes (global)
            thresh = np.partition(mag, -k)[-k]
            mask = np.abs(F) >= thresh
        Fmod = F * mask
        nonzero_count = np.count_nonzero(Fmod)
        nonzeros.append(nonzero_count)
        recon = ifft2d(Fmod).real
        images.append(np.clip(recon[:h,:w], 0, 1))
        print(f"Compression {frac*100:.3g}% -> nonzeros: {nonzero_count}/{N} ({nonzero_count/N:.6f})")
    # plot 2x3
    plt.figure(figsize=(12,8))
    for i, img_i in enumerate(images):
        plt.subplot(2,3,i+1)
        plt.imshow(img_i, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Keep {levels[i]*100:.3g}% ({nonzeros[i]} coeffs)")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def mode_runtime_plot(max_power=10, trials=10):
    # compute runtimes for square arrays sizes 2^5 .. 2^max_power
    sizes = [2**p for p in range(5, max_power+1)]
    naive_means = []
    naive_stds = []
    fft_means = []
    fft_stds = []
    for N in sizes:
        print(f"Measuring N={N}x{N}")
        naive_times = []
        fft_times = []
        for t in range(trials):
            A = np.random.randn(N, N)
            # time naive (we will limit naive to reasonably small N to avoid huge runtimes)
            if N <= 256:  # naive for up to 256x256 might still be slow but tolerable depending on machine
                start = time.time()
                _ = fft2d(A, use_naive=True)
                naive_times.append(time.time() - start)
            else:
                naive_times.append(float('nan'))
            # time fft
            start = time.time()
            _ = fft2d(A, use_naive=False)
            fft_times.append(time.time() - start)
        naive_times_np = np.array(naive_times)
        fft_times_np = np.array(fft_times)
        naive_means.append(np.nanmean(naive_times_np))
        naive_stds.append(np.nanstd(naive_times_np))
        fft_means.append(np.mean(fft_times_np))
        fft_stds.append(np.std(fft_times_np))
    # plotting
    plt.figure()
    plt.errorbar(sizes, fft_means, yerr=fft_stds, label='FFT (cooley-tukey)')
    # plot naive only where available
    sizes_naive = [s for s, m in zip(sizes, naive_means) if not math.isnan(m)]
    naive_means_plot = [m for m in naive_means if not math.isnan(m)]
    naive_stds_plot = [s for s in naive_stds if not math.isnan(s)]
    if sizes_naive:
        plt.errorbar(sizes_naive, naive_means_plot, yerr=naive_stds_plot, label='Naive DFT')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Problem size (N, square matrix N x N)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime comparison: Naive DFT vs Cooley-Tukey FFT')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()


# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description='FFT assignment tool')
    parser.add_argument('-m', '--mode', type=int, choices=[1,2,3,4], default=1,
                        help='Mode: 1 show FFT, 2 denoise, 3 compress, 4 runtime plots')
    parser.add_argument('-i', '--image', type=str, default=None, help='Image filename')
    parser.add_argument('--cutoff', type=float, default=0.08, help='Denoise cutoff ratio (default 0.08)')
    parser.add_argument('--maxpow', type=int, default=10, help='Max power for runtime plotting (2^maxpow)')
    parser.add_argument('--trials', type=int, default=6, help='Trials per problem size (runtime mode)')
    args = parser.parse_args()

    if args.mode in (1,2,3) and args.image is None:
        print('Error: modes 1-3 require an image filename via -i', file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.image:
        img = load_grayscale_image(args.image)
    else:
        img = None

    if args.mode == 1:
        mode_display_fft(img)
    elif args.mode == 2:
        mode_denoise(img, cutoff_ratio=args.cutoff)
    elif args.mode == 3:
        mode_compress(img)
    elif args.mode == 4:
        mode_runtime_plot(max_power=args.maxpow, trials=args.trials)


if __name__ == '__main__':
    main()

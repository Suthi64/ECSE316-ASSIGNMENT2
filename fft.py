import argparse
import math
import sys
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image



#Utility functions

def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    return 1 if n == 0 else 2 ** ((n - 1).bit_length())


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def pad_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Pad a 2D array with zeros up to target_shape (height, width)."""
    h, w = image.shape
    th, tw = target_shape
    out = np.zeros((th, tw), dtype=image.dtype)
    out[:h, :w] = image
    return out


def fftshift(x: np.ndarray) -> np.ndarray:
    """Shift zero-frequency component to center of spectrum."""
    if x.ndim == 1:
        n = x.shape[0]
        return np.roll(x, n // 2)
    elif x.ndim == 2:
        h, w = x.shape
        return np.roll(np.roll(x, h // 2, axis=0), w // 2, axis=1)
    else:
        raise ValueError("fftshift only supports 1D and 2D arrays")


def ifftshift(x: np.ndarray) -> np.ndarray:
    """Inverse of fftshift."""
    if x.ndim == 1:
        n = x.shape[0]
        return np.roll(x, -(n // 2))
    elif x.ndim == 2:
        h, w = x.shape
        return np.roll(np.roll(x, -(h // 2), axis=0), -(w // 2), axis=1)
    else:
        raise ValueError("ifftshift only supports 1D and 2D arrays")


# 1D DFT / FFT implementations

def naive_dft(x: np.ndarray) -> np.ndarray:
    """Naive O(N^2) 1D DFT."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        s = 0.0 + 0.0j
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            s += x[n] * np.exp(angle)
        X[k] = s
    return X


def naive_idft(X: np.ndarray) -> np.ndarray:
    """Naive O(N^2) 1D inverse DFT."""
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    x = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        s = 0.0 + 0.0j
        for k in range(N):
            angle = 2j * math.pi * k * n / N
            s += X[k] * np.exp(angle)
        x[n] = s / N
    return x


def cooley_tukey_fft(x: np.ndarray) -> np.ndarray:
    """
    Recursive Cooley–Tukey FFT.
    Assumes len(x) is a power of two.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N == 0:
        return np.array([], dtype=np.complex128)
    if N == 1:
        return x.copy()
    if not is_power_of_two(N):
        raise ValueError(f"cooley_tukey_fft: length {N} is not a power of two")

    even = cooley_tukey_fft(x[0::2])
    odd = cooley_tukey_fft(x[1::2])

    X = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        twiddle = np.exp(-2j * math.pi * k / N) * odd[k]
        X[k] = even[k] + twiddle
        X[k + N // 2] = even[k] - twiddle
    return X


def inverse_fft(X: np.ndarray) -> np.ndarray:
    """Inverse FFT via conjugate trick: ifft(X) = conj(fft(conj(X))) / N."""
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    if N == 0:
        return np.array([], dtype=np.complex128)
    x = cooley_tukey_fft(np.conjugate(X))
    x = np.conjugate(x) / float(N)
    return x


# 2D transforms

def dft2d(matrix: np.ndarray) -> np.ndarray:
    """2D DFT via naive 1D DFT on rows then columns."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape
    # rows
    tmp = np.zeros((h, w), dtype=np.complex128)
    for r in range(h):
        tmp[r, :] = naive_dft(matrix[r, :])
    # columns
    out = np.zeros_like(tmp)
    for c in range(w):
        out[:, c] = naive_dft(tmp[:, c])
    return out


def fft2d(matrix: np.ndarray, use_naive: bool = False) -> np.ndarray:
    """2D DFT via either naive DFT or Cooley–Tukey FFT on rows then columns."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape

    out = np.zeros((h, w), dtype=np.complex128)
    for r in range(h):
        if use_naive:
            out[r, :] = naive_dft(matrix[r, :])
        else:
            out[r, :] = cooley_tukey_fft(matrix[r, :])

    out2 = np.zeros_like(out)
    for c in range(w):
        if use_naive:
            out2[:, c] = naive_dft(out[:, c])
        else:
            out2[:, c] = cooley_tukey_fft(out[:, c])
    return out2


def ifft2d(matrix: np.ndarray) -> np.ndarray:
    """2D inverse FFT via inverse 1D FFT on rows then columns."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape

    # First: inverse FFT on rows
    out = np.zeros((h, w), dtype=np.complex128)
    for r in range(h):
        out[r, :] = inverse_fft(matrix[r, :])

    # Second: inverse FFT on columns
    out2 = np.zeros_like(out)
    for c in range(w):
        out2[:, c] = inverse_fft(out[:, c])
    return out2


# Image helpers

def load_grayscale_image(path: str) -> np.ndarray:
    """Load a grayscale image as float64 in [0,1] using PIL."""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr



def show_image_pair(left: np.ndarray, right: np.ndarray,
                    titles=("Original", "FFT (log scale)")):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(left, cmap='gray', vmin=0, vmax=1)
    plt.title(titles[0])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # Add a tiny epsilon so LogNorm is well defined
    eps = 1e-8
    plt.imshow(right + eps, cmap='gray', norm=LogNorm())
    plt.title(titles[1])
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Modes implementations 

def mode_display_fft(img: np.ndarray, fft_kind: str = 'MS'):
    """
    Mode 1.
    Show original image next to log magnitude of its 2D FFT.
    fft_kind: 'MS' (custom fft2d), 'numpy', or 'both'.
    """
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))

    # How many panels: original + 1 or 2 spectra
    ncols = 3 if fft_kind == 'both' else 2
    plt.figure(figsize=(6 * ncols, 5))

    # Panel 1. original image
    plt.subplot(1, ncols, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title("Original image")
    plt.axis('off')

    col = 2

    # Custom fft2d, match old "FFT Magnitude Spectrum" look 
    if fft_kind in ('MS', 'both'):
        F = fft2d(padded, use_naive=False)

        # log(1 + |F|) 
        mag_log = np.array([[math.log(1.0 + abs(z)) for z in row] for row in F])

        # shift DC to center
        mag_shift = fftshift(mag_log)

        # linear normalize to [0,1]
        mag_min = mag_shift.min()
        mag_max = mag_shift.max()
        mag_norm = (mag_shift - mag_min) / (mag_max - mag_min + 1e-12)

        # crop back to original size so it matches the paper figure
        mag_norm = mag_norm[:h, :w]

        plt.subplot(1, ncols, col)
        plt.imshow(mag_norm, cmap='gray', vmin=0, vmax=1)
        plt.title("Custom fft2d |F| (log)")
        plt.axis('off')

        if fft_kind == 'both':
            col += 1

    # NumPy fft2, separate styling (LogNorm etc)
    if fft_kind in ('numpy', 'both'):
        F_np = fft2d(padded, use_naive=False)
        mag_np = np.abs(F_np)
        mag_shift_np = fftshift(mag_np)

        eps = 1e-8
        plt.subplot(1, ncols, col)
        plt.imshow(mag_shift_np + eps, cmap='gray', norm=LogNorm())
        plt.title("NumPy fft2 |F| (log)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def mode_denoise(img: np.ndarray, cutoff_ratio: float = 0.08):
    """
    Mode 2.
    Denoise by low pass filtering in the frequency domain.
    cutoff_ratio controls radius as a fraction of min dimension.
    """
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))

    F = fft2d(padded, use_naive=False)
    Fshift = fftshift(F)

    cy, cx = th // 2, tw // 2
    radius = int(min(th, tw) * cutoff_ratio)

    Y, X = np.ogrid[:th, :tw]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius * radius

    total_coeffs = F.size
    before_nonzero = np.count_nonzero(F)
    Fshift_filtered = Fshift * mask
    after_nonzero = np.count_nonzero(Fshift_filtered)

    print(f"DENOISE: kept {after_nonzero}/{total_coeffs} coefficients "
          f"({after_nonzero / total_coeffs:.4f})")

    Funshift = ifftshift(Fshift_filtered)
    recon = ifft2d(Funshift).real

    recon_cropped = recon[:h, :w]
    recon_cropped = np.clip(recon_cropped, 0.0, 1.0)

    show_image_pair(img, recon_cropped,
                    titles=("Original", "Denoised (low pass)"))


def mode_compress(img: np.ndarray):
    """
    Mode 3.
    Compress by keeping only the largest Fourier coefficients by magnitude.
    Compression levels interpreted as percentage of coefficients set to zero:
      0, 50, 90, 95, 99, 99.9
    """
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))

    F = fft2d(padded, use_naive=False)
    mag = np.abs(F).flatten()
    N = mag.size

    # Compression levels in percent of coefficients dropped
    compression_levels = [0, 50, 90, 95, 99, 99.9]

    images = []
    nonzeros = []
    fractions = []

    for level in compression_levels:
        if level == 0:
            # No compression, keep all coefficients
            mask = np.ones_like(F, dtype=bool)
        else:
            keep_fraction = 1.0 - (level / 100.0)
            keep_count = max(1, int(round(N * keep_fraction)))

            # threshold so that we keep the keep_count largest magnitudes
            thresh = np.partition(mag, -keep_count)[-keep_count]
            mask = np.abs(F) >= thresh

        Fmod = F * mask
        nonzero_count = np.count_nonzero(Fmod)
        frac_nonzero = nonzero_count / N
        nonzeros.append(nonzero_count)
        fractions.append(frac_nonzero)

        print(f"Compression {level:.1f}%: "
              f"non zero coefficients {nonzero_count}/{N} "
              f"({frac_nonzero:.6f})")

        recon = ifft2d(Fmod).real
        recon_cropped = recon[:h, :w]
        recon_cropped = np.clip(recon_cropped, 0.0, 1.0)
        images.append(recon_cropped)

    # plot 2x3 grid
    plt.figure(figsize=(12, 8))
    for i, level in enumerate(compression_levels):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        plt.title(
            f"Compression {level:.1f}%\n"
            f"Non zero: {nonzeros[i]}\n"
            f"Frac: {fractions[i]:.4f}"
        )
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def mode_runtime_plot(max_power: int = 10, trials: int = 10):
    """
    Mode 4.
    Compare runtime of naive 2D DFT and FFT for N x N matrices.
    Sizes: 2^5 up to 2^max_power (default 2^10).
    """
    sizes = [2 ** p for p in range(5, max_power + 1)]
    naive_means, naive_stds = [], []
    fft_means, fft_stds = [], []

    for N in sizes:
        print(f"\nMeasuring N={N} x {N}")
        naive_times = []
        fft_times = []

        for t in range(trials):
            A = np.random.randn(N, N)

            # Naive only for moderate sizes
            if N <= 256:
                start = time.time()
                _ = fft2d(A, use_naive=True)
                naive_times.append(time.time() - start)
            else:
                naive_times.append(float("nan"))

            start = time.time()
            _ = fft2d(A, use_naive=False)
            fft_times.append(time.time() - start)

        naive_times_np = np.array(naive_times)
        fft_times_np = np.array(fft_times)

        naive_means.append(np.nanmean(naive_times_np))
        naive_stds.append(np.nanstd(naive_times_np))
        fft_means.append(np.mean(fft_times_np))
        fft_stds.append(np.std(fft_times_np))

        print(f"  Naive mean: {naive_means[-1]:.4e} s, "
              f"std: {naive_stds[-1]:.4e}")
        print(f"  FFT   mean: {fft_means[-1]:.4e} s, "
              f"std: {fft_stds[-1]:.4e}")

    # plotting
    plt.figure()
    plt.errorbar(sizes, fft_means, yerr=fft_stds, label="FFT (Cooley–Tukey)")

    # naive only where not nan
    sizes_naive = [s for s, m in zip(sizes, naive_means) if not math.isnan(m)]
    naive_means_plot = [m for m in naive_means if not math.isnan(m)]
    naive_stds_plot = [s for s in naive_stds if not math.isnan(s)]

    if sizes_naive:
        plt.errorbar(sizes_naive, naive_means_plot, yerr=naive_stds_plot,
                     label="Naive DFT")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Problem size N (matrix N x N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime comparison: Naive 2D DFT vs FFT")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# CLI

def main():
    parser = argparse.ArgumentParser(description="FFT assignment tool")
    parser.add_argument(
        "-m", "--mode", type=int, choices=[1, 2, 3, 4], default=1,
        help="Mode: 1 show FFT, 2 denoise, 3 compress, 4 runtime plots"
    )
    parser.add_argument(
        "-i", "--image", type=str, default="moonlanding.png",
        help="Image filename (default: moonlanding.png) for modes 1–3"
    )
    parser.add_argument(
        "--fft",
        type=str,
        choices=["MS", "numpy", "both"],
        default="MS",
        help="In mode 1: MS = custom fft2d spectrum, "
             "numpy = NumPy fft2, both = show both"
    )
    parser.add_argument(
        "--cutoff", type=float, default=0.08,
        help="Denoise cutoff ratio as fraction of min dimension (default 0.08)"
    )
    parser.add_argument(
        "--maxpow", type=int, default=10,
        help="Max power for runtime plotting (sizes 2^5..2^maxpow, default 10)"
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Trials per problem size in runtime mode (default 10)"
    )
    args = parser.parse_args()

    img = None
    if args.mode in (1, 2, 3):
        try:
            img = load_grayscale_image(args.image)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if args.mode == 1:
        mode_display_fft(img, fft_kind=args.fft)
    elif args.mode == 2:
        mode_denoise(img, cutoff_ratio=args.cutoff)
    elif args.mode == 3:
        mode_compress(img)
    elif args.mode == 4:
        mode_runtime_plot(max_power=args.maxpow, trials=args.trials)
    else:
        print("Invalid mode. choose 1, 2, 3, or 4")
        sys.exit(1)


if __name__ == "__main__":
    main()
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
    """Shift zero-frequency component to center of a 1D OR 2D array."""
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


# 1D DFT/FFT implementations and their Corresponding Inverse Transforms

def naive_dft(x: np.ndarray) -> np.ndarray:
    """Compute the 1D DFT."""
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
    """Compute the 1D inverse DFT."""
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
    Compute the 1D Fast Fourier Transform (FFT) using the recursive Cooley–Tukey algorithm.
    Assumes that the len(x) is a power of two.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N == 0:
        return np.array([], dtype=np.complex128)
    if N == 1:
        return x.copy()
    if not is_power_of_two(N):
        raise ValueError(f"Input length {N} must be a power of two")

    even = cooley_tukey_fft(x[0::2])
    odd = cooley_tukey_fft(x[1::2])

    X = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        twiddle = np.exp(-2j * math.pi * k / N) * odd[k]
        X[k] = even[k] + twiddle
        X[k + N // 2] = even[k] - twiddle
    return X


def inverse_fft(X: np.ndarray) -> np.ndarray:
    """Compute the 1D inverse FFT using the conjugate symmetry method."""
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    if N == 0:
        return np.array([], dtype=np.complex128)
    x = cooley_tukey_fft(np.conjugate(X))
    x = np.conjugate(x) / float(N)
    return x


# 2D transforms

def dft2d(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the 2D DFT using the naive approach.
    Applies the 1D DFT first on each row, then on each column of the input matrix.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape
    # Apply 1D DFT to each rows
    tmp = np.zeros((h, w), dtype=np.complex128)
    for r in range(h):
        tmp[r, :] = naive_dft(matrix[r, :])
    # Apply 1D DFT to each columns
    out = np.zeros_like(tmp)
    for c in range(w):
        out[:, c] = naive_dft(tmp[:, c])
    return out


def fft2d(matrix: np.ndarray, use_naive: bool = False) -> np.ndarray:
    """Compute the 2D DFT."""
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
    """Compute the 2D inverse FFT using inverse 1D FFT on rows then columns."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    h, w = matrix.shape

    # Apply 1D inverse FFT to each rows
    out = np.zeros((h, w), dtype=np.complex128)
    for r in range(h):
        out[r, :] = inverse_fft(matrix[r, :])

    # Apply 1D inverse FFT to each columns
    out2 = np.zeros_like(out)
    for c in range(w):
        out2[:, c] = inverse_fft(out[:, c])
    return out2


# Image helpers

def load_grayscale_image(path: str) -> np.ndarray:
    """Load an image in grayscale and convert it to a normalized NumPy array."""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr



def show_image_pair(left: np.ndarray, right: np.ndarray, titles=("Original", "FFT (log scale)")):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(left, cmap='gray', vmin=0, vmax=1)
    plt.title(titles[0])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    eps = 1e-8
    plt.imshow(right + eps, cmap='gray', norm=LogNorm())
    plt.title(titles[1])
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Modes implementations 

def mode_display_fft(img: np.ndarray, fft_kind: str = 'MS'):
    """
    Mode 1: Show the original image next to the log-magnitude of its 2D FFT.

    Parameters: 
    fft_kind: 'MS' for custom fft2d, 'numpy' for NumPy FFT, or 'both' to display both.
    """
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))

    # To determine the number of panels
    ncols = 3 if fft_kind == 'both' else 2
    plt.figure(figsize=(6 * ncols, 5))

    # Panel 1 is to display the original image
    plt.subplot(1, ncols, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title("Original image")
    plt.axis('off')

    col = 2

    # Custom fft2d magnitude spectrum 
    if fft_kind in ('MS', 'both'):
        F = fft2d(padded, use_naive=False)

        # Compute log-magnitude for better visualization 
        mag_log = np.array([[math.log(1.0 + abs(z)) for z in row] for row in F])

        # Shift zero-frequency component to center
        mag_shift = fftshift(mag_log)

        # Normalize values to [0,1] for consistent display
        mag_min = mag_shift.min()
        mag_max = mag_shift.max()
        mag_norm = (mag_shift - mag_min) / (mag_max - mag_min + 1e-12)

        # Crop back to original image size
        mag_norm = mag_norm[:h, :w]

        plt.subplot(1, ncols, col)
        plt.imshow(mag_norm, cmap='gray', vmin=0, vmax=1)
        plt.title("FFT2D Magnitude Spectrum")
        plt.axis('off')

        if fft_kind == 'both':
            col += 1

    # This is the panel for NumPy FFT magnitude spectrum 
    if fft_kind in ('numpy', 'both'):
        F_np = fft2d(padded, use_naive=False)
        mag_np = np.abs(F_np)
        mag_shift_np = fftshift(mag_np)

        eps = 1e-8
        plt.subplot(1, ncols, col)
        plt.imshow(mag_shift_np + eps, cmap='gray', norm=LogNorm())
        plt.title("NumPy FFT2")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def mode_denoise(img: np.ndarray, cutoff_ratio: float = 0.08):
    """
    Mode 2: Denoise by low pass filtering in the frequency domain.
    
    Parameters:
    - cutoff_ratio: controls radius as a fraction of min dimension.
    """
    mode_extra_filters(img,
                   cutoff_pixels=50,    
                   threshold=0,     
                   topk=500)           
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
    Mode 3: Compress by keeping only the largest Fourier coefficients by magnitude.
    Compression levels indicate the percentage of coefficients that are zeroed out: 0, 50, 90, 95, 99, 99.9.
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
            # Keep all coefficients so no compression
            mask = np.ones_like(F, dtype=bool)
        else:
            keep_fraction = 1.0 - (level / 100.0)
            keep_count = max(1, int(round(N * keep_fraction)))

            # To create mask for largest-magnitude coefficients
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

    # Plot the image in a 2x3 grid
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

def mode_extra_filters(img: np.ndarray,ccutoff_pixels: int = 50,cthreshold: float = 1000.0, topk: int = 500):
    """
    Show:
      - Low pass filter with radius = cutoff_pixels
      - High pass filter with radius = cutoff_pixels
      - Magnitude thresholding with given threshold
      - Keep topk Fourier coefficients by magnitude
      - Low pass + threshold combined
    Together with the original image in a 6 panel figure.
    """
    h, w = img.shape
    th, tw = next_power_of_two(h), next_power_of_two(w)
    padded = pad_to_shape(img, (th, tw))

    # Compute the FFT and shift zero-frequency to center
    F = fft2d(padded, use_naive=False)
    Fshift = fftshift(F)
    H, W = Fshift.shape

    # This a helper function to reconstruct image from masked FFT
    def reconstruct(mask: np.ndarray) -> np.ndarray:
        Fmasked = Fshift * mask
        Funshift = ifftshift(Fmasked)
        recon = ifft2d(Funshift).real
        recon = recon[:h, :w]
        recon = np.clip(recon, 0.0, 1.0)
        return recon

    # In order to build radial masks for low pass and high pass
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    lp_mask = r2 <= cutoff_pixels ** 2
    hp_mask = ~lp_mask

    # Apply low pass filter
    img_lp = reconstruct(lp_mask)

    # Apply high pass filter
    img_hp = reconstruct(hp_mask)

    # Magnitude thresholding
    mag = np.abs(Fshift)
    th_mask = mag >= threshold
    img_th = reconstruct(th_mask)

    # Keep top-k coefficients by magnitude
    flat_mag = mag.ravel()
    topk = min(topk, flat_mag.size)
    kth = np.partition(flat_mag, -topk)[-topk]  
    top_mask = mag >= kth
    img_top = reconstruct(top_mask)

    # Combine the low pass and threshold mask
    combined_mask = lp_mask & th_mask
    img_combined = reconstruct(combined_mask)

    # Plot all 6 panels in a single figure
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img_lp, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Low pass (cutoff={cutoff_pixels})")
    axes[1].axis("off")

    axes[2].imshow(img_hp, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"High pass (cutoff={cutoff_pixels})")
    axes[2].axis("off")

    axes[3].imshow(img_th, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Magnitude Thresholding (threshold={int(threshold)})")
    axes[3].axis("off")

    axes[4].imshow(img_top, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title(f"Keep Top {topk} Coefs")
    axes[4].axis("off")

    axes[5].imshow(img_combined, cmap="gray", vmin=0, vmax=1)
    axes[5].set_title(f"Low-Pass & Threshold (cutoff={cutoff_pixels}, threshold={int(threshold)})")
    axes[5].axis("off")

    plt.tight_layout()
    plt.show()


def mode_runtime_plot(max_power: int = 8, trials: int = 10):
    """
    Mode 4: Measure and plot runtime of naive 2D DFT and FFT for N x N matrices.
    Problem sizes are powers of two from 2^3 up to 2^8 (so max 256 x 256).
    """
    # Limit the maximum size to 256 x 256
    max_power = min(max_power, 8)

    sizes = [2 ** p for p in range(3, max_power + 1)]

    naive_means, naive_stds = [], []
    fft_means, fft_stds = [], []

    print("Starting Runtime Complexity Analysis for Naive DFT and FFT...")

    for N in sizes:
        print(f"\nProblem Size: {N} x {N}")
        naive_times = []
        fft_times = []

        for run in range(trials):
            # To start generate a random complex matrix
            A = np.random.randn(N, N) + 1j * np.random.randn(N, N)

            # Measure the runtime of naive 2D DFT
            start = time.time()
            _ = fft2d(A, use_naive=True)
            end = time.time()
            naive_times.append(end - start)
            print(f"  Run {run + 1} naive: {naive_times[-1]:.4f} s")

            # Measure the runtime of FFT 2D
            start = time.time()
            _ = fft2d(A, use_naive=False)
            end = time.time()
            fft_times.append(end - start)
            print(f"  Run {run + 1} FFT:   {fft_times[-1]:.4f} s")

        naive_times = np.array(naive_times)
        fft_times = np.array(fft_times)

        naive_mean = naive_times.mean()
        naive_std = naive_times.std()
        fft_mean = fft_times.mean()
        fft_std = fft_times.std()

        naive_means.append(naive_mean)
        naive_stds.append(naive_std)
        fft_means.append(fft_mean)
        fft_stds.append(fft_std)

        print(f"Naive DFT2D  - mean {naive_mean:.4e} s, std {naive_std:.4e} s")
        print(f"FFT2D       - mean {fft_mean:.4e} s, std {fft_std:.4e} s")

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sizes,
        naive_means,
        yerr=2 * np.array(naive_stds),
        label="Naive DFT",
        marker="o",
        capsize=5,
    )
    plt.errorbar(
        sizes,
        fft_means,
        yerr=2 * np.array(fft_stds),
        label="FFT",
        marker="s",
        capsize=5,
    )

    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.xticks(sizes, [f"$2^{int(math.log2(n))}$" for n in sizes])

    plt.xlabel("Problem Size (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Complexity of Naive DFT vs FFT")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# The Command Line Interface Implementation

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
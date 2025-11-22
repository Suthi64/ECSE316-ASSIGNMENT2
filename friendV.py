import math
import cmath
import argparse
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import random


def dft(x: List[complex]) -> List[complex]:
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D list of complex numbers using the naive O(N^2) algorithm.
    """
    N = len(x)
    X = []
    for k in range(N):
        sum_val = 0.0 + 0.0j
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            sum_val += x[n] * cmath.exp(angle)
        X.append(sum_val)
    return X


def idft(x: List[complex]) -> List[complex]:
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of a 1D list of complex numbers using the naive O(N^2) algorithm.
    """
    N = len(x)
    x_inv = []
    for n in range(N):
        sum_val = 0.0 + 0.0j
        for k in range(N):
            angle = 2j * math.pi * k * n / N
            sum_val += x[k] * cmath.exp(angle)
        x_inv.append(sum_val / N)
    return x_inv


def fft(x: List[complex]) -> List[complex]:
    """
    Compute the Fast Fourier Transform (FFT) of a 1D list of complex numbers using the Cooley-Turkey algorithm.
    Assumes that the length of x is a power of 2.
    """
    N = len(x)
    if N <= 1:
        return x
    if N & (N - 1) != 0:
        raise ValueError("Size of x must be a power of 2")

    # FFT of even and odd terms
    even = fft(x[0::2])
    odd = fft(x[1::2])

    combined = [0] * N
    for k in range(N // 2):
        angle = -2j * math.pi * k / N
        twiddle = cmath.exp(angle) * odd[k]
        combined[k] = even[k] + twiddle
        combined[k + N // 2] = even[k] - twiddle
    return combined


def ifft(x: List[complex]) -> List[complex]:
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of a 1D list of complex numbers.
    Utilizes the FFT algorithm.
    """
    N = len(x)
    # Take the complex conjugate of the input
    x_conj = [num.conjugate() for num in x]
    # Perform FFT on the conjugated input
    y = fft(x_conj)
    # Take the complex conjugate of the result and scale by 1/N
    y_conj = [num.conjugate() / N for num in y]
    return y_conj


def transpose(matrix: List[List[complex]]) -> List[List[complex]]:
    """
    Transpose a 2D matrix represented as a list of lists.
    """
    return [list(row) for row in zip(*matrix)]


def dft2d(matrix: List[List[complex]]) -> List[List[complex]]:
    """
    Compute the 2D Discrete Fourier Transform (DFT) of a 2D matrix using the naive O(N^2) algorithm.
    """
    # Apply DFT on each row
    transformed_rows = [dft(row) for row in matrix]
    # Transpose the matrix to work on columns
    transposed = transpose(transformed_rows)
    # Apply DFT on each column
    transformed_cols = [dft(col) for col in transposed]
    # Transpose back to original layout
    return transpose(transformed_cols)


def idft2d(matrix: List[List[complex]]) -> List[List[complex]]:
    """
    Compute the Inverse 2D Discrete Fourier Transform (IDFT) of a 2D matrix using the naive O(N^2) algorithm.
    """
    # Apply IDFT on each row
    transformed_rows = [idft(row) for row in matrix]
    # Transpose the matrix to work on columns
    transposed = transpose(transformed_rows)
    # Apply IDFT on each column
    transformed_cols = [idft(col) for col in transposed]
    # Transpose back to original layout
    return transpose(transformed_cols)


def fft2d(matrix: List[List[complex]]) -> List[List[complex]]:
    """
    Compute the 2D Fast Fourier Transform (FFT) of a 2D matrix.
    Assumes that both dimensions are powers of 2.
    """
    # Apply FFT on each row
    transformed_rows = [fft(row) for row in matrix]
    # Transpose the matrix to work on columns
    transposed = transpose(transformed_rows)
    # Apply FFT on each column
    transformed_cols = [fft(col) for col in transposed]
    # Transpose back to original layout
    return transpose(transformed_cols)


def ifft2d(matrix: List[List[complex]]) -> List[List[complex]]:
    """
    Compute the Inverse 2D Fast Fourier Transform (IFFT) of a 2D matrix.
    """
    # Apply IFFT on each row
    transformed_rows = [ifft(row) for row in matrix]
    # Transpose the matrix to work on columns
    transposed = transpose(transformed_rows)
    # Apply IFFT on each column
    transformed_cols = [ifft(col) for col in transposed]
    # Transpose back to original layout
    return transpose(transformed_cols)


def next_power_of_two(n: int) -> int:
    """
    Compute the next power of 2 greater than or equal to n.
    """
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


def pad_image(image: Image.Image, new_width: int, new_height: int) -> Image.Image:
    """
    Pad the image with black pixels to reach the desired width and height.
    """
    padded_image = Image.new('L', (new_width, new_height))
    padded_image.paste(image, (0, 0))
    return padded_image


def resize_image(image: Image.Image, new_width: int, new_height: int) -> Image.Image:
    """
    Resize the image to the desired width and height using nearest neighbor interpolation.
    """
    return image.resize((new_width, new_height), Image.NEAREST)


def image_to_matrix(image_path: str, pad: bool = True, resize_flag: bool = False) -> (
List[List[complex]], int, int, int, int):
    """
    Convert an image to a 2D matrix of complex numbers (grayscale).
    Pads or resizes the image to the next power of 2 if required.

    Returns the matrix and original/new dimensions.
    """
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    original_width, original_height = img.size
    print(f"Original Image Size: {original_width}x{original_height}")

    # Compute next power of 2 for width and height
    new_width = next_power_of_two(original_width)
    new_height = next_power_of_two(original_height)

    if pad:
        if new_width != original_width or new_height != original_height:
            print(f"Padding image to {new_width}x{new_height}...")
            img = pad_image(img, new_width, new_height)
    elif resize_flag:
        if new_width != original_width or new_height != original_height:
            print(f"Resizing image to {new_width}x{new_height}...")
            img = resize_image(img, new_width, new_height)
    else:
        if new_width != original_width or new_height != original_height:
            print("Error: Image dimensions are not powers of 2. Use padding or resizing.")
            sys.exit(1)

    padded_width, padded_height = img.size
    print(f"Padded/Resized Image Size: {padded_width}x{padded_height}")

    pixels = list(img.getdata())
    matrix = []
    for y in range(padded_height):
        row = [complex(pixels[y * padded_width + x]) for x in range(padded_width)]
        matrix.append(row)
    return matrix, original_width, original_height, padded_width, padded_height


def matrix_to_image(matrix: List[List[complex]], title: str, original_width: int = None, original_height: int = None):
    """
    Convert a 2D matrix of complex numbers to an image and display it.
    For FFT magnitude, take the logarithm of the magnitude to enhance visibility.
    Optionally crop to original dimensions.
    """
    # Compute magnitude spectrum
    magnitude = []
    for row in matrix:
        mag_row = [math.log(1 + abs(pixel)) for pixel in row]
        magnitude.append(mag_row)

    # Shift the zero-frequency component to the center
    magnitude_shifted = np.fft.fftshift(np.array(magnitude))

    # Normalize to 0-255
    magnitude_normalized = 255 * (magnitude_shifted - magnitude_shifted.min()) / (
                magnitude_shifted.max() - magnitude_shifted.min())
    magnitude_image = magnitude_normalized.astype(np.uint8)

    # Optionally crop to original dimensions
    if original_width and original_height:
        magnitude_image = magnitude_image[:original_height, :original_width]

    plt.imshow(magnitude_image, cmap='gray')
    plt.title(title)
    plt.axis('off')


def denoise_fft(fft_result: List[List[complex]], cutoff_radius: int) -> List[List[complex]]:
    """
    Denoise the image by zeroing out high-frequency components beyond the cutoff_radius.
    This function treats the frequency domain as circular, keeping low frequencies near the center and near the edges.

    Parameters:
    - fft_result: 2D list of complex numbers representing the FFT of the image.
    - cutoff_radius: Radius beyond which frequencies are considered high and are zeroed out.

    Returns:
    - Denoised FFT result as a 2D list of complex numbers.
    """
    height = len(fft_result)
    width = len(fft_result[0]) if height > 0 else 0
    center_y, center_x = height // 2, width // 2

    denoised_fft = []
    for y in range(height):
        row = []
        for x in range(width):
            # Calculate distance from center
            distance_center = math.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            # Calculate distance from wrapped edges for circularity
            distance_wrapped = min(math.sqrt((y) ** 2 + (x) ** 2),
                                   math.sqrt((y - height) ** 2 + (x) ** 2),
                                   math.sqrt((y) ** 2 + (x - width) ** 2),
                                   math.sqrt((y - height) ** 2 + (x - width) ** 2))
            # Keep frequencies near center or near wrapped center
            if distance_center <= cutoff_radius or distance_wrapped <= cutoff_radius:
                row.append(fft_result[y][x])
            else:
                row.append(0.0 + 0.0j)
        denoised_fft.append(row)
    return denoised_fft


def compress_fft(fft_result: List[List[complex]], compression_levels: List[float]) -> List[List[List[complex]]]:
    """
    Compress the image by setting a certain percentage of Fourier coefficients to zero based on magnitude.

    Parameters:
    - fft_result: 2D list of complex numbers representing the FFT of the image.
    - compression_levels: List of compression percentages (e.g., [0, 50, 90, 95, 99, 99.9]).

    Returns:
    - List of compressed FFT results corresponding to each compression level.
    """
    # Flatten all magnitudes to compute thresholds
    magnitudes = [abs(coeff) for row in fft_result for coeff in row]
    total_coefficients = len(magnitudes)

    # Sort magnitudes in descending order
    sorted_magnitudes = sorted(magnitudes, reverse=True)

    compressed_ffts = []
    for level in compression_levels:
        if level == 0:
            # No compression
            compressed_fft = [row.copy() for row in fft_result]
        else:
            # Determine the number of coefficients to keep
            keep_fraction = 1 - (level / 100)
            keep_count = int(math.ceil(keep_fraction * total_coefficients))
            # Threshold magnitude
            if keep_count == 0:
                threshold = sorted_magnitudes[-1] + 1  # All set to zero
            else:
                threshold = sorted_magnitudes[keep_count - 1]
            # Compress FFT by setting coefficients below threshold to zero
            compressed_fft = []
            for row in fft_result:
                compressed_row = []
                for coeff in row:
                    if abs(coeff) >= threshold:
                        compressed_row.append(coeff)
                    else:
                        compressed_row.append(0.0 + 0.0j)
                compressed_fft.append(compressed_row)
        compressed_ffts.append(compressed_fft)
    return compressed_ffts


def generate_random_matrix(size: int) -> List[List[complex]]:
    """
    Generate a random 2D matrix of the given size with random complex numbers.
    """
    return [[complex(random.random(), random.random()) for _ in range(size)] for _ in range(size)]


def plot_runtime_complexity(sizes: List[int], naive_means: List[float], naive_stds: List[float],
                            fft_means: List[float], fft_stds: List[float]):
    """
    Plot the runtime complexity of naive DFT and FFT algorithms.

    Parameters:
    - sizes: List of problem sizes (N).
    - naive_means: List of mean runtimes for naive DFT.
    - naive_stds: List of standard deviations for naive DFT.
    - fft_means: List of mean runtimes for FFT.
    - fft_stds: List of standard deviations for FFT.
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, naive_means, yerr=2 * np.array(naive_stds), label='Naive DFT', marker='o', capsize=5)
    plt.errorbar(sizes, fft_means, yerr=2 * np.array(fft_stds), label='FFT', marker='s', capsize=5)
    plt.xscale('log', base=2)  # Corrected keyword argument
    plt.yscale('log')
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Complexity of Naive DFT vs FFT')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Perform FFT operations on an image.")
    parser.add_argument(
        '-m', '--mode',
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help='Mode of operation. [1] FFT Visualization. [2] Denoise Image. [3] Compress Image. [4] Runtime Complexity Analysis.'
    )
    parser.add_argument(
        '-i', '--image',
        type=str,
        default='moonlanding.png',
        help='Filename of the image for the DFT (default: moonlanding.png). Applicable for modes 1, 2, and 3.'
    )
    parser.add_argument(
        '--pad',
        action='store_true',
        help='Pad the image to the next power of 2 dimensions (default behavior). Applicable for modes 1, 2, and 3.'
    )
    parser.add_argument(
        '--resize',
        action='store_true',
        help='Resize the image to the next power of 2 dimensions instead of padding. Applicable for modes 1, 2, and 3.'
    )
    parser.add_argument(
        '--cutoff',
        type=int,
        default=None,
        help='Cutoff radius for high-frequency removal in denoising mode (default: 30%% of min(width, height)/2). Applicable only in mode 2.'
    )
    args = parser.parse_args()

    if args.resize and args.pad:
        print("Error: Choose either --pad or --resize, not both.")
        sys.exit(1)

    if args.mode == 1:
        # Mode 1: FFT Visualization
        print(f"Loading image '{args.image}'...")
        matrix, orig_w, orig_h, padded_w, padded_h = image_to_matrix(
            args.image,
            pad=not args.resize,
            resize_flag=args.resize
        )

        # Check if dimensions are powers of 2 (should be after padding/resizing)
        def is_power_of_two(n):
            return (n != 0) and (n & (n - 1) == 0)

        if not (is_power_of_two(padded_w) and is_power_of_two(padded_h)):
            print("Error: Image dimensions are not powers of 2 even after processing.")
            sys.exit(1)

        print("Performing 2D FFT...")
        fft_result = fft2d(matrix)
        print("FFT completed.")

        print("Displaying Original Image and FFT Magnitude Spectrum...")
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 2, 1)
        original_img = Image.open(args.image).convert('L')
        if not (args.pad or args.resize):
            display_img = original_img
        elif args.pad:
            display_img = pad_image(original_img, padded_w, padded_h)
        elif args.resize:
            display_img = resize_image(original_img, padded_w, padded_h)
        else:
            display_img = original_img  # Fallback
        original_np = np.array(display_img, dtype=np.uint8)
        plt.imshow(original_np, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # FFT Magnitude Spectrum
        plt.subplot(1, 2, 2)
        # Compute magnitude and shift
        magnitude = np.array([[math.log(1 + abs(pixel)) for pixel in row] for row in fft_result])
        magnitude_shifted = np.fft.fftshift(magnitude)
        # Normalize
        magnitude_normalized = 255 * (magnitude_shifted - magnitude_shifted.min()) / (
                    magnitude_shifted.max() - magnitude_shifted.min())
        magnitude_image = magnitude_normalized.astype(np.uint8)
        # Crop to original dimensions if padded
        if orig_w and orig_h:
            magnitude_image = magnitude_image[:orig_h, :orig_w]
        plt.imshow(magnitude_image, cmap='gray')
        plt.title('FFT Magnitude Spectrum')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    elif args.mode == 2:
        # Mode 2: Denoise Image
        print(f"Loading image '{args.image}'...")
        matrix, orig_w, orig_h, padded_w, padded_h = image_to_matrix(
            args.image,
            pad=not args.resize,
            resize_flag=args.resize
        )

        # Check if dimensions are powers of 2 (should be after padding/resizing)
        def is_power_of_two(n):
            return (n != 0) and (n & (n - 1) == 0)

        if not (is_power_of_two(padded_w) and is_power_of_two(padded_h)):
            print("Error: Image dimensions are not powers of 2 even after processing.")
            sys.exit(1)

        print("Performing 2D FFT...")
        fft_result = fft2d(matrix)
        print("FFT completed.")

        # Determine cutoff radius
        if args.cutoff is not None:
            cutoff_radius = args.cutoff
            print(f"Using user-specified cutoff radius: {cutoff_radius}")
        else:
            # Default cutoff: 30% of half the minimum dimension
            cutoff_radius = int(0.3 * min(padded_w, padded_h) / 2)
            print(f"No cutoff radius specified. Using default cutoff radius: {cutoff_radius}")

        print("Applying low-pass filter to denoise the image...")
        denoised_fft = denoise_fft(fft_result, cutoff_radius)
        print("Low-pass filter applied.")

        # Count non-zero coefficients
        total_coefficients = padded_w * padded_h
        non_zero_coefficients = sum(1 for row in denoised_fft for coeff in row if coeff != 0)
        fraction_non_zero = non_zero_coefficients / total_coefficients
        print(f"Number of non-zero Fourier coefficients used: {non_zero_coefficients}")
        print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}")

        print("Performing Inverse 2D FFT to reconstruct the denoised image...")
        ifft_result = ifft2d(denoised_fft)
        print("Inverse FFT completed.")

        # Extract real part and normalize
        denoised_real = np.array([[pixel.real for pixel in row] for row in ifft_result])
        denoised_real = np.clip(denoised_real, 0, 255)
        denoised_image = denoised_real.astype(np.uint8)

        print("Displaying Original and Denoised Images...")
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 2, 1)
        original_img = Image.open(args.image).convert('L')
        if not (args.pad or args.resize):
            display_img = original_img
        elif args.pad:
            display_img = pad_image(original_img, padded_w, padded_h)
        elif args.resize:
            display_img = resize_image(original_img, padded_w, padded_h)
        else:
            display_img = original_img  # Fallback
        original_np = np.array(display_img, dtype=np.uint8)
        plt.imshow(original_np, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Denoised Image
        plt.subplot(1, 2, 2)
        # If the image was padded or resized, crop back to original size for fair comparison
        denoised_cropped = denoised_image[:orig_h, :orig_w] if (orig_w and orig_h) else denoised_image
        plt.imshow(denoised_cropped, cmap='gray')
        plt.title('Denoised Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    elif args.mode == 3:
        # Mode 3: Compress Image
        print(f"Loading image '{args.image}'...")
        matrix, orig_w, orig_h, padded_w, padded_h = image_to_matrix(
            args.image,
            pad=not args.resize,
            resize_flag=args.resize
        )

        # Check if dimensions are powers of 2 (should be after padding/resizing)
        def is_power_of_two(n):
            return (n != 0) and (n & (n - 1) == 0)

        if not (is_power_of_two(padded_w) and is_power_of_two(padded_h)):
            print("Error: Image dimensions are not powers of 2 even after processing.")
            sys.exit(1)

        print("Performing 2D FFT...")
        fft_result = fft2d(matrix)
        print("FFT completed.")

        # Define compression levels
        compression_levels = [0, 50, 90, 95, 99, 99.9]

        print("Compressing the image by zeroing out Fourier coefficients at different levels...")
        compressed_ffts = compress_fft(fft_result, compression_levels)

        # Prepare to display images
        compressed_images = []
        non_zero_counts = []
        fractions_non_zero = []

        for idx, (level, compressed_fft) in enumerate(zip(compression_levels, compressed_ffts)):
            if level == 0:
                print(f"Compression Level: {level}% (Original Image)")
            else:
                print(f"Compression Level: {level}%")
            # Count non-zero coefficients
            non_zero_count = sum(1 for row in compressed_fft for coeff in row if coeff != 0)
            fraction_non_zero = non_zero_count / (padded_w * padded_h)
            print(f"Number of non-zero Fourier coefficients: {non_zero_count}")
            print(f"Fraction of non-zero coefficients: {fraction_non_zero:.4f}\n")
            non_zero_counts.append(non_zero_count)
            fractions_non_zero.append(fraction_non_zero)

            # Perform Inverse FFT
            ifft_result = ifft2d(compressed_fft)

            # Extract real part and normalize
            compressed_real = np.array([[pixel.real for pixel in row] for row in ifft_result])
            compressed_real = np.clip(compressed_real, 0, 255)
            compressed_image = compressed_real.astype(np.uint8)

            # Crop to original dimensions if padded
            compressed_cropped = compressed_image[:orig_h, :orig_w] if (orig_w and orig_h) else compressed_image
            compressed_images.append(compressed_cropped)

        # Display the compressed images in a 2x3 subplot
        print("Displaying Compressed Images at Different Levels...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Compressed Images at Different Levels", fontsize=20)

        for i, level in enumerate(compression_levels):
            row = i // 3
            col = i % 3
            axes[row, col].imshow(compressed_images[i], cmap='gray')
            axes[row, col].set_title(
                f"Compression: {level}%\nNon-zero Coefs: {non_zero_counts[i]}\nFraction: {fractions_non_zero[i]:.4f}")
            axes[row, col].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    elif args.mode == 4:
        # Mode 4: Runtime Complexity Analysis
        print("Starting Runtime Complexity Analysis for Naive DFT and FFT...")
        # Define problem sizes (powers of 2 from 32 to 256)
        sizes = [2 ** i for i in range(3, 9)]  # 32, 64, 128, 256
        naive_means = []
        naive_stds = []
        fft_means = []
        fft_stds = []
        num_runs = 10  # Number of runs per size

        for size in sizes:
            print(f"\nProblem Size: {size}x{size}")
            # Generate a random matrix of the given size
            matrix = generate_random_matrix(size)

            # Measure runtime for naive DFT2D
            print("  Measuring Naive DFT2D...")
            naive_times = []
            for run in range(num_runs):
                start_time = time.time()
                naive_result = dft2d(matrix)
                end_time = time.time()
                runtime = end_time - start_time
                naive_times.append(runtime)
                print(f"    Run {run + 1}: {runtime:.4f} seconds")
            naive_mean = np.mean(naive_times)
            naive_std = np.std(naive_times)
            naive_means.append(naive_mean)
            naive_stds.append(naive_std)
            print(f"  Naive DFT2D - Mean: {naive_mean:.4f} s, Std Dev: {naive_std:.4f} s")

            # Measure runtime for FFT2D
            print("  Measuring FFT2D...")
            fft_times = []
            for run in range(num_runs):
                start_time = time.time()
                fft_result = fft2d(matrix)
                end_time = time.time()
                runtime = end_time - start_time
                fft_times.append(runtime)
                print(f"    Run {run + 1}: {runtime:.4f} seconds")
            fft_mean = np.mean(fft_times)
            fft_std = np.std(fft_times)
            fft_means.append(fft_mean)
            fft_stds.append(fft_std)
            print(f"  FFT2D - Mean: {fft_mean:.4f} s, Std Dev: {fft_std:.4f} s")

        # Plot the runtime complexities
        print("\nPlotting Runtime Complexities...")
        plot_runtime_complexity(sizes, naive_means, naive_stds, fft_means, fft_stds)
        print("Runtime Complexity Analysis Completed.")

    else:
        print("Invalid mode selected. Choose either Mode 1, Mode 2, Mode 3, or Mode 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()

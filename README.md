# ECSE316-ASSIGNMENT2

We used Python 3.13.5

Before running the project, you must create and activate a Python virtual environment.

1. Create a virtual environment
python3 -m venv venv

2. Activate it
macOS / Linux:
source venv/bin/activate
Windows:
venv\Scripts\activate

3.Inside your virtual environment, install dependencies:
pip install numpy matplotlib pillow

The program is controlled using a mode system (-m 1/2/3/4).
You run it with:

python3 fft.py [-m mode] [-i image]
python3 fft.py -m [1/2/3/4] -i moonlanding.png (what we used for this lab)

-i moonlanding.png is not required

Mode 1: Display FFT Spectrum

Shows the original image and the FFT magnitude spectrum:

python3 fft.py -m 1

Other FFT view options:

to see the original FFT2D magnitude spectrum
python3 fft.py -m 1 --fft MS 

to see the NumPy FFT2 
python3 fft.py -m 1 --fft numpy 

python3 fft.py -m 1 --fft both
to see both the NumPy, original and base image

Mode 2: Denoising

To see all 6 panels
python3 fft.py -m 2

If you want different parameters:
python3 fft.py -m 2 --cutoff 0.08 

Mode 3: Compression

This command runs all compression levels (0%, 50%, 90%, 95%, 99%, 99.9%)
python3 fft.py -m 3

Mode 4: Runtime Benchmarking

Plots runtime for naive DFT vs FFT:
python3 fft.py -m 4

Custom problem size or trial count:
python3 fft.py -m 4 --maxpow 8 --trials 5
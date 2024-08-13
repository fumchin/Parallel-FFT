# Parallelization of 1D FFT & STFT for Audio Processing

## Overview
This project focuses on accelerating the Short-Time Fourier Transform (STFT) and Fast Fourier Transform (FFT) algorithms through parallel computing. By leveraging parallel processing, the project enhances the efficiency of audio signal processing, particularly in applying a low-pass filter to audio signals.

## Key Features
- **Parallel STFT**: The audio signal is divided into smaller frames, each processed in parallel to apply the STFT, significantly reducing processing time.
- **Parallel FFT**: Implemented using the Cooley-Tukey algorithm, the FFT computation is parallelized, addressing the challenge of locating threads correctly during the butterfly computation.
- **Hybrid Parallel Processing**: Combines both parallel STFT and FFT to optimize the overlap-adding process efficiently using MPI communication.

## Implementation Details
- **Platform**: Lenovo T14 with Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz, 8 logical processors.
- **Libraries**: Utilized `pthread.h`, `AudioFile.h` for audio processing, and `mpi.h` for parallel computation.
- **Test Data**: Processing of `Tim_Henson_VS_Ichika_Nito.wav` with varying FFT window sizes and thread counts.

## Results
The project demonstrates that parallel STFT offers more significant speedup compared to parallel FFT, with performance improvements as FFT window sizes increase.

## Conclusion
Parallelizing audio processing tasks such as STFT and FFT can significantly reduce computation time, making it feasible for real-time applications in audio processing.

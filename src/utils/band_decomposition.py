"""
Frequency band decomposition utilities for EEG signals.

This module provides functions for decomposing EEG signals into
six canonical frequency bands using FFT-based filtering.
"""

import numpy as np
from scipy.fft import fft, ifft


def decompose_bands(signal, fs=256, bands=None):
    """
    Decompose EEG signal into frequency bands using FFT.
    
    Args:
        signal: 1D EEG signal array
        fs: Sampling frequency (default: 256 Hz)
        bands: Dictionary of band names and frequency ranges.
               If None, uses default bands.
               
    Returns:
        Dictionary with band names as keys and band signals as values
    """
    if bands is None:
        bands = {
            'Delta_band': (0, 4),
            'Theta_band': (4, 8),
            'Alpha_band': (8, 13),
            'Beta_band': (13, 30),
            'Gamma_band': (30, 50),
            'High_Frequencies_band': (50, fs/2)
        }
    
    # Compute FFT
    signal_fft = fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Only use positive frequencies
    n = len(signal)
    positive_freqs = freqs[:n//2]
    signal_fft_positive = signal_fft[:n//2]
    
    # Create full spectrum for reconstruction
    signal_fft_full = np.zeros_like(signal_fft)
    signal_fft_full[:n//2] = signal_fft_positive
    signal_fft_full[n//2:] = np.conj(signal_fft_positive[::-1])
    
    decomposed = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        # Create frequency mask
        mask = np.zeros(len(signal_fft), dtype=bool)
        
        # Find frequency indices
        low_idx = np.argmin(np.abs(positive_freqs - low_freq))
        high_idx = np.argmin(np.abs(positive_freqs - high_freq))
        
        # Apply mask to positive frequencies
        mask[:high_idx] = True
        mask[:low_idx] = False
        
        # Mirror for negative frequencies
        mask[n//2:] = mask[n//2-1::-1]
        
        # Extract band
        band_fft = signal_fft_full * mask
        band_signal = np.real(ifft(band_fft))
        
        decomposed[band_name] = band_signal
    
    return decomposed


def reconstruct_signal(decomposed_bands):
    """
    Reconstruct full EEG signal from decomposed bands.
    
    Args:
        decomposed_bands: Dictionary with band names as keys and band signals as values
        
    Returns:
        Reconstructed signal (sum of all bands)
    """
    bands = list(decomposed_bands.values())
    return np.sum(bands, axis=0)


"""
Evaluation metrics for EEG denoising.
"""

import numpy as np
from scipy.fft import fft
from scipy.stats import pearsonr


def rrmse_t(pred, true):
    """
    Relative Root Mean Squared Error in time domain.
    
    Args:
        pred: Predicted signal
        true: Ground truth signal
        
    Returns:
        RRMSE_t value
    """
    return np.sqrt(np.mean((pred - true) ** 2)) / np.sqrt(np.mean(true ** 2))


def rrmse_s(pred, true):
    """
    Relative Root Mean Squared Error in frequency domain.
    
    Args:
        pred: Predicted signal
        true: Ground truth signal
        
    Returns:
        RRMSE_s value
    """
    pred_fft = np.abs(fft(pred))
    true_fft = np.abs(fft(true))
    return np.sqrt(np.mean((pred_fft - true_fft) ** 2)) / np.sqrt(np.mean(true_fft ** 2))


def correlation_coefficient(pred, true):
    """
    Pearson correlation coefficient.
    
    Args:
        pred: Predicted signal
        true: Ground truth signal
        
    Returns:
        Correlation coefficient value
    """
    return pearsonr(pred, true)[0]


def compute_all_metrics(pred, true):
    """
    Compute all evaluation metrics.
    
    Args:
        pred: Predicted signal
        true: Ground truth signal
        
    Returns:
        Dictionary with rrmse_t, rrmse_s, and cc
    """
    return {
        'rrmse_t': rrmse_t(pred, true),
        'rrmse_s': rrmse_s(pred, true),
        'cc': correlation_coefficient(pred, true)
    }


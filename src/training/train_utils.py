"""
Training utilities for EEG denoising models.
"""

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience=10, min_delta=1e-4):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def reset(self):
        """Reset the early stopping state."""
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False


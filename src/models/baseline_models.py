"""
Baseline model implementations for EEG denoising.

This module contains CNN, LSTM, and 1D-ResCNN baseline models
used for comparison in the paper.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    CNN baseline model for EEG denoising.
    
    Architecture: 4 convolutional layers with batch normalization and ReLU.
    """
    
    def __init__(self, input_len=512, num_filters=64):
        super(CNNBaseline, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )
        
        self.decoder = nn.Linear(num_filters * input_len, input_len)
        
    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        x = self.encoder(x)  # [batch_size, num_filters, seq_len]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.decoder(x)  # [batch_size, seq_len]
        return x


class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for EEG denoising.
    """
    
    def __init__(self, input_len=512, hidden_dim=128, num_layers=2):
        super(LSTMBaseline, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        self.decoder = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 1]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        output = self.decoder(lstm_out)  # [batch_size, seq_len, 1]
        return output.squeeze(-1)  # [batch_size, seq_len]


class ResCNN1D(nn.Module):
    """
    1D Residual CNN baseline model for EEG denoising.
    
    Uses multi-scale kernels (3, 5, 7) with residual connections.
    """
    
    def __init__(self, input_len=512, num_filters=64):
        super(ResCNN1D, self).__init__()
        
        # Multi-scale parallel branches
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )
        
        # Fusion and output
        self.fusion = nn.Conv1d(num_filters * 3, num_filters, kernel_size=1)
        self.output = nn.Linear(num_filters * input_len, input_len)
        
    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        branch3_out = self.branch3(x)
        branch5_out = self.branch5(x)
        branch7_out = self.branch7(x)
        
        # Concatenate multi-scale features
        fused = torch.cat([branch3_out, branch5_out, branch7_out], dim=1)
        fused = self.fusion(fused)  # [batch_size, num_filters, seq_len]
        
        # Flatten and decode
        fused = fused.view(fused.size(0), -1)
        output = self.output(fused)  # [batch_size, seq_len]
        return output


"""
State-of-the-art model implementations for EEG denoising.

This module contains EEGDNet and EEGDiR implementations
used for comparison in the paper.

Note: These are simplified implementations. For full details,
refer to the original papers and their official repositories.
"""

import torch
import torch.nn as nn
import math


class EEGDNet(nn.Module):
    """
    EEGDNet: 2D Transformer-based architecture for EEG denoising.
    
    Reference: Pu et al. "EEGDNet: Fusing Non-Local and Local Self-Similarities
    for EEG Signal Denoising with 2D-Transformer." (2022)
    
    Note: This is a simplified 1D adaptation. The original uses 2D attention.
    """
    
    def __init__(self, seq_len=512, model_dim=64, num_heads=4, num_layers=2):
        super(EEGDNet, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(1, model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(model_dim, 1)
        
    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 1]
        x = self.input_proj(x)  # [batch_size, seq_len, model_dim]
        x = self.transformer(x)  # [batch_size, seq_len, model_dim]
        x = self.output_proj(x)  # [batch_size, seq_len, 1]
        return x.squeeze(-1)  # [batch_size, seq_len]


class EEGDiR(nn.Module):
    """
    EEGDiR: Retentive Network for EEG denoising.
    
    Reference: Wang et al. "EEGDiR: A Retentive Network for EEG Signal Denoising." (2024)
    
    Note: This is a simplified implementation. The original uses a more complex
    retentive mechanism with memory-efficient attention.
    """
    
    def __init__(self, seq_len=512, model_dim=64, num_heads=4, num_layers=2):
        super(EEGDiR, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(1, model_dim)
        
        # Simplified retentive layers (using standard Transformer as approximation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=model_dim * 4
        )
        self.retentive_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output = nn.Linear(model_dim, 1)
        
    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 1]
        x = self.embedding(x)  # [batch_size, seq_len, model_dim]
        x = self.retentive_layers(x)  # [batch_size, seq_len, model_dim]
        x = self.output(x)  # [batch_size, seq_len, 1]
        return x.squeeze(-1)  # [batch_size, seq_len]


"""
Band2CleanFormer Model Implementation

A CNN-Transformer hybrid architecture for EEG denoising that processes
six frequency bands jointly with inter-band attention.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, channels, sequence_length]
        x = x + self.pe[:, :, :x.size(2)]
        return x


class Band2CleanFormer(nn.Module):
    """
    Band2CleanFormer: CNN-Transformer hybrid for multi-band EEG denoising.
    
    Args:
        seq_len: Sequence length (default: 512 for EOG, 1024 for EMG)
        input_channels: Number of input bands (default: 6)
        model_dim: Model dimension (default: 64)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
    """
    
    def __init__(self, seq_len=512, input_channels=6, model_dim=64, num_heads=4, num_layers=2):
        super(Band2CleanFormer, self).__init__()

        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, model_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(model_dim),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(model_dim),
        )

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model=model_dim, max_len=seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction Head
        self.decoder = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 6, seq_len]
            
        Returns:
            Denoised signal of shape [batch_size, seq_len]
        """
        # x shape: [batch_size, 6, seq_len]
        x = self.encoder(x)  # → [B, model_dim, seq_len]
        x = self.pos_encoding(x)  # Add positional encoding
        x = x.permute(0, 2, 1)  # [B, seq_len, model_dim] for Transformer
        x = self.transformer(x)  # → [B, seq_len, model_dim]
        x = x.permute(0, 2, 1)  # → [B, model_dim, seq_len]
        x = self.decoder(x)     # → [B, 1, seq_len]
        return x.squeeze(1)     # → [B, seq_len]


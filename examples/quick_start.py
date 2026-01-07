"""
Quick start example for Band2CleanFormer.

This script demonstrates how to:
1. Load a trained model
2. Preprocess EEG data into frequency bands
3. Denoise contaminated EEG signals
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.band2cleanformer import Band2CleanFormer


def load_model(model_path, seq_len=512, device='cuda'):
    """
    Load a trained Band2CleanFormer model.
    
    Args:
        model_path: Path to the saved model checkpoint
        seq_len: Sequence length (512 for EOG, 1024 for EMG)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = Band2CleanFormer(seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def denoise_eeg(model, contaminated_bands, device='cuda'):
    """
    Denoise EEG signal from frequency bands.
    
    Args:
        model: Trained Band2CleanFormer model
        contaminated_bands: Dictionary with band names as keys and 
                           contaminated signals as values
        device: Device to run inference on
        
    Returns:
        Denoised EEG signal
    """
    # Define band order
    bands = ["Delta_band", "Theta_band", "Alpha_band", 
             "Beta_band", "Gamma_band", "High_Frequencies_band"]
    
    # Stack bands into input tensor [6, seq_len]
    band_stack = []
    for band in bands:
        if band not in contaminated_bands:
            raise ValueError(f"Missing band: {band}")
        band_stack.append(contaminated_bands[band])
    
    input_tensor = torch.tensor(
        np.stack(band_stack, axis=0), 
        dtype=torch.float32
    ).unsqueeze(0).to(device)  # [1, 6, seq_len]
    
    # Denoise
    with torch.no_grad():
        denoised = model(input_tensor).cpu().numpy().flatten()
    
    return denoised


def main():
    """
    Example usage of Band2CleanFormer.
    """
    print("Band2CleanFormer Quick Start Example")
    print("=" * 50)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Load model (update path to your trained model)
    model_path = "checkpoints/band2cleanformer_eog.pth"
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please train a model first using train_eog.py or train_emg.py")
        return
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, seq_len=512, device=device)
    print("Model loaded successfully!")
    
    # Example: Prepare contaminated bands
    # In practice, you would load these from your preprocessed data
    print("\nExample: Denoising EEG signal...")
    print("Note: This is a placeholder. Replace with your actual data loading.")
    
    # Placeholder: Create dummy contaminated bands
    seq_len = 512
    contaminated_bands = {
        "Delta_band": np.random.randn(seq_len),
        "Theta_band": np.random.randn(seq_len),
        "Alpha_band": np.random.randn(seq_len),
        "Beta_band": np.random.randn(seq_len),
        "Gamma_band": np.random.randn(seq_len),
        "High_Frequencies_band": np.random.randn(seq_len),
    }
    
    # Denoise
    denoised = denoise_eeg(model, contaminated_bands, device=device)
    print(f"Denoised signal shape: {denoised.shape}")
    print("Denoising complete!")
    
    print("\n" + "=" * 50)
    print("For full training and evaluation, use:")
    print("  python src/training/train_eog.py --help")
    print("  python src/training/train_emg.py --help")


if __name__ == '__main__':
    main()


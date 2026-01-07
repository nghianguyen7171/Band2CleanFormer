"""
Training script for Band2CleanFormer on EMG-contaminated EEG data.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.band2cleanformer import Band2CleanFormer
from src.data.dataset import load_and_split_data, EEGDenoiseDataset
from src.training.train_utils import EarlyStopping
from src.evaluation.metrics import compute_all_metrics


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


def evaluate_model(model, test_data, bands, device):
    """Evaluate the model on test data."""
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        # Get test data for all bands
        test_data_dict = {}
        for band in bands:
            test_clean, test_contaminated, _ = test_data[band]
            test_data_dict[band] = (test_clean, test_contaminated)
        
        # Evaluate on all test samples
        for i in range(len(test_data_dict[bands[0]][0])):
            # Create 6-band input
            band_stack = []
            for band in bands:
                _, test_contaminated_b, _ = test_data[band]
                band_stack.append(test_contaminated_b[i])
            
            input_tensor = torch.tensor(
                np.stack(band_stack, axis=0), 
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Target is sum of clean bands
            target = np.stack([test_data[band][0][i] for band in bands], axis=0).sum(axis=0)
            denoised = model(input_tensor).cpu().numpy().flatten()
            
            # Compute metrics
            metrics = compute_all_metrics(denoised, target)
            metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'rrmse_t': np.mean([m['rrmse_t'] for m in metrics_list]),
        'rrmse_s': np.mean([m['rrmse_s'] for m in metrics_list]),
        'cc': np.mean([m['cc'] for m in metrics_list])
    }
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Band2CleanFormer for EMG denoising')
    parser.add_argument('--clean_dir', type=str, required=True,
                        help='Directory containing clean EEG band data')
    parser.add_argument('--contaminated_dir', type=str, required=True,
                        help='Directory containing contaminated EEG band data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=1024,
                        help='Sequence length (1024 for EMG)')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='checkpoints/band2cleanformer_emg.pth',
                        help='Path to save the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Define EEG frequency bands
    bands = ["Delta_band", "Theta_band", "Alpha_band", "Beta_band", "Gamma_band", "High_Frequencies_band"]
    
    # Load and split data for each band
    print("Loading and splitting data...")
    train_data = {}
    test_data = {}
    
    for band in bands:
        train_clean, train_contaminated, test_clean, test_contaminated, snr_labels_train, snr_labels_test = load_and_split_data(
            args.clean_dir, args.contaminated_dir, band
        )
        train_data[band] = (train_clean, train_contaminated)
        test_data[band] = (test_clean, test_contaminated, snr_labels_test)
        print(f"Loaded {band}: Train={len(train_clean)}, Test={len(test_clean)}")
    
    # Create dataset and dataloader
    train_dataset = EEGDenoiseDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = Band2CleanFormer(
        seq_len=args.seq_len,
        input_channels=6,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    
    for epoch in range(args.epochs):
        avg_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {avg_loss:.6f}")
        
        if early_stopping(avg_loss):
            print("Early stopping triggered.")
            break
    
    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_data, bands, device)
    print(f"Test Metrics - RRMSE_t: {metrics['rrmse_t']:.4f}, "
          f"RRMSE_s: {metrics['rrmse_s']:.4f}, CC: {metrics['cc']:.4f}")


if __name__ == '__main__':
    main()


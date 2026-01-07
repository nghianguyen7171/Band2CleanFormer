#!/usr/bin/env python3
"""
Main entry point for training Band2CleanFormer.

This script provides a unified interface for training on both EOG and EMG data.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.training.train_eog import main as train_eog
from src.training.train_emg import main as train_emg


def main():
    parser = argparse.ArgumentParser(
        description='Train Band2CleanFormer for EEG denoising',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on EOG data
  python scripts/run_training.py eog --clean_dir /path/to/clean --contaminated_dir /path/to/contaminated
  
  # Train on EMG data
  python scripts/run_training.py emg --clean_dir /path/to/clean --contaminated_dir /path/to/contaminated
        """
    )
    
    subparsers = parser.add_subparsers(dest='artifact_type', help='Artifact type to train on')
    
    # EOG parser
    eog_parser = subparsers.add_parser('eog', help='Train on EOG-contaminated EEG')
    eog_parser.add_argument('--clean_dir', type=str, required=True,
                           help='Directory containing clean EEG band data')
    eog_parser.add_argument('--contaminated_dir', type=str, required=True,
                           help='Directory containing contaminated EEG band data')
    eog_parser.add_argument('--batch_size', type=int, default=128)
    eog_parser.add_argument('--epochs', type=int, default=200)
    eog_parser.add_argument('--lr', type=float, default=0.001)
    eog_parser.add_argument('--save_path', type=str, default='checkpoints/band2cleanformer_eog.pth')
    
    # EMG parser
    emg_parser = subparsers.add_parser('emg', help='Train on EMG-contaminated EEG')
    emg_parser.add_argument('--clean_dir', type=str, required=True,
                           help='Directory containing clean EEG band data')
    emg_parser.add_argument('--contaminated_dir', type=str, required=True,
                           help='Directory containing contaminated EEG band data')
    emg_parser.add_argument('--batch_size', type=int, default=128)
    emg_parser.add_argument('--epochs', type=int, default=100)
    emg_parser.add_argument('--lr', type=float, default=0.001)
    emg_parser.add_argument('--save_path', type=str, default='checkpoints/band2cleanformer_emg.pth')
    
    args = parser.parse_args()
    
    if args.artifact_type == 'eog':
        # Convert namespace to list for train_eog
        sys.argv = ['train_eog.py'] + [
            '--clean_dir', args.clean_dir,
            '--contaminated_dir', args.contaminated_dir,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--save_path', args.save_path
        ]
        train_eog()
    elif args.artifact_type == 'emg':
        # Convert namespace to list for train_emg
        sys.argv = ['train_emg.py'] + [
            '--clean_dir', args.clean_dir,
            '--contaminated_dir', args.contaminated_dir,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--save_path', args.save_path
        ]
        train_emg()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


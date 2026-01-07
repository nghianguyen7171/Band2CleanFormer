"""
Dataset classes for EEG denoising.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_and_split_data(clean_dir, contaminated_dir, band_name, test_size=0.2, random_state=42):
    """
    Loads EEG data for a specific frequency band and splits into train and test sets.
    
    Args:
        clean_dir: Directory containing clean EEG band data
        contaminated_dir: Directory containing contaminated EEG band data (with SNR subfolders)
        band_name: Name of the frequency band (e.g., "Delta_band")
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_clean, train_contaminated, test_clean, test_contaminated, snr_labels_train, snr_labels_test
    """
    # Load clean EEG data
    clean_band_path = os.path.join(clean_dir, f"{band_name}.npy")
    if not os.path.exists(clean_band_path):
        raise FileNotFoundError(f"Clean band file not found: {clean_band_path}")
    clean_band = np.load(clean_band_path)

    # Load contaminated EEG data across all SNR levels
    contaminated_band = []
    snr_labels = []

    for snr_folder in sorted(os.listdir(contaminated_dir)):
        contaminated_band_path = os.path.join(contaminated_dir, snr_folder, f"{band_name}.npy")
        if os.path.exists(contaminated_band_path):
            contaminated_data = np.load(contaminated_band_path)
            contaminated_band.append(contaminated_data)
            snr_labels.extend([snr_folder] * len(contaminated_data))

    if len(contaminated_band) == 0:
        raise ValueError(f"No contaminated data found in {contaminated_dir}")

    # Convert lists to numpy arrays
    contaminated_band = np.concatenate(contaminated_band, axis=0)
    # Convert string labels ("SNR_-7") to integers (-7)
    snr_labels = np.array([int(snr.replace("SNR_", "")) for snr in snr_labels])

    # Ensure clean_band is correctly repeated to match contaminated EEG samples
    clean_band_repeated = np.tile(
        clean_band, 
        (len(contaminated_band) // len(clean_band) + 1, 1)
    )[:len(contaminated_band)]

    # Stratified Train-Test Split
    train_clean, test_clean, train_contaminated, test_contaminated, snr_labels_train, snr_labels_test = train_test_split(
        clean_band_repeated, 
        contaminated_band, 
        snr_labels, 
        test_size=test_size, 
        stratify=snr_labels, 
        random_state=random_state
    )

    return train_clean, train_contaminated, test_clean, test_contaminated, snr_labels_train, snr_labels_test


class EEGDenoiseDataset(Dataset):
    """
    Dataset class for multi-band EEG denoising.
    
    Expects a dictionary of band data: {band_name: (clean_data, contaminated_data)}
    """
    
    def __init__(self, train_data_dict):
        """
        Args:
            train_data_dict: Dictionary with band names as keys and tuples of 
                           (clean_data, contaminated_data) as values
        """
        self.train_data_dict = train_data_dict
        self.band_names = list(train_data_dict.keys())
        self.length = len(train_data_dict[self.band_names[0]][0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns a 6-band input tensor and corresponding clean target.
        
        Returns:
            input_tensor: [6, seq_len] - Stacked contaminated bands
            target: [seq_len] - Sum of clean bands
        """
        band_stack = []
        target_sum = None

        for band_name in self.band_names:
            clean_band, contaminated_band = self.train_data_dict[band_name]
            band_stack.append(contaminated_band[idx])
            
            if target_sum is None:
                target_sum = clean_band[idx].copy()
            else:
                target_sum += clean_band[idx]

        input_tensor = np.stack(band_stack, axis=0)  # [6, seq_len]
        
        return torch.FloatTensor(input_tensor), torch.FloatTensor(target_sum)


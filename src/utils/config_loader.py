"""
Configuration loader for Band2CleanFormer.

This module provides utilities for loading and validating configuration files.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config):
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['data', 'model', 'training', 'paths']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate data paths
    for artifact_type in ['eog', 'emg']:
        if artifact_type not in config['data']:
            raise ValueError(f"Missing data configuration for {artifact_type}")
        
        for path_type in ['clean_dir', 'contaminated_dir']:
            if path_type not in config['data'][artifact_type]:
                raise ValueError(f"Missing {path_type} in {artifact_type} data configuration")
    
    # Validate model parameters
    model_keys = ['seq_len_eog', 'seq_len_emg', 'input_channels', 'model_dim', 'num_heads', 'num_layers']
    for key in model_keys:
        if key not in config['model']:
            raise ValueError(f"Missing model parameter: {key}")
    
    return True


def get_config(config_path='config/config.yaml'):
    """
    Load and validate configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
    """
    config = load_config(config_path)
    validate_config(config)
    return config


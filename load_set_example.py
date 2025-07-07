#!/usr/bin/env python3
"""
Simple example script demonstrating .set file loading and validation.
"""

from pathlib import Path
from brainheart.io.loaders import EEGLoader
from brainheart.io.validators import DataValidator
from matplotlib import pyplot as plt
import numpy as np
from brainheart.config.config_manager import ConfigManager
local_config = ConfigManager()

def load_set_file(file_path: str):
    """
    Load a .set file and validate the data.
    
    Parameters
    ----------
    file_path : str
        Path to the .set file
    
    Returns
    -------
    PhysiologicalData or None
        Loaded data if successful, None otherwise
    """
    try:
        # Initialize loader
        loader = EEGLoader(verbose=False)

        # Check if file can be loaded
        if not loader.can_load(file_path):
            print(f"Cannot load file: {file_path}")
            return None

        # Load the data
        print(f"Loading {file_path}...")
        data = loader.load(file_path)

        # Print basic info
        print(f"✓ Successfully loaded {file_path}")
        print(f"  - Data type: {data.data_type}")
        print(f"  - Shape: {data.data.shape}")
        print(f"  - Channels: {data.spatial_info.n_channels}")
        print(f"  - Samples: {data.temporal_info.n_samples}")
        print(f"  - Duration: {data.temporal_info.duration:.2f} seconds")
        print(f"  - Sampling rate: {data.temporal_info.sampling_rate} Hz")

        return data

    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def validate_data(data):
    """
    Validate the loaded data.
    
    Parameters
    ----------
    data : PhysiologicalData
        Data to validate
    
    Returns
    -------
    bool
        True if validation passes
    """
    if data is None:
        return False

    print("\nValidating data...")

    # Initialize validator
    validator = DataValidator(strict=False)

    # Get validation errors
    errors = validator.get_validation_errors(data)

    if not errors:
        print("✓ Data validation passed")
        return True
    else:
        print(f"✗ Data validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False


def plot_eeg(eeg_data, offset=1, downsample=100):
    fig, ax = plt.subplots(figsize=(12, 10))

    data_ds = eeg_data[:, ::downsample]
    for i in range(data_ds.shape[0]):
        ax.plot(data_ds[i] * 100 + i * offset, 'k-', linewidth=0.5)

    ax.set_ylabel('Channel (offset)')
    ax.set_xlabel('Time Points')
    ax.set_title('EEG')
    ax.set_yticks(np.arange(0, eeg_data.shape[0]) * offset)
    ax.set_yticklabels([f'Ch{i + 1}' for i in range(35)])
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate .set file loading."""

    # Example usage
    print("BrainHeart .set File Loading Example")
    print("=" * 40)

    # You can specify your .set file path here
    set_file_path = input("Enter path to .set file (or press Enter for demo): ").strip()

    if not set_file_path:
        print("No file path provided. Creating demo scenario...")
        print("Note: This will use placeholder data since no real .set file was provided.")

        set_file_path = local_config.get_path("data_path", "./data") / "eeglab_eeg_ecg" / "sub-001_ses-01_task-GXtESCTT_eeg.set"

    # Load the file
    data = load_set_file(set_file_path)

    # Validate the data
    if data:
        is_valid = validate_data(data)

        if is_valid:
            print("\n✓ File successfully loaded and validated!")
            print("\nData summary:")
            print(f"  Subject ID: {data.metadata.subject_id}")
            print(f"  Channel names: {data.spatial_info.channel_names[:5]}...")
            print(f"  Channel types: {set(data.spatial_info.channel_types)}")

            # Show first few samples
            print(f"\nFirst 5 samples of first channel:")
            print(f"  {data.data[0, :5]}")

        else:
            print("\n⚠  File loaded but validation failed")
    else:
        print("\n✗ Failed to load file")


if __name__ == "__main__":
    main()

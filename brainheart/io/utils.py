"""
Utility functions for BrainHeart I/O operations.
"""

import warnings
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np

from .base import DataFormat, PhysiologicalData, ValidationError
from .loaders import EEGLoader, ECGLoader, fMRILoader, WidefieldLoader
from .validators import DataValidator


class BrainHeartIOError(Exception):
    """Base exception for BrainHeart I/O operations."""
    pass


class UnsupportedFormatError(BrainHeartIOError):
    """Exception raised when file format is not supported."""
    pass


class DataIntegrityError(BrainHeartIOError):
    """Exception raised when data integrity checks fail."""
    pass


class TemporalAlignmentError(BrainHeartIOError):
    """Exception raised when temporal alignment fails."""
    pass


def detect_format(file_path: Union[str, Path]) -> Optional[DataFormat]:
    """
    Automatically detect file format from path and content.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
        
    Returns
    -------
    DataFormat or None
        Detected format, or None if format cannot be determined
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # First try extension-based detection
    format_from_ext = _detect_format_from_extension(path)
    if format_from_ext:
        return format_from_ext
    
    # Try content-based detection
    format_from_content = _detect_format_from_content(path)
    if format_from_content:
        return format_from_content
    
    return None


def _detect_format_from_extension(path: Path) -> Optional[DataFormat]:
    """Detect format from file extension."""
    suffix = path.suffix.lower().lstrip('.')
    
    extension_map = {
        # EEG formats
        'edf': DataFormat.EDF,
        'bdf': DataFormat.BDF,
        'fif': DataFormat.FIFF,
        'vhdr': DataFormat.BRAINVISION,
        'set': DataFormat.EEGLAB,
        
        # ECG formats
        'xml': DataFormat.PHILIPS_XML,
        
        # fMRI formats
        'nii': DataFormat.NIFTI,
        'img': DataFormat.ANALYZE,
        'mnc': DataFormat.MINC,
        
        # Imaging formats
        'tiff': DataFormat.TIFF,
        'tif': DataFormat.TIFF,
        'h5': DataFormat.HDF5,
        'hdf5': DataFormat.HDF5,
        'mat': DataFormat.MAT,
        'npz': DataFormat.NPZ
    }
    
    # Special case for WFDB format
    if suffix == 'dat' and path.with_suffix('.hea').exists():
        return DataFormat.WFDB
    elif suffix == 'hea':
        return DataFormat.WFDB
    
    return extension_map.get(suffix)


def _detect_format_from_content(path: Path) -> Optional[DataFormat]:
    """Detect format from file content (magic numbers, headers)."""
    try:
        with open(path, 'rb') as f:
            header = f.read(1024)  # Read first 1KB
        
        # Check for specific file signatures
        if header.startswith(b'\x89HDF'):
            return DataFormat.HDF5
        elif b'MATLAB' in header:
            return DataFormat.MAT
        elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
            return DataFormat.TIFF
        elif b'FIFF' in header[:16]:
            return DataFormat.FIFF
        
        # For text-based formats, check content
        try:
            text_header = header.decode('utf-8', errors='ignore')
            if '<restingecg' in text_header.lower():
                return DataFormat.PHILIPS_XML
        except:
            pass
            
    except Exception:
        pass
    
    return None


def list_supported_formats() -> Dict[str, List[str]]:
    """
    List all supported file formats by data type.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping data types to supported formats
    """
    return {
        'eeg': ['EDF', 'BDF', 'FIFF', 'BrainVision', 'EEGLAB'],
        'ecg': ['WFDB', 'MIT', 'Philips XML'],
        'fmri': ['NIfTI', 'ANALYZE', 'MINC'],
        'widefield': ['TIFF', 'HDF5', 'MAT', 'NPZ']
    }


def get_appropriate_loader(file_path: Union[str, Path]) -> Optional[object]:
    """
    Get the appropriate loader for a file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
        
    Returns
    -------
    BaseLoader or None
        Appropriate loader instance, or None if no suitable loader found
    """
    loaders = [EEGLoader(), ECGLoader(), fMRILoader(), WidefieldLoader()]
    
    for loader in loaders:
        if loader.can_load(file_path):
            return loader
    
    return None


def load_data(file_path: Union[str, Path], **kwargs) -> PhysiologicalData:
    """
    Universal data loading function.
    
    Parameters
    ----------
    file_path : str or Path
        Path to data file
    validate : bool, optional
        Whether to validate loaded data (default: True)
    loader_type : str, optional
        Force specific loader type ('eeg', 'ecg', 'fmri', 'widefield')
    **kwargs
        Additional parameters passed to the loader
    
    Returns
    -------
    PhysiologicalData
        Loaded physiological data
        
    Raises
    ------
    UnsupportedFormatError
        If file format is not supported
    DataIntegrityError
        If data validation fails
    """
    validate = kwargs.pop('validate', True)
    loader_type = kwargs.pop('loader_type', None)
    
    path = Path(file_path)
    
    # Get appropriate loader
    if loader_type:
        loader_map = {
            'eeg': EEGLoader(),
            'ecg': ECGLoader(),
            'fmri': fMRILoader(),
            'widefield': WidefieldLoader()
        }
        loader = loader_map.get(loader_type)
        if not loader:
            raise ValueError(f"Unknown loader type: {loader_type}")
    else:
        loader = get_appropriate_loader(path)
    
    if not loader:
        detected_format = detect_format(path)
        raise UnsupportedFormatError(
            f"No suitable loader found for file: {path}\n"
            f"Detected format: {detected_format}"
        )
    
    # Load data
    try:
        data = loader.load(path, **kwargs)
    except Exception as e:
        raise BrainHeartIOError(f"Failed to load data from {path}: {str(e)}")
    
    # Validate if requested
    if validate:
        validator = DataValidator()
        if not validator.validate(data):
            errors = validator.get_validation_errors(data)
            raise DataIntegrityError(f"Data validation failed:\n" + "\n".join(errors))
    
    return data


def save_data(data: PhysiologicalData, file_path: Union[str, Path], 
              format_type: Optional[str] = None, **kwargs):
    """
    Save physiological data to file.
    
    Parameters
    ----------
    data : PhysiologicalData
        Data to save
    file_path : str or Path
        Output file path
    format_type : str, optional
        Output format ('numpy', 'hdf5', 'mat', 'csv')
    **kwargs
        Additional parameters for saving
    """
    path = Path(file_path)
    
    if format_type is None:
        # Infer format from extension
        format_type = _infer_save_format(path)
    
    if format_type == 'numpy':
        _save_numpy(data, path, **kwargs)
    elif format_type == 'hdf5':
        _save_hdf5(data, path, **kwargs)
    elif format_type == 'mat':
        _save_mat(data, path, **kwargs)
    elif format_type == 'csv':
        _save_csv(data, path, **kwargs)
    else:
        raise UnsupportedFormatError(f"Unsupported save format: {format_type}")


def _infer_save_format(path: Path) -> str:
    """Infer save format from file extension."""
    suffix = path.suffix.lower().lstrip('.')
    
    format_map = {
        'npz': 'numpy',
        'h5': 'hdf5',
        'hdf5': 'hdf5',
        'mat': 'mat',
        'csv': 'csv'
    }
    
    return format_map.get(suffix, 'numpy')


def _save_numpy(data: PhysiologicalData, path: Path, **kwargs):
    """Save data in NumPy format."""
    save_dict = {
        'data': data.data,
        'sampling_rate': data.temporal_info.sampling_rate,
        'n_samples': data.temporal_info.n_samples,
        'n_channels': data.spatial_info.n_channels,
        'channel_names': np.array(data.spatial_info.channel_names),
        'channel_types': np.array(data.spatial_info.channel_types),
        'data_type': data.data_type,
        'subject_id': data.metadata.subject_id or 'unknown'
    }
    
    np.savez_compressed(path, **save_dict)


def _save_hdf5(data: PhysiologicalData, path: Path, **kwargs):
    """Save data in HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 support")
    
    with h5py.File(path, 'w') as f:
        # Save data arrays
        f.create_dataset('data', data=data.data, compression='gzip')
        
        # Save metadata
        f.attrs['sampling_rate'] = data.temporal_info.sampling_rate
        f.attrs['n_samples'] = data.temporal_info.n_samples
        f.attrs['n_channels'] = data.spatial_info.n_channels
        f.attrs['data_type'] = data.data_type
        f.attrs['subject_id'] = data.metadata.subject_id or 'unknown'
        
        # Save channel info
        f.create_dataset('channel_names', data=[s.encode() for s in data.spatial_info.channel_names])
        f.create_dataset('channel_types', data=[s.encode() for s in data.spatial_info.channel_types])


def _save_mat(data: PhysiologicalData, path: Path, **kwargs):
    """Save data in MATLAB format."""
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError("scipy is required for MATLAB format support")
    
    save_dict = {
        'data': data.data,
        'sampling_rate': data.temporal_info.sampling_rate,
        'n_samples': data.temporal_info.n_samples,
        'n_channels': data.spatial_info.n_channels,
        'channel_names': data.spatial_info.channel_names,
        'channel_types': data.spatial_info.channel_types,
        'data_type': data.data_type,
        'subject_id': data.metadata.subject_id or 'unknown'
    }
    
    savemat(path, save_dict)


def _save_csv(data: PhysiologicalData, path: Path, **kwargs):
    """Save data in CSV format."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for CSV support")
    
    # Create time vector
    time_vec = np.arange(data.temporal_info.n_samples) / data.temporal_info.sampling_rate
    
    # Create DataFrame
    if data.data.ndim == 1:
        df_data = {'time': time_vec, 'signal': data.data}
    else:
        df_data = {'time': time_vec}
        for i, ch_name in enumerate(data.spatial_info.channel_names):
            df_data[ch_name] = data.data[i]
    
    df = pd.DataFrame(df_data)
    df.to_csv(path, index=False)


def align_temporal_data(data_list: List[PhysiologicalData], 
                       method: str = 'interpolation') -> List[PhysiologicalData]:
    """
    Align multiple physiological datasets temporally.
    
    Parameters
    ----------
    data_list : List[PhysiologicalData]
        List of physiological datasets to align
    method : str, optional
        Alignment method ('interpolation', 'resampling', 'cropping')
        
    Returns
    -------
    List[PhysiologicalData]
        Temporally aligned datasets
    """
    if len(data_list) < 2:
        return data_list
    
    if method == 'interpolation':
        return _align_by_interpolation(data_list)
    elif method == 'resampling':
        return _align_by_resampling(data_list)
    elif method == 'cropping':
        return _align_by_cropping(data_list)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_by_interpolation(data_list: List[PhysiologicalData]) -> List[PhysiologicalData]:
    """Align by interpolating to common time grid."""
    # Find common sampling rate (highest)
    max_sr = max(data.temporal_info.sampling_rate for data in data_list)
    
    # Find common duration (shortest)
    min_duration = min(data.temporal_info.duration for data in data_list)
    
    # Create common time vector
    n_samples_common = int(max_sr * min_duration)
    common_time = np.linspace(0, min_duration, n_samples_common)
    
    aligned_data = []
    for data in data_list:
        # Original time vector
        orig_time = np.arange(data.temporal_info.n_samples) / data.temporal_info.sampling_rate
        
        # Interpolate to common time grid
        if data.data.ndim == 1:
            interp_data = np.interp(common_time, orig_time, data.data)
            interp_data = interp_data.reshape(1, -1)
        else:
            interp_data = np.zeros((data.data.shape[0], len(common_time)))
            for ch in range(data.data.shape[0]):
                interp_data[ch] = np.interp(common_time, orig_time, data.data[ch])
        
        # Create new temporal info
        new_temporal_info = TemporalInfo(
            sampling_rate=max_sr,
            n_samples=n_samples_common,
            duration=min_duration
        )
        
        # Create aligned data object
        aligned = PhysiologicalData(
            data=interp_data,
            temporal_info=new_temporal_info,
            spatial_info=data.spatial_info,
            metadata=data.metadata,
            data_type=data.data_type,
            format_info=data.format_info
        )
        aligned_data.append(aligned)
    
    return aligned_data


def _align_by_resampling(data_list: List[PhysiologicalData]) -> List[PhysiologicalData]:
    """Align by resampling to common sampling rate."""
    # Find target sampling rate (lowest to avoid aliasing)
    target_sr = min(data.temporal_info.sampling_rate for data in data_list)
    
    aligned_data = []
    for data in data_list:
        if data.temporal_info.sampling_rate == target_sr:
            aligned_data.append(data)
        else:
            # Resample data
            from scipy.signal import resample
            
            target_n_samples = int(data.temporal_info.n_samples * target_sr / data.temporal_info.sampling_rate)
            
            if data.data.ndim == 1:
                resampled_data = resample(data.data, target_n_samples)
                resampled_data = resampled_data.reshape(1, -1)
            else:
                resampled_data = np.zeros((data.data.shape[0], target_n_samples))
                for ch in range(data.data.shape[0]):
                    resampled_data[ch] = resample(data.data[ch], target_n_samples)
            
            # Create new temporal info
            new_temporal_info = TemporalInfo(
                sampling_rate=target_sr,
                n_samples=target_n_samples,
                duration=target_n_samples / target_sr
            )
            
            # Create resampled data object
            resampled = PhysiologicalData(
                data=resampled_data,
                temporal_info=new_temporal_info,
                spatial_info=data.spatial_info,
                metadata=data.metadata,
                data_type=data.data_type,
                format_info=data.format_info
            )
            aligned_data.append(resampled)
    
    return aligned_data


def _align_by_cropping(data_list: List[PhysiologicalData]) -> List[PhysiologicalData]:
    """Align by cropping to common duration."""
    # Find shortest duration
    min_duration = min(data.temporal_info.duration for data in data_list)
    
    aligned_data = []
    for data in data_list:
        if data.temporal_info.duration <= min_duration:
            aligned_data.append(data)
        else:
            # Crop data
            max_samples = int(min_duration * data.temporal_info.sampling_rate)
            cropped_data = data.data[..., :max_samples]
            
            # Create new temporal info
            new_temporal_info = TemporalInfo(
                sampling_rate=data.temporal_info.sampling_rate,
                n_samples=max_samples,
                duration=min_duration
            )
            
            # Create cropped data object
            cropped = PhysiologicalData(
                data=cropped_data,
                temporal_info=new_temporal_info,
                spatial_info=data.spatial_info,
                metadata=data.metadata,
                data_type=data.data_type,
                format_info=data.format_info
            )
            aligned_data.append(cropped)
    
    return aligned_data


# Import necessary base classes
from .base import TemporalInfo
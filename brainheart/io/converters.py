"""
Data format conversion utilities for BrainHeart toolkit.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
from pathlib import Path

from .base import BaseConverter, PhysiologicalData, TemporalInfo, SpatialInfo, MetaData


class StandardFormat:
    """
    Defines the BrainHeart standard format for physiological data.
    
    This format ensures compatibility across different data types and enables
    unified analysis workflows for brain-heart interactions.
    """
    
    REQUIRED_FIELDS = [
        'data', 'temporal_info', 'spatial_info', 'metadata', 'data_type'
    ]
    
    SUPPORTED_DATA_TYPES = ['eeg', 'ecg', 'fmri', 'widefield']
    
    @staticmethod
    def validate_standard_format(data: PhysiologicalData) -> bool:
        """Validate that data conforms to BrainHeart standard format."""
        try:
            # Check required fields exist
            for field in StandardFormat.REQUIRED_FIELDS:
                if not hasattr(data, field):
                    return False
            
            # Check data type is supported
            if data.data_type not in StandardFormat.SUPPORTED_DATA_TYPES:
                return False
            
            # Check data array properties
            if not isinstance(data.data, np.ndarray):
                return False
            
            # Check temporal info consistency
            if data.temporal_info.n_samples != data.data.shape[-1]:
                return False
            
            # Check spatial info consistency
            # TODO: need to decide if the number of channels is the total number of channels or the shape
            # currently, use total number of channels
            expected_channels = data.data.size/data.data.shape[-1] if data.data.ndim >= 2 else 1
            if data.spatial_info.n_channels != expected_channels:
                return False
            
            return True
            
        except Exception:
            return False


class DataConverter(BaseConverter):
    """
    Universal data converter for physiological signals.
    
    Converts between different data formats while preserving essential
    information for brain-heart interaction analysis.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.conversion_registry = self._build_conversion_registry()
    
    def convert(self, data: PhysiologicalData, target_format: str) -> PhysiologicalData:
        """
        Convert physiological data to target format.
        
        Parameters
        ----------
        data : PhysiologicalData
            Input data to convert
        target_format : str
            Target format ('standard', 'mne_raw', 'numpy', 'pandas')
        
        Returns
        -------
        PhysiologicalData
            Converted data
        """
        if target_format == 'standard':
            return self._to_standard_format(data)
        elif target_format == 'mne_raw':
            return self._to_mne_raw(data)
        elif target_format == 'numpy':
            return self._to_numpy_format(data)
        elif target_format == 'pandas':
            return self._to_pandas_format(data)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def supported_conversions(self) -> Dict[str, List[str]]:
        """Return supported conversion mappings."""
        return {
            'eeg': ['standard', 'mne_raw', 'numpy', 'pandas'],
            'ecg': ['standard', 'numpy', 'pandas'],
            'fmri': ['standard', 'numpy', 'nibabel'],
            'widefield': ['standard', 'numpy', 'tiff_stack']
        }
    
    def _to_standard_format(self, data: PhysiologicalData) -> PhysiologicalData:
        """Convert to BrainHeart standard format."""
        if StandardFormat.validate_standard_format(data):
            return data  # Already in standard format
        
        # Perform standardization
        standardized_data = self._standardize_data_array(data.data, data.data_type)
        
        # Ensure consistent metadata
        standardized_metadata = self._standardize_metadata(data.metadata)
        
        return PhysiologicalData(
            data=standardized_data,
            temporal_info=data.temporal_info,
            spatial_info=data.spatial_info,
            metadata=standardized_metadata,
            data_type=data.data_type,
            format_info={'converted_to': 'standard', 'original_format': data.format_info.get('original_format', 'unknown')}
        )
    
    def _to_mne_raw(self, data: PhysiologicalData) -> PhysiologicalData:
        """Convert to MNE-Python Raw format."""
        if data.data_type not in ['eeg', 'ecg']:
            raise ValueError(f"MNE Raw format not supported for {data.data_type}")
        
        # Create MNE info structure
        # In real implementation:
        # import mne
        # info = mne.create_info(
        #     ch_names=data.spatial_info.channel_names,
        #     sfreq=data.temporal_info.sampling_rate,
        #     ch_types=data.spatial_info.channel_types
        # )
        # raw = mne.io.RawArray(data.data, info)
        
        # Placeholder - return modified PhysiologicalData with MNE format info
        mne_format_info = {
            'format': 'mne_raw',
            'ch_names': data.spatial_info.channel_names,
            'sfreq': data.temporal_info.sampling_rate,
            'ch_types': data.spatial_info.channel_types
        }
        
        return PhysiologicalData(
            data=data.data,
            temporal_info=data.temporal_info,
            spatial_info=data.spatial_info,
            metadata=data.metadata,
            data_type=data.data_type,
            format_info=mne_format_info
        )
    
    def _to_numpy_format(self, data: PhysiologicalData) -> PhysiologicalData:
        """Convert to pure NumPy format."""
        numpy_format_info = {
            'format': 'numpy',
            'shape': data.data.shape,
            'dtype': str(data.data.dtype),
            'sampling_rate': data.temporal_info.sampling_rate
        }
        
        return PhysiologicalData(
            data=data.data,
            temporal_info=data.temporal_info,
            spatial_info=data.spatial_info,
            metadata=data.metadata,
            data_type=data.data_type,
            format_info=numpy_format_info
        )
    
    def _to_pandas_format(self, data: PhysiologicalData) -> PhysiologicalData:
        """Convert to pandas DataFrame format."""
        # For time series data, create DataFrame with time index
        # In real implementation would use pandas.DataFrame
        
        pandas_format_info = {
            'format': 'pandas',
            'columns': data.spatial_info.channel_names,
            'index_type': 'time',
            'sampling_rate': data.temporal_info.sampling_rate
        }
        
        return PhysiologicalData(
            data=data.data,
            temporal_info=data.temporal_info,
            spatial_info=data.spatial_info,
            metadata=data.metadata,
            data_type=data.data_type,
            format_info=pandas_format_info
        )
    
    def _standardize_data_array(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Standardize data array dimensions and properties."""
        if data_type == 'eeg':
            # EEG: ensure (channels, time) format
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim > 2:
                # Flatten spatial dimensions for multichannel data
                data = data.reshape(-1, data.shape[-1])
        
        elif data_type == 'ecg':
            # ECG: ensure (leads, time) format
            if data.ndim == 1:
                data = data.reshape(1, -1)
        
        elif data_type == 'fmri':
            # fMRI: maintain 4D structure (x, y, z, time) or flatten to (voxels, time)
            if data.ndim == 4:
                # Can keep 4D or flatten to 2D
                pass
            elif data.ndim == 2:
                # Already in (voxels, time) format
                pass
            else:
                raise ValueError(f"Unexpected fMRI data dimensions: {data.shape}")
        
        elif data_type == 'widefield':
            # Widefield: maintain 3D structure (height, width, time) or flatten
            if data.ndim == 3:
                # Can keep 3D or flatten to 2D
                pass
            elif data.ndim == 2:
                # Already in (pixels, time) format
                pass
            else:
                raise ValueError(f"Unexpected widefield data dimensions: {data.shape}")
        
        return data
    
    def _standardize_metadata(self, metadata: MetaData) -> MetaData:
        """Standardize metadata fields."""
        # Ensure required fields are present
        if metadata.subject_id is None:
            metadata.subject_id = "unknown"
        
        if metadata.session_id is None:
            metadata.session_id = "01"
        
        if metadata.acquisition_date is None:
            from datetime import datetime
            metadata.acquisition_date = datetime.now().isoformat()
        
        return metadata
    
    def _build_conversion_registry(self) -> Dict[str, Dict[str, callable]]:
        """Build registry of conversion functions."""
        return {
            'eeg': {
                'mne_raw': self._to_mne_raw,
                'numpy': self._to_numpy_format,
                'pandas': self._to_pandas_format
            },
            'ecg': {
                'numpy': self._to_numpy_format,
                'pandas': self._to_pandas_format
            },
            'fmri': {
                'numpy': self._to_numpy_format,
                'nibabel': self._to_nibabel_format
            },
            'widefield': {
                'numpy': self._to_numpy_format,
                'tiff_stack': self._to_tiff_stack
            }
        }
    
    def _to_nibabel_format(self, data: PhysiologicalData) -> PhysiologicalData:
        """Convert fMRI data to NiBabel format."""
        # Implementation would create nibabel Nifti1Image
        nibabel_format_info = {
            'format': 'nibabel',
            'shape': data.data.shape,
            'tr': 1.0 / data.temporal_info.sampling_rate
        }
        
        return PhysiologicalData(
            data=data.data,
            temporal_info=data.temporal_info,
            spatial_info=data.spatial_info,
            metadata=data.metadata,
            data_type=data.data_type,
            format_info=nibabel_format_info
        )
    
    def _to_tiff_stack(self, data: PhysiologicalData) -> PhysiologicalData:
        """Convert widefield data to TIFF stack format."""
        # Implementation would save as TIFF stack
        tiff_format_info = {
            'format': 'tiff_stack',
            'shape': data.data.shape,
            'frame_rate': data.temporal_info.sampling_rate
        }
        
        return PhysiologicalData(
            data=data.data,
            temporal_info=data.temporal_info,
            spatial_info=data.spatial_info,
            metadata=data.metadata,
            data_type=data.data_type,
            format_info=tiff_format_info
        )


class MNEIntegrator:
    """
    Integration utilities for MNE-Python ecosystem.
    
    Provides seamless conversion between BrainHeart format and MNE objects.
    """
    
    @staticmethod
    def to_mne_raw(data: PhysiologicalData):
        """Convert PhysiologicalData to MNE Raw object."""
        # Implementation would use MNE
        # import mne
        # info = mne.create_info(...)
        # raw = mne.io.RawArray(data.data, info)
        # return raw
        raise NotImplementedError("MNE integration not implemented yet")
    
    @staticmethod
    def from_mne_raw(raw) -> PhysiologicalData:
        """Convert MNE Raw object to PhysiologicalData."""
        # Implementation would extract from MNE Raw
        raise NotImplementedError("MNE integration not implemented yet")
    
    @staticmethod
    def to_mne_epochs(data: PhysiologicalData, events: np.ndarray):
        """Convert PhysiologicalData to MNE Epochs object."""
        raise NotImplementedError("MNE Epochs integration not implemented yet")
    
    @staticmethod
    def from_mne_epochs(epochs) -> PhysiologicalData:
        """Convert MNE Epochs object to PhysiologicalData."""
        raise NotImplementedError("MNE Epochs integration not implemented yet")
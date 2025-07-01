"""
Data validation and format testing utilities for BrainHeart toolkit.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from .base import BaseValidator, PhysiologicalData, ValidationError


class DataValidator(BaseValidator):
    """
    Comprehensive validator for physiological data.
    
    Ensures data integrity, format compliance, and readiness for
    brain-heart interaction analysis.
    """
    
    def __init__(self, strict: bool = False):
        super().__init__(strict)
        self.validation_rules = self._build_validation_rules()
    
    def validate(self, data: PhysiologicalData) -> bool:
        """
        Validate physiological data.
        
        Parameters
        ----------
        data : PhysiologicalData
            Data to validate
        
        Returns
        -------
        bool
            True if validation passes
        """
        errors = self.get_validation_errors(data)
        return len(errors) == 0
    
    def get_validation_errors(self, data: PhysiologicalData) -> List[str]:
        """
        Get list of validation errors.
        
        Parameters
        ----------
        data : PhysiologicalData
            Data to validate
            
        Returns
        -------
        List[str]
            List of validation error messages
        """
        errors = []
        
        # Basic structure validation
        errors.extend(self._validate_structure(data))
        
        # Data array validation
        errors.extend(self._validate_data_array(data))
        
        # Temporal information validation
        errors.extend(self._validate_temporal_info(data))
        
        # Spatial information validation
        errors.extend(self._validate_spatial_info(data))
        
        # Metadata validation
        errors.extend(self._validate_metadata(data))
        
        # Data type specific validation
        errors.extend(self._validate_data_type_specific(data))
        
        return errors
    
    def _validate_structure(self, data: PhysiologicalData) -> List[str]:
        """Validate basic data structure."""
        errors = []
        
        required_attrs = ['data', 'temporal_info', 'spatial_info', 'metadata', 'data_type']
        for attr in required_attrs:
            if not hasattr(data, attr):
                errors.append(f"Missing required attribute: {attr}")
        
        return errors
    
    def _validate_data_array(self, data: PhysiologicalData) -> List[str]:
        """Validate data array properties."""
        errors = []
        
        if not isinstance(data.data, np.ndarray):
            errors.append("Data must be a numpy array")
            return errors
        
        # Check for invalid values
        if np.any(np.isnan(data.data)):
            if self.strict:
                errors.append("Data contains NaN values")
            else:
                errors.append("Warning: Data contains NaN values")
        
        if np.any(np.isinf(data.data)):
            errors.append("Data contains infinite values")
        
        # Check data range for different types
        if data.data_type == 'eeg':
            # EEG typically in microvolts, reasonable range
            if np.max(np.abs(data.data)) > 1000:
                errors.append("Warning: EEG data values seem unusually large (>1000 µV)")
        
        elif data.data_type == 'ecg':
            # ECG typically in millivolts
            if np.max(np.abs(data.data)) > 10:
                errors.append("Warning: ECG data values seem unusually large (>10 mV)")
        
        # Check dimensionality
        if data.data.ndim == 0:
            errors.append("Data cannot be 0-dimensional")
        elif data.data.ndim > 4:
            errors.append(f"Data has too many dimensions: {data.data.ndim}")
        
        return errors
    
    def _validate_temporal_info(self, data: PhysiologicalData) -> List[str]:
        """Validate temporal information."""
        errors = []
        
        # Check sampling rate
        if data.temporal_info.sampling_rate <= 0:
            errors.append("Sampling rate must be positive")
        
        # Check consistency with data shape
        expected_samples = data.data.shape[-1]
        if data.temporal_info.n_samples != expected_samples:
            errors.append(f"Temporal info n_samples ({data.temporal_info.n_samples}) "
                         f"doesn't match data shape ({expected_samples})")
        
        # Check duration consistency
        expected_duration = data.temporal_info.n_samples / data.temporal_info.sampling_rate
        if abs(data.temporal_info.duration - expected_duration) > 0.001:
            errors.append(f"Duration inconsistent with n_samples and sampling_rate")
        
        # Check reasonable sampling rates for different data types
        sr = data.temporal_info.sampling_rate
        if data.data_type == 'eeg':
            if sr < 100 or sr > 10000:
                errors.append(f"Warning: Unusual EEG sampling rate: {sr} Hz")
        elif data.data_type == 'ecg':
            if sr < 100 or sr > 2000:
                errors.append(f"Warning: Unusual ECG sampling rate: {sr} Hz")
        elif data.data_type == 'fmri':
            if sr < 0.1 or sr > 10:
                errors.append(f"Warning: Unusual fMRI sampling rate: {sr} Hz")
        elif data.data_type == 'widefield':
            if sr < 1 or sr > 1000:
                errors.append(f"Warning: Unusual widefield sampling rate: {sr} Hz")
        
        return errors
    
    def _validate_spatial_info(self, data: PhysiologicalData) -> List[str]:
        """Validate spatial information."""
        errors = []
        
        # Check channel count consistency
        if data.data.ndim >= 2:
            expected_channels = data.data.shape[0]
            if data.spatial_info.n_channels != expected_channels:
                errors.append(f"Spatial info n_channels ({data.spatial_info.n_channels}) "
                             f"doesn't match data shape ({expected_channels})")
        
        # Check channel names
        if len(data.spatial_info.channel_names) != data.spatial_info.n_channels:
            errors.append("Number of channel names doesn't match n_channels")
        
        # Check channel types
        if len(data.spatial_info.channel_types) != data.spatial_info.n_channels:
            errors.append("Number of channel types doesn't match n_channels")
        
        # Validate channel types for data type
        valid_ch_types = self._get_valid_channel_types(data.data_type)
        for ch_type in data.spatial_info.channel_types:
            if ch_type not in valid_ch_types:
                errors.append(f"Invalid channel type '{ch_type}' for {data.data_type} data")
        
        return errors
    
    def _validate_metadata(self, data: PhysiologicalData) -> List[str]:
        """Validate metadata."""
        errors = []
        
        # Check subject ID format
        if data.metadata.subject_id and not isinstance(data.metadata.subject_id, str):
            errors.append("Subject ID must be a string")
        
        # Check bad channels are valid
        if data.metadata.bad_channels:
            valid_channels = set(data.spatial_info.channel_names)
            for bad_ch in data.metadata.bad_channels:
                if bad_ch not in valid_channels:
                    errors.append(f"Bad channel '{bad_ch}' not in channel list")
        
        return errors
    
    def _validate_data_type_specific(self, data: PhysiologicalData) -> List[str]:
        """Validate data type specific requirements."""
        errors = []
        
        if data.data_type == 'eeg':
            errors.extend(self._validate_eeg_specific(data))
        elif data.data_type == 'ecg':
            errors.extend(self._validate_ecg_specific(data))
        elif data.data_type == 'fmri':
            errors.extend(self._validate_fmri_specific(data))
        elif data.data_type == 'widefield':
            errors.extend(self._validate_widefield_specific(data))
        else:
            errors.append(f"Unknown data type: {data.data_type}")
        
        return errors
    
    def _validate_eeg_specific(self, data: PhysiologicalData) -> List[str]:
        """EEG-specific validation."""
        errors = []
        
        # Check for reasonable number of channels
        if data.spatial_info.n_channels > 1024:
            errors.append("Warning: Very high number of EEG channels")
        
        # Check for standard EEG channel types
        eeg_ch_types = set(data.spatial_info.channel_types)
        expected_types = {'eeg', 'eog', 'emg', 'misc', 'stim'}
        unexpected_types = eeg_ch_types - expected_types
        if unexpected_types:
            errors.append(f"Unexpected EEG channel types: {unexpected_types}")
        
        return errors
    
    def _validate_ecg_specific(self, data: PhysiologicalData) -> List[str]:
        """ECG-specific validation."""
        errors = []
        
        # Check for reasonable number of leads
        if data.spatial_info.n_channels > 12:
            errors.append("Warning: Unusually high number of ECG leads")
        
        return errors
    
    def _validate_fmri_specific(self, data: PhysiologicalData) -> List[str]:
        """fMRI-specific validation."""
        errors = []
        
        # Check for 4D data structure
        if data.data.ndim not in [2, 4]:
            errors.append("fMRI data should be 2D (voxels, time) or 4D (x, y, z, time)")
        
        return errors
    
    def _validate_widefield_specific(self, data: PhysiologicalData) -> List[str]:
        """Widefield imaging specific validation."""
        errors = []
        
        # Check for appropriate data structure
        if data.data.ndim not in [2, 3]:
            errors.append("Widefield data should be 2D (pixels, time) or 3D (height, width, time)")
        
        return errors
    
    def _get_valid_channel_types(self, data_type: str) -> List[str]:
        """Get valid channel types for data type."""
        type_map = {
            'eeg': ['eeg', 'eog', 'emg', 'misc', 'stim', 'ecg'],
            'ecg': ['ecg', 'misc'],
            'fmri': ['fmri', 'misc'],
            'widefield': ['widefield', 'calcium', 'voltage', 'hemodynamic', 'misc']
        }
        return type_map.get(data_type, ['misc'])
    
    def _build_validation_rules(self) -> Dict[str, Any]:
        """Build validation rules registry."""
        return {
            'sampling_rate_ranges': {
                'eeg': (100, 10000),
                'ecg': (100, 2000),
                'fmri': (0.1, 10),
                'widefield': (1, 1000)
            },
            'channel_count_limits': {
                'eeg': (1, 1024),
                'ecg': (1, 12),
                'fmri': (1, 1000000),
                'widefield': (1, 1000000)
            }
        }


class FormatTester:
    """
    Testing framework for data format compatibility and conversion accuracy.
    """
    
    def __init__(self):
        self.test_results = {}
    
    def run_format_tests(self, data: PhysiologicalData) -> Dict[str, bool]:
        """
        Run comprehensive format tests.
        
        Parameters
        ----------
        data : PhysiologicalData
            Data to test
            
        Returns
        -------
        Dict[str, bool]
            Test results
        """
        results = {}
        
        # Test data loading
        results['data_loading'] = self._test_data_loading(data)
        
        # Test format conversion
        results['format_conversion'] = self._test_format_conversion(data)
        
        # Test data integrity
        results['data_integrity'] = self._test_data_integrity(data)
        
        # Test temporal alignment
        results['temporal_alignment'] = self._test_temporal_alignment(data)
        
        # Test metadata preservation
        results['metadata_preservation'] = self._test_metadata_preservation(data)
        
        self.test_results[data.metadata.subject_id or 'unknown'] = results
        return results
    
    def _test_data_loading(self, data: PhysiologicalData) -> bool:
        """Test data loading functionality."""
        try:
            # Check if data was loaded successfully
            if data.data is None or data.data.size == 0:
                return False
            
            # Check basic properties
            if data.temporal_info.sampling_rate <= 0:
                return False
            
            if data.spatial_info.n_channels <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def _test_format_conversion(self, data: PhysiologicalData) -> bool:
        """Test format conversion accuracy."""
        try:
            from .converters import DataConverter
            
            converter = DataConverter()
            
            # Test round-trip conversion
            converted = converter.convert(data, 'standard')
            reconverted = converter.convert(converted, 'numpy')
            
            # Check data preservation
            if not np.allclose(data.data, reconverted.data, rtol=1e-10):
                return False
            
            return True
        except Exception:
            return False
    
    def _test_data_integrity(self, data: PhysiologicalData) -> bool:
        """Test data integrity."""
        try:
            validator = DataValidator()
            return validator.validate(data)
        except Exception:
            return False
    
    def _test_temporal_alignment(self, data: PhysiologicalData) -> bool:
        """Test temporal alignment accuracy."""
        try:
            # Check time vector consistency
            expected_duration = data.temporal_info.n_samples / data.temporal_info.sampling_rate
            if abs(data.temporal_info.duration - expected_duration) > 0.001:
                return False
            
            return True
        except Exception:
            return False
    
    def _test_metadata_preservation(self, data: PhysiologicalData) -> bool:
        """Test metadata preservation."""
        try:
            # Check essential metadata exists
            if not data.metadata.subject_id:
                return False
            
            if not data.data_type:
                return False
            
            return True
        except Exception:
            return False
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        report = "BrainHeart Format Testing Report\n"
        report += "=" * 40 + "\n\n"
        
        for subject_id, results in self.test_results.items():
            report += f"Subject: {subject_id}\n"
            report += "-" * 20 + "\n"
            
            for test_name, passed in results.items():
                status = "PASS" if passed else "FAIL"
                report += f"{test_name}: {status}\n"
            
            report += "\n"
        
        return report
    
    def run_benchmark_tests(self, test_data_paths: List[Path]) -> Dict[str, Any]:
        """Run benchmark tests on multiple datasets."""
        benchmark_results = {
            'loading_times': [],
            'conversion_times': [],
            'validation_times': [],
            'memory_usage': [],
            'success_rates': {}
        }
        
        # Implementation would load each test file and measure performance
        # This is a placeholder for the benchmark framework
        
        return benchmark_results
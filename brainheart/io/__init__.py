"""
Data I/O module for BrainHeart toolkit.

This module provides unified data loading and format conversion capabilities
for multimodal physiological data including EEG, ECG, fMRI, and widefield imaging.
"""

from .base import BaseLoader, DataFormat, ValidationError
from .loaders import EEGLoader, ECGLoader, fMRILoader, WidefieldLoader
from .converters import DataConverter, StandardFormat
from .validators import DataValidator, FormatTester
from .utils import detect_format, list_supported_formats

__all__ = [
    "BaseLoader", "DataFormat", "ValidationError",
    "EEGLoader", "ECGLoader", "fMRILoader", "WidefieldLoader", 
    "DataConverter", "StandardFormat",
    "DataValidator", "FormatTester",
    "detect_format", "list_supported_formats"
]
"""
Base classes and data structures for BrainHeart I/O system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field


class DataFormat(Enum):
    """Supported data formats for physiological signals."""
    # EEG formats
    EDF = "edf"
    BDF = "bdf"
    FIFF = "fif"
    BRAINVISION = "vhdr"
    EEGLAB = "set"

    # ECoG formats
    #TODO: add common ECoG formats

    # ECG formats
    WFDB = "wfdb"
    MIT = "mit"
    PHILIPS_XML = "xml"
    
    # fMRI formats
    NIFTI = "nii"
    ANALYZE = "img"
    MINC = "mnc"
    
    # Widefield imaging
    TIFF = "tiff"
    HDF5 = "h5"
    MAT = "mat"
    NPZ = "npz"
    BIN = "bin"


@dataclass
class TemporalInfo:
    """Temporal information for physiological signals."""
    sampling_rate: float
    n_samples: int
    duration: float
    start_time: Optional[float] = None
    time_unit: str = "seconds"
    
    def __post_init__(self):
        if self.duration == 0:
            self.duration = self.n_samples / self.sampling_rate


@dataclass
class SpatialInfo:
    """Spatial information for physiological signals."""
    n_channels: int
    channel_names: List[str] = field(default_factory=list)
    channel_types: List[str] = field(default_factory=list)
    channel_locations: Optional[Dict[str, Any]] = None
    coordinate_system: Optional[str] = None


@dataclass
class MetaData:
    """Metadata container for physiological recordings."""
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    task: Optional[str] = None
    acquisition_date: Optional[str] = None
    equipment: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, Any]]] = None
    bad_channels: List[str] = field(default_factory=list)
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class PhysiologicalData:
    """Standardized container for physiological data."""
    data: np.ndarray
    temporal_info: TemporalInfo
    spatial_info: SpatialInfo
    metadata: MetaData
    data_type: str  # 'eeg', 'ecg', 'fmri', 'widefield'
    format_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate data dimensions
        if self.data.ndim not in [1, 2, 3, 4]:
            raise ValueError("Data must be 1D, 2D, 3D, or 4D array")
        
        # Ensure consistent channel information
        if len(self.spatial_info.channel_names) == 0:
            self.spatial_info.channel_names = [
                f"Ch{i+1}" for i in range(self.spatial_info.n_channels)
            ]
        
        if len(self.spatial_info.channel_types) == 0:
            self.spatial_info.channel_types = [
                "unknown" for _ in range(self.spatial_info.n_channels)
            ]


class ValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class BaseLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.supported_formats = []
    
    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> PhysiologicalData:
        """Load data from file."""
        pass
    
    @abstractmethod
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """Check if this loader can handle the given file."""
        pass
    
    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate file exists and is readable."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
    
    def _log(self, message: str, level: str = "info"):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{level.upper()}] {message}")
    
    def _detect_format(self, file_path: Union[str, Path]) -> Optional[DataFormat]:
        """Detect file format from extension."""
        path = Path(file_path)
        suffix = path.suffix.lower().lstrip('.')
        
        # Map extensions to formats
        format_map = {
            'edf': DataFormat.EDF,
            'bdf': DataFormat.BDF,
            'fif': DataFormat.FIFF,
            'vhdr': DataFormat.BRAINVISION,
            'set': DataFormat.EEGLAB,
            'xml': DataFormat.PHILIPS_XML,
            'nii': DataFormat.NIFTI,
            'img': DataFormat.ANALYZE,
            'mnc': DataFormat.MINC,
            'tiff': DataFormat.TIFF,
            'tif': DataFormat.TIFF,
            'h5': DataFormat.HDF5,
            'hdf5': DataFormat.HDF5,
            'mat': DataFormat.MAT,
            'npz': DataFormat.NPZ
        }
        
        return format_map.get(suffix)


class BaseConverter(ABC):
    """Abstract base class for format converters."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    @abstractmethod
    def convert(self, data: PhysiologicalData, target_format: str) -> PhysiologicalData:
        """Convert data to target format."""
        pass
    
    @abstractmethod
    def supported_conversions(self) -> Dict[str, List[str]]:
        """Return supported conversion mappings."""
        pass


class BaseValidator(ABC):
    """Abstract base class for data validators."""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    @abstractmethod
    def validate(self, data: PhysiologicalData) -> bool:
        """Validate physiological data."""
        pass
    
    @abstractmethod
    def get_validation_errors(self, data: PhysiologicalData) -> List[str]:
        """Get list of validation errors."""
        pass
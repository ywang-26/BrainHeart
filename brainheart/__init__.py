"""
BrainHeart: Integrated Brain-Heart Interaction Analysis Toolkit

A comprehensive open-source Python toolkit for analyzing multimodal temporal 
dynamics of the brain-heart axis.
"""

__version__ = "0.1.0"
__author__ = "Yunmiao Wang, Alex Zhao"

from . import io
from . import data
from . import utils

__all__ = ["io", "data", "utils"]
"""Core module for DataSentry library.

This module contains base classes and utilities used throughout the library.
"""

from datasentry.core.base import (
    BaseDetector,
    BaseFixer,
    BaseVisualizer,
    DetectionResult,
    FixResult,
    SeverityLevel,
)
from datasentry.core.report import ReportGenerator
from datasentry.core.utils import validate_data, get_feature_names

__all__ = [
    "BaseDetector",
    "BaseFixer",
    "BaseVisualizer",
    "DetectionResult",
    "FixResult",
    "SeverityLevel",
    "ReportGenerator",
    "validate_data",
    "get_feature_names",
]

"""DataSentry - A Professional Data Quality Framework for ML Pipelines.

DataSentry provides comprehensive detection and remediation of data quality issues
including imbalance, label noise, data leakage, missing values, outliers, 
redundancy, and data shift.

Example:
    >>> from datasentry import DataSentry
    >>> 
    >>> # Initialize
    >>> ds = DataSentry()
    >>> 
    >>> # Run full analysis
    >>> report = ds.generate_full_report(X_train, y_train, X_test=X_test)
    >>> 
    >>> # Fix all issues
    >>> X_clean, y_clean = ds.fix_all(X_train, y_train)

For more information, visit: https://github.com/yourusername/datasentry
"""

__version__ = "1.0.0"
__author__ = "DataSentry Team"
__email__ = "contact@datasentry.dev"
__license__ = "MIT"

# Core classes
from datasentry.core.base import (
    BaseDetector,
    BaseFixer,
    BaseVisualizer,
    DetectionResult,
    FixResult,
    SeverityLevel,
)
from datasentry.core.report import ReportGenerator
from datasentry.core.utils import validate_data

# Detectors
from datasentry.detectors import (
    ImbalanceDetector,
    LabelNoiseDetector,
    DataLeakageDetector,
    MissingValueDetector,
    OutlierDetector,
    RedundancyDetector,
    ShiftDetector,
)

# Fixers
from datasentry.fixers import (
    ImbalanceFixer,
    LabelNoiseFixer,
    DataLeakageFixer,
    MissingValueFixer,
    OutlierFixer,
    RedundancyFixer,
    ShiftFixer,
)

# Visualizers
from datasentry.visualizers import (
    ImbalanceVisualizer,
    NoiseVisualizer,
    LeakageVisualizer,
    MissingVisualizer,
    OutlierVisualizer,
    RedundancyVisualizer,
    ShiftVisualizer,
)

# Main orchestrator
from datasentry.main import DataSentry

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Main class
    "DataSentry",
    # Core classes
    "BaseDetector",
    "BaseFixer",
    "BaseVisualizer",
    "DetectionResult",
    "FixResult",
    "SeverityLevel",
    "ReportGenerator",
    "validate_data",
    # Detectors
    "ImbalanceDetector",
    "LabelNoiseDetector",
    "DataLeakageDetector",
    "MissingValueDetector",
    "OutlierDetector",
    "RedundancyDetector",
    "ShiftDetector",
    # Fixers
    "ImbalanceFixer",
    "LabelNoiseFixer",
    "DataLeakageFixer",
    "MissingValueFixer",
    "OutlierFixer",
    "RedundancyFixer",
    "ShiftFixer",
    # Visualizers
    "ImbalanceVisualizer",
    "NoiseVisualizer",
    "LeakageVisualizer",
    "MissingVisualizer",
    "OutlierVisualizer",
    "RedundancyVisualizer",
    "ShiftVisualizer",
]

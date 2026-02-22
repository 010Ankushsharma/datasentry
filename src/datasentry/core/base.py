"""Base classes for DataSentry detectors, fixers, and visualizers.

This module defines the abstract base classes that all components must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SeverityLevel(Enum):
    """Severity levels for data quality issues.
    
    Attributes:
        NONE: No issue detected.
        LOW: Minor issue, may not affect model performance significantly.
        MEDIUM: Moderate issue, likely to affect model performance.
        HIGH: Serious issue, will significantly impact model performance.
        CRITICAL: Critical issue, model training may fail or produce invalid results.
    """
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectionResult:
    """Container for detection results.
    
    This class standardizes the output format for all detectors in the library.
    
    Attributes:
        detector_name: Name of the detector that produced this result.
        issue_detected: Whether any issue was detected.
        severity: Severity level of the detected issue.
        score: Numerical score representing the magnitude of the issue (0.0 to 1.0).
        details: Detailed information about the detected issue.
        recommendations: List of recommended actions to fix the issue.
        metadata: Additional metadata about the detection.
    
    Example:
        >>> result = DetectionResult(
        ...     detector_name="ImbalanceDetector",
        ...     issue_detected=True,
        ...     severity=SeverityLevel.HIGH,
        ...     score=0.85,
        ...     details={"imbalance_ratio": 10.5, "minority_class": 0},
        ...     recommendations=["Use SMOTE oversampling", "Apply class weights"],
        ... )
    """
    detector_name: str
    issue_detected: bool
    severity: SeverityLevel
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the score is within valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the detection result.
        """
        return {
            "detector_name": self.detector_name,
            "issue_detected": self.issue_detected,
            "severity": self.severity.name,
            "score": self.score,
            "details": self.details,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """String representation of the detection result."""
        status = "DETECTED" if self.issue_detected else "NOT DETECTED"
        return (
            f"{self.detector_name}: {status}\n"
            f"  Severity: {self.severity.name}\n"
            f"  Score: {self.score:.3f}\n"
            f"  Recommendations: {', '.join(self.recommendations) if self.recommendations else 'None'}"
        )


@dataclass
class FixResult:
    """Container for fix operation results.
    
    Attributes:
        fixer_name: Name of the fixer that performed the operation.
        success: Whether the fix was successful.
        X_transformed: Transformed feature matrix.
        y_transformed: Transformed target vector (if applicable).
        details: Details about the fix operation.
        warnings: List of warnings generated during fixing.
    """
    fixer_name: str
    success: bool
    X_transformed: Optional[Union[np.ndarray, pd.DataFrame]] = None
    y_transformed: Optional[Union[np.ndarray, pd.Series]] = None
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "fixer_name": self.fixer_name,
            "success": self.success,
            "details": self.details,
            "warnings": self.warnings,
        }


class BaseDetector(ABC):
    """Abstract base class for all data quality detectors.
    
    All detectors in the DataSentry library must inherit from this class
    and implement the required abstract methods.
    
    Example:
        >>> class MyDetector(BaseDetector):
        ...     def __init__(self, threshold=0.5):
        ...         super().__init__("MyDetector")
        ...         self.threshold = threshold
        ...     
        ...     def detect(self, X, y=None):
        ...         # Detection logic here
        ...         return DetectionResult(...)
    """
    
    def __init__(self, name: str):
        """Initialize the detector.
        
        Args:
            name: Name of the detector.
        """
        self.name = name
        self._last_result: Optional[DetectionResult] = None
    
    @abstractmethod
    def detect(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect data quality issues.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,), optional.
        
        Returns:
            DetectionResult containing detection results.
        """
        pass
    
    def get_last_result(self) -> Optional[DetectionResult]:
        """Get the result from the last detection.
        
        Returns:
            The last DetectionResult or None if detect hasn't been called.
        """
        return self._last_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the detector's configuration and capabilities.
        
        Returns:
            Dictionary containing detector information.
        """
        return {
            "name": self.name,
            "description": self.__doc__,
            "parameters": self._get_parameters(),
        }
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get detector parameters for summary.
        
        Returns:
            Dictionary of parameter names and values.
        """
        # Exclude private attributes and common attributes
        exclude = {"name", "_last_result"}
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith("_") and k not in exclude
        }


class BaseFixer(ABC):
    """Abstract base class for all data quality fixers.
    
    All fixers in the DataSentry library must inherit from this class
    and implement the required abstract methods.
    
    Example:
        >>> class MyFixer(BaseFixer):
        ...     def __init__(self, strategy='mean'):
        ...         super().__init__("MyFixer")
        ...         self.strategy = strategy
        ...     
        ...     def fix(self, X, y=None, **kwargs):
        ...         # Fixing logic here
        ...         return FixResult(...)
    """
    
    def __init__(self, name: str):
        """Initialize the fixer.
        
        Args:
            name: Name of the fixer.
        """
        self.name = name
        self._last_result: Optional[FixResult] = None
    
    @abstractmethod
    def fix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> FixResult:
        """Fix data quality issues.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,), optional.
            **kwargs: Additional fixer-specific parameters.
        
        Returns:
            FixResult containing the transformed data and operation details.
        """
        pass
    
    def get_last_result(self) -> Optional[FixResult]:
        """Get the result from the last fix operation.
        
        Returns:
            The last FixResult or None if fix hasn't been called.
        """
        return self._last_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the fixer's configuration and capabilities.
        
        Returns:
            Dictionary containing fixer information.
        """
        return {
            "name": self.name,
            "description": self.__doc__,
            "parameters": self._get_parameters(),
        }
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get fixer parameters for summary.
        
        Returns:
            Dictionary of parameter names and values.
        """
        exclude = {"name", "_last_result"}
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith("_") and k not in exclude
        }


class BaseVisualizer(ABC):
    """Abstract base class for all data quality visualizers.
    
    All visualizers in the DataSentry library must inherit from this class
    and implement the required abstract methods.
    
    Example:
        >>> class MyVisualizer(BaseVisualizer):
        ...     def plot(self, X, y=None, **kwargs):
        ...         # Visualization logic here
        ...         return fig
    """
    
    def __init__(self, name: str):
        """Initialize the visualizer.
        
        Args:
            name: Name of the visualizer.
        """
        self.name = name
    
    @abstractmethod
    def plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> Any:
        """Create a visualization.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,), optional.
            **kwargs: Additional visualization-specific parameters.
        
        Returns:
            Matplotlib figure or other visualization object.
        """
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the visualizer's configuration.
        
        Returns:
            Dictionary containing visualizer information.
        """
        return {
            "name": self.name,
            "description": self.__doc__,
            "parameters": self._get_parameters(),
        }
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get visualizer parameters for summary.
        
        Returns:
            Dictionary of parameter names and values.
        """
        exclude = {"name"}
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith("_") and k not in exclude
        }

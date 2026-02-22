"""Main DataSentry orchestrator class.

This module provides the main DataSentry class that orchestrates all
detectors, fixers, and visualizers for comprehensive data quality management.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasentry.core.base import DetectionResult, FixResult, SeverityLevel
from datasentry.core.report import ReportGenerator
from datasentry.core.utils import validate_data

# Detectors
from datasentry.detectors.imbalance import ImbalanceDetector
from datasentry.detectors.label_noise import LabelNoiseDetector
from datasentry.detectors.data_leakage import DataLeakageDetector
from datasentry.detectors.missing_values import MissingValueDetector
from datasentry.detectors.outliers import OutlierDetector
from datasentry.detectors.redundancy import RedundancyDetector
from datasentry.detectors.shift import ShiftDetector

# Fixers
from datasentry.fixers.imbalance_fixer import ImbalanceFixer
from datasentry.fixers.label_noise_fixer import LabelNoiseFixer
from datasentry.fixers.data_leakage_fixer import DataLeakageFixer
from datasentry.fixers.missing_fixer import MissingValueFixer
from datasentry.fixers.outlier_fixer import OutlierFixer
from datasentry.fixers.redundancy_fixer import RedundancyFixer
from datasentry.fixers.shift_fixer import ShiftFixer

# Visualizers
from datasentry.visualizers.imbalance_viz import ImbalanceVisualizer
from datasentry.visualizers.noise_viz import NoiseVisualizer
from datasentry.visualizers.leakage_viz import LeakageVisualizer
from datasentry.visualizers.missing_viz import MissingVisualizer
from datasentry.visualizers.outlier_viz import OutlierVisualizer
from datasentry.visualizers.redundancy_viz import RedundancyVisualizer
from datasentry.visualizers.shift_viz import ShiftVisualizer


class DataSentry:
    """Main orchestrator for data quality detection and remediation.
    
    DataSentry provides a unified interface for:
    - Detecting data quality issues (imbalance, noise, leakage, etc.)
    - Fixing detected issues with appropriate strategies
    - Visualizing data quality metrics
    - Generating comprehensive reports
    
    Attributes:
        random_state: Random seed for reproducibility.
        verbose: Whether to print progress messages.
    
    Example:
        >>> ds = DataSentry(random_state=42, verbose=True)
        >>> 
        >>> # Individual detection
        >>> result = ds.detect_imbalance(X, y)
        >>> print(result.issue_detected)
        
        >>> # Full report
        >>> report = ds.generate_full_report(X, y, X_test=X_test)
        >>> print(report['health_score'])
        
        >>> # Fix issues
        >>> X_fixed, y_fixed = ds.fix_all(X, y)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Initialize DataSentry.
        
        Args:
            random_state: Random seed for reproducibility.
            verbose: Whether to print progress messages.
        """
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize detectors
        self._detectors = {
            "imbalance": ImbalanceDetector(),
            "label_noise": LabelNoiseDetector(random_state=random_state),
            "data_leakage": DataLeakageDetector(random_state=random_state),
            "missing_values": MissingValueDetector(),
            "outliers": OutlierDetector(random_state=random_state),
            "redundancy": RedundancyDetector(),
            "shift": ShiftDetector(random_state=random_state),
        }
        
        # Initialize fixers
        self._fixers = {
            "imbalance": ImbalanceFixer(random_state=random_state),
            "label_noise": LabelNoiseFixer(random_state=random_state),
            "data_leakage": DataLeakageFixer(),
            "missing_values": MissingValueFixer(),
            "outliers": OutlierFixer(),
            "redundancy": RedundancyFixer(),
            "shift": ShiftFixer(),
        }
        
        # Initialize visualizers
        self._visualizers = {
            "imbalance": ImbalanceVisualizer(),
            "label_noise": NoiseVisualizer(),
            "data_leakage": LeakageVisualizer(),
            "missing_values": MissingVisualizer(),
            "outliers": OutlierVisualizer(),
            "redundancy": RedundancyVisualizer(),
            "shift": ShiftVisualizer(),
        }
        
        # Store last results
        self._last_detection_results: Dict[str, DetectionResult] = {}
        self._last_fix_results: Dict[str, FixResult] = {}
    
    def _log(self, message: str) -> None:
        """Print log message if verbose.
        
        Args:
            message: Message to print.
        """
        if self.verbose:
            print(f"[DataSentry] {message}")
    
    # ==================== Detection Methods ====================
    
    def detect_imbalance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> DetectionResult:
        """Detect class imbalance.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        
        Returns:
            DetectionResult with imbalance analysis.
        """
        self._log("Detecting class imbalance...")
        result = self._detectors["imbalance"].detect(X, y)
        self._last_detection_results["imbalance"] = result
        return result
    
    def detect_label_noise(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> DetectionResult:
        """Detect label noise.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        
        Returns:
            DetectionResult with label noise analysis.
        """
        self._log("Detecting label noise...")
        result = self._detectors["label_noise"].detect(X, y)
        self._last_detection_results["label_noise"] = result
        return result
    
    def detect_data_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> DetectionResult:
        """Detect data leakage.
        
        Args:
            X: Feature matrix (training data).
            y: Target vector (optional).
            X_test: Test data (optional, for contamination check).
        
        Returns:
            DetectionResult with leakage analysis.
        """
        self._log("Detecting data leakage...")
        result = self._detectors["data_leakage"].detect(X, y, X_test)
        self._last_detection_results["data_leakage"] = result
        return result
    
    def detect_missing_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect missing values.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
        
        Returns:
            DetectionResult with missing value analysis.
        """
        self._log("Detecting missing values...")
        result = self._detectors["missing_values"].detect(X, y)
        self._last_detection_results["missing_values"] = result
        return result
    
    def detect_outliers(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect outliers.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
        
        Returns:
            DetectionResult with outlier analysis.
        """
        self._log("Detecting outliers...")
        result = self._detectors["outliers"].detect(X, y)
        self._last_detection_results["outliers"] = result
        return result
    
    def detect_redundancy(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect feature redundancy.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
        
        Returns:
            DetectionResult with redundancy analysis.
        """
        self._log("Detecting feature redundancy...")
        result = self._detectors["redundancy"].detect(X, y)
        self._last_detection_results["redundancy"] = result
        return result
    
    def detect_shift(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect data distribution shift.
        
        Args:
            X: Training feature matrix.
            y: Training target vector (optional).
            X_test: Test feature matrix (required).
            y_test: Test target vector (optional).
        
        Returns:
            DetectionResult with shift analysis.
        """
        self._log("Detecting data shift...")
        result = self._detectors["shift"].detect(X, y, X_test, y_test)
        self._last_detection_results["shift"] = result
        return result
    
    def detect_all(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, DetectionResult]:
        """Run all detection methods.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional, required for some detectors).
            X_test: Test data (optional, required for shift detection).
            y_test: Test target (optional).
        
        Returns:
            Dictionary mapping detector names to DetectionResults.
        """
        self._log("Running all detection methods...")
        
        results = {}
        
        # Detectors that don't require y
        results["missing_values"] = self.detect_missing_values(X, y)
        results["outliers"] = self.detect_outliers(X, y)
        results["redundancy"] = self.detect_redundancy(X, y)
        results["data_leakage"] = self.detect_data_leakage(X, y, X_test)
        
        # Detectors that require y
        if y is not None:
            results["imbalance"] = self.detect_imbalance(X, y)
            results["label_noise"] = self.detect_label_noise(X, y)
        
        # Shift detection requires X_test
        if X_test is not None:
            results["shift"] = self.detect_shift(X, y, X_test, y_test)
        
        self._log(f"Detection complete. {sum(1 for r in results.values() if r.issue_detected)} issues found.")
        
        return results
    
    # ==================== Fix Methods ====================
    
    def fix_imbalance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        method: str = "smote",
        **kwargs
    ) -> FixResult:
        """Fix class imbalance.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            method: Resampling method.
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log(f"Fixing imbalance using {method}...")
        fixer = ImbalanceFixer(method=method, random_state=self.random_state, **kwargs)
        result = fixer.fix(X, y)
        self._last_fix_results["imbalance"] = result
        return result
    
    def fix_label_noise(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        method: str = "remove",
        **kwargs
    ) -> FixResult:
        """Fix label noise.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            method: Cleaning method ('remove', 'relabel', 'weight').
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log(f"Fixing label noise using {method}...")
        fixer = LabelNoiseFixer(method=method, random_state=self.random_state, **kwargs)
        result = fixer.fix(X, y)
        self._last_fix_results["label_noise"] = result
        return result
    
    def fix_data_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        features_to_remove: Optional[List[str]] = None,
        **kwargs
    ) -> FixResult:
        """Fix data leakage.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            features_to_remove: Features to remove.
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log("Fixing data leakage...")
        fixer = DataLeakageFixer(**kwargs)
        result = fixer.fix(X, y, features_to_remove=features_to_remove)
        self._last_fix_results["data_leakage"] = result
        return result
    
    def fix_missing_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        strategy: str = "mean",
        **kwargs
    ) -> FixResult:
        """Fix missing values.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            strategy: Imputation strategy.
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log(f"Fixing missing values using {strategy}...")
        fixer = MissingValueFixer(strategy=strategy, **kwargs)
        result = fixer.fix(X, y)
        self._last_fix_results["missing_values"] = result
        return result
    
    def fix_outliers(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        method: str = "cap",
        **kwargs
    ) -> FixResult:
        """Fix outliers.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            method: Handling method ('remove', 'cap', 'transform', 'winsorize').
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log(f"Fixing outliers using {method}...")
        fixer = OutlierFixer(method=method, **kwargs)
        result = fixer.fix(X, y)
        self._last_fix_results["outliers"] = result
        return result
    
    def fix_redundancy(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        method: str = "remove",
        **kwargs
    ) -> FixResult:
        """Fix feature redundancy.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            method: Handling method ('remove', 'pca', 'variance_threshold').
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log(f"Fixing redundancy using {method}...")
        fixer = RedundancyFixer(method=method, **kwargs)
        result = fixer.fix(X, y)
        self._last_fix_results["redundancy"] = result
        return result
    
    def fix_shift(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        method: str = "standardize",
        **kwargs
    ) -> FixResult:
        """Fix data shift.
        
        Args:
            X: Training feature matrix.
            y: Training target vector (optional).
            X_test: Test feature matrix (optional).
            method: Adaptation method.
            **kwargs: Additional parameters.
        
        Returns:
            FixResult with transformed data.
        """
        self._log(f"Fixing shift using {method}...")
        fixer = ShiftFixer(method=method, **kwargs)
        result = fixer.fix(X, y, X_test=X_test)
        self._last_fix_results["shift"] = result
        return result
    
    def fix_all(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        fix_config: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.Series]]]:
        """Fix all detected issues.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            X_test: Test data (optional).
            fix_config: Configuration for each fixer.
                Example: {'missing_values': {'strategy': 'knn'}, 'outliers': {'method': 'cap'}}
        
        Returns:
            Tuple of (X_fixed, y_fixed).
        """
        self._log("Running fix_all...")
        
        fix_config = fix_config or {}
        X_current = X
        y_current = y
        
        # Fix missing values first (important for other operations)
        if "missing_values" in fix_config:
            result = self.fix_missing_values(X_current, y_current, **fix_config["missing_values"])
            if result.success:
                X_current = result.X_transformed
                y_current = result.y_transformed
        
        # Fix outliers
        if "outliers" in fix_config:
            result = self.fix_outliers(X_current, y_current, **fix_config["outliers"])
            if result.success:
                X_current = result.X_transformed
                y_current = result.y_transformed
        
        # Fix redundancy
        if "redundancy" in fix_config:
            result = self.fix_redundancy(X_current, y_current, **fix_config["redundancy"])
            if result.success:
                X_current = result.X_transformed
                y_current = result.y_transformed
        
        # Fix data leakage
        if "data_leakage" in fix_config:
            result = self.fix_data_leakage(X_current, y_current, **fix_config["data_leakage"])
            if result.success:
                X_current = result.X_transformed
                y_current = result.y_transformed
        
        # Fix imbalance (requires y)
        if y_current is not None and "imbalance" in fix_config:
            result = self.fix_imbalance(X_current, y_current, **fix_config["imbalance"])
            if result.success:
                X_current = result.X_transformed
                y_current = result.y_transformed
        
        # Fix label noise (requires y)
        if y_current is not None and "label_noise" in fix_config:
            result = self.fix_label_noise(X_current, y_current, **fix_config["label_noise"])
            if result.success:
                X_current = result.X_transformed
                y_current = result.y_transformed
        
        # Fix shift (requires X_test)
        if X_test is not None and "shift" in fix_config:
            result = self.fix_shift(X_current, y_current, X_test, **fix_config["shift"])
            if result.success:
                X_current = result.X_transformed
                # Note: X_test is also transformed in result.details
        
        self._log("Fix_all complete.")
        return X_current, y_current
    
    # ==================== Visualization Methods ====================
    
    def visualize_imbalance(
        self,
        y: Union[np.ndarray, pd.Series],
        plot_type: str = "both",
        **kwargs
    ) -> Any:
        """Visualize class imbalance.
        
        Args:
            y: Target vector.
            plot_type: Type of plot.
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["imbalance"].plot(y=y, plot_type=plot_type, **kwargs)
    
    def visualize_label_noise(
        self,
        y: Union[np.ndarray, pd.Series],
        plot_type: str = "confusion",
        **kwargs
    ) -> Any:
        """Visualize label noise.
        
        Args:
            y: Given labels.
            plot_type: Type of plot.
            **kwargs: Additional parameters (y_pred, noise_scores).
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["label_noise"].plot(y=y, plot_type=plot_type, **kwargs)
    
    def visualize_data_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        plot_type: str = "correlation",
        **kwargs
    ) -> Any:
        """Visualize data leakage.
        
        Args:
            X: Feature matrix.
            plot_type: Type of plot.
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["data_leakage"].plot(X, plot_type=plot_type, **kwargs)
    
    def visualize_missing_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        plot_type: str = "matrix",
        **kwargs
    ) -> Any:
        """Visualize missing values.
        
        Args:
            X: Feature matrix.
            plot_type: Type of plot.
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["missing_values"].plot(X, plot_type=plot_type, **kwargs)
    
    def visualize_outliers(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        plot_type: str = "box",
        **kwargs
    ) -> Any:
        """Visualize outliers.
        
        Args:
            X: Feature matrix.
            plot_type: Type of plot.
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["outliers"].plot(X, plot_type=plot_type, **kwargs)
    
    def visualize_redundancy(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        plot_type: str = "heatmap",
        **kwargs
    ) -> Any:
        """Visualize feature redundancy.
        
        Args:
            X: Feature matrix.
            plot_type: Type of plot.
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["redundancy"].plot(X, plot_type=plot_type, **kwargs)
    
    def visualize_shift(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        plot_type: str = "comparison",
        **kwargs
    ) -> Any:
        """Visualize data shift.
        
        Args:
            X: Training feature matrix.
            X_test: Test feature matrix.
            plot_type: Type of plot.
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure.
        """
        return self._visualizers["shift"].plot(X, X_test=X_test, plot_type=plot_type, **kwargs)
    
    def visualize_all(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """Create all visualizations.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            X_test: Test data (optional).
        
        Returns:
            Dictionary mapping visualization names to figures.
        """
        self._log("Generating all visualizations...")
        
        figures = {}
        
        try:
            figures["missing_values"] = self.visualize_missing_values(X, "matrix")
        except Exception as e:
            self._log(f"Could not generate missing_values visualization: {e}")
        
        try:
            figures["outliers"] = self.visualize_outliers(X, "box")
        except Exception as e:
            self._log(f"Could not generate outliers visualization: {e}")
        
        try:
            figures["redundancy"] = self.visualize_redundancy(X, "heatmap")
        except Exception as e:
            self._log(f"Could not generate redundancy visualization: {e}")
        
        if y is not None:
            try:
                figures["imbalance"] = self.visualize_imbalance(y, "both")
            except Exception as e:
                self._log(f"Could not generate imbalance visualization: {e}")
        
        if X_test is not None:
            try:
                figures["shift"] = self.visualize_shift(X, X_test, "comparison")
            except Exception as e:
                self._log(f"Could not generate shift visualization: {e}")
        
        return figures
    
    # ==================== Report Methods ====================
    
    def generate_full_report(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report.
        
        Args:
            X: Training feature matrix.
            y: Training target vector (optional).
            X_test: Test feature matrix (optional).
            y_test: Test target vector (optional).
            metadata: Additional metadata for the report.
        
        Returns:
            Dictionary containing the full report.
        """
        self._log("Generating full report...")
        
        # Run all detections
        detection_results = self.detect_all(X, y, X_test, y_test)
        
        # Generate report
        report_gen = ReportGenerator(
            results=list(detection_results.values()),
            metadata=metadata
        )
        
        report = report_gen.generate_summary()
        
        self._log(f"Report generated. Health Score: {report['report_metadata']['health_score']:.2%}")
        
        return report
    
    def get_last_detection_results(self) -> Dict[str, DetectionResult]:
        """Get results from last detection operations.
        
        Returns:
            Dictionary of detection results.
        """
        return self._last_detection_results.copy()
    
    def get_last_fix_results(self) -> Dict[str, FixResult]:
        """Get results from last fix operations.
        
        Returns:
            Dictionary of fix results.
        """
        return self._last_fix_results.copy()

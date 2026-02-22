"""Data shift detection for DataSentry library.

This module provides detection of data distribution shifts between datasets.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from datasentry.core.base import BaseDetector, DetectionResult, SeverityLevel
from datasentry.core.utils import validate_data, convert_to_dataframe


class ShiftDetector(BaseDetector):
    """Detect data distribution shifts between datasets.
    
    Data shift (or drift) occurs when the distribution of data changes
    over time or between datasets. Types of shift include:
    - Covariate shift: Feature distribution changes
    - Concept shift: Relationship between features and target changes
    - Label shift: Target distribution changes
    
    Attributes:
        shift_threshold: Threshold for flagging significant shift.
        methods: List of detection methods to use.
    
    Example:
        >>> detector = ShiftDetector()
        >>> X_train = np.random.randn(100, 5)
        >>> X_test = np.random.randn(50, 5) * 2  # Different distribution
        >>> result = detector.detect(X_train, X_test=X_test)
        >>> print(result.issue_detected)
        True
    """
    
    def __init__(
        self,
        shift_threshold: float = 0.1,
        methods: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
        n_bins: int = 20,
        random_state: Optional[int] = 42
    ):
        """Initialize the shift detector.
        
        Args:
            shift_threshold: Threshold for flagging significant shift.
            methods: List of methods to use ('ks', 'psi', 'adversarial', 'pca').
            p_value_threshold: P-value threshold for statistical tests.
            n_bins: Number of bins for distribution comparison.
            random_state: Random seed for reproducibility.
        """
        super().__init__("ShiftDetector")
        self.shift_threshold = shift_threshold
        self.methods = methods or ["ks", "psi", "adversarial"]
        self.p_value_threshold = p_value_threshold
        self.n_bins = n_bins
        self.random_state = random_state
    
    def detect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DetectionResult:
        """Detect data shift between training and test datasets.
        
        Args:
            X: Training feature matrix.
            y: Training target vector (optional).
            X_test: Test feature matrix (required for shift detection).
            y_test: Test target vector (optional, for concept shift).
        
        Returns:
            DetectionResult with shift analysis.
        """
        # Validate inputs
        X, y = validate_data(X, y)
        
        if X_test is None:
            return DetectionResult(
                detector_name=self.name,
                issue_detected=False,
                severity=SeverityLevel.NONE,
                score=0.0,
                details={"error": "X_test is required for shift detection"},
                recommendations=["Provide X_test to detect distribution shift"],
            )
        
        X_test, y_test = validate_data(X_test, y_test)
        
        # Convert to DataFrames
        df_train = convert_to_dataframe(X)
        df_test = convert_to_dataframe(X_test)
        
        # Ensure same columns
        common_cols = list(set(df_train.columns) & set(df_test.columns))
        if not common_cols:
            return DetectionResult(
                detector_name=self.name,
                issue_detected=False,
                severity=SeverityLevel.NONE,
                score=0.0,
                details={"error": "No common features between train and test"},
                recommendations=["Ensure train and test have the same features"],
            )
        
        df_train = df_train[common_cols]
        df_test = df_test[common_cols]
        
        # Run detection methods
        all_results = {}
        
        if "ks" in self.methods:
            all_results["ks_test"] = self._ks_test(df_train, df_test)
        
        if "psi" in self.methods:
            all_results["psi"] = self._population_stability_index(df_train, df_test)
        
        if "adversarial" in self.methods:
            all_results["adversarial"] = self._adversarial_validation(df_train, df_test)
        
        if "pca" in self.methods:
            all_results["pca"] = self._pca_based_detection(df_train, df_test)
        
        # Check for concept shift if targets provided
        if y is not None and y_test is not None:
            all_results["concept_shift"] = self._check_concept_shift(y, y_test)
        
        # Aggregate results
        has_shift = any(r.get("has_shift", False) for r in all_results.values())
        
        # Calculate overall severity
        severity = self._determine_severity(all_results)
        
        # Calculate overall score
        score = self._calculate_shift_score(all_results)
        
        # Prepare details
        details = {
            "n_features": len(common_cols),
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "methods_used": self.methods,
            "results_by_method": all_results,
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results, details)
        
        result = DetectionResult(
            detector_name=self.name,
            issue_detected=has_shift,
            severity=severity,
            score=score,
            details=details,
            recommendations=recommendations,
        )
        
        self._last_result = result
        return result
    
    def _ks_test(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for each feature.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
        
        Returns:
            Dictionary with KS test results.
        """
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        
        shifted_features = []
        ks_stats = []
        
        for col in numeric_cols:
            train_vals = df_train[col].dropna()
            test_vals = df_test[col].dropna()
            
            if len(train_vals) == 0 or len(test_vals) == 0:
                continue
            
            stat, p_value = stats.ks_2samp(train_vals, test_vals)
            ks_stats.append({
                "feature": col,
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "shifted": p_value < self.p_value_threshold,
            })
            
            if p_value < self.p_value_threshold:
                shifted_features.append(col)
        
        # Calculate proportion of shifted features
        shift_ratio = len(shifted_features) / len(numeric_cols) if len(numeric_cols) > 0 else 0
        
        return {
            "has_shift": len(shifted_features) > 0,
            "n_shifted_features": len(shifted_features),
            "shift_ratio": round(shift_ratio, 4),
            "shifted_features": shifted_features[:20],
            "ks_statistics": sorted(ks_stats, key=lambda x: x["statistic"], reverse=True)[:20],
            "method": "Kolmogorov-Smirnov test",
        }
    
    def _population_stability_index(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) for each feature.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
        
        Returns:
            Dictionary with PSI results.
        """
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        
        psi_results = []
        shifted_features = []
        
        for col in numeric_cols:
            train_vals = df_train[col].dropna()
            test_vals = df_test[col].dropna()
            
            if len(train_vals) == 0 or len(test_vals) == 0:
                continue
            
            # Create bins based on training data
            min_val, max_val = train_vals.min(), train_vals.max()
            if min_val == max_val:
                continue
            
            bins = np.linspace(min_val, max_val, self.n_bins + 1)
            
            # Calculate distributions
            train_hist, _ = np.histogram(train_vals, bins=bins)
            test_hist, _ = np.histogram(test_vals, bins=bins)
            
            # Add small constant to avoid division by zero
            train_dist = (train_hist + 0.001) / (train_hist.sum() + 0.001 * len(train_hist))
            test_dist = (test_hist + 0.001) / (test_hist.sum() + 0.001 * len(test_hist))
            
            # Calculate PSI
            psi = np.sum((test_dist - train_dist) * np.log(test_dist / train_dist))
            
            psi_results.append({
                "feature": col,
                "psi": round(float(psi), 4),
            })
            
            # PSI interpretation: <0.1 negligible, 0.1-0.25 moderate, >0.25 significant
            if psi > 0.25:
                shifted_features.append(col)
        
        # Calculate average PSI
        avg_psi = np.mean([r["psi"] for r in psi_results]) if psi_results else 0
        
        return {
            "has_shift": len(shifted_features) > 0,
            "n_shifted_features": len(shifted_features),
            "avg_psi": round(float(avg_psi), 4),
            "shifted_features": shifted_features,
            "psi_by_feature": sorted(psi_results, key=lambda x: x["psi"], reverse=True)[:20],
            "method": "Population Stability Index",
        }
    
    def _adversarial_validation(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Use adversarial validation to detect distribution shift.
        
        Trains a classifier to distinguish between train and test samples.
        High accuracy indicates distribution shift.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
        
        Returns:
            Dictionary with adversarial validation results.
        """
        # Select numeric columns and handle missing values
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        
        train_data = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
        test_data = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
        
        # Create combined dataset
        X_combined = pd.concat([train_data, test_data], ignore_index=True)
        y_combined = np.array([0] * len(train_data) + [1] * len(test_data))
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        try:
            scores = cross_val_score(clf, X_combined, y_combined, cv=3, scoring='roc_auc')
            avg_auc = scores.mean()
            
            # Fit to get feature importances
            clf.fit(X_combined, y_combined)
            importances = dict(zip(numeric_cols, clf.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # AUC > 0.6 indicates significant shift
            has_shift = avg_auc > 0.6
            
            return {
                "has_shift": has_shift,
                "auc_score": round(float(avg_auc), 4),
                "interpretation": "AUC > 0.6 indicates distribution shift",
                "top_discriminative_features": [
                    {"feature": f, "importance": round(float(i), 4)}
                    for f, i in top_features
                ],
                "method": "Adversarial Validation",
            }
        except Exception as e:
            return {
                "has_shift": False,
                "error": str(e),
                "method": "Adversarial Validation",
            }
    
    def _pca_based_detection(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect shift using PCA reconstruction error.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
        
        Returns:
            Dictionary with PCA-based detection results.
        """
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        
        # Prepare data
        train_data = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
        test_data = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
        
        # Standardize
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Fit PCA
        n_components = min(10, train_scaled.shape[1], train_scaled.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(train_scaled)
        
        # Calculate reconstruction errors
        train_reconstructed = pca.inverse_transform(pca.transform(train_scaled))
        test_reconstructed = pca.inverse_transform(pca.transform(test_scaled))
        
        train_error = np.mean((train_scaled - train_reconstructed) ** 2)
        test_error = np.mean((test_scaled - test_reconstructed) ** 2)
        
        # Significant difference in reconstruction error indicates shift
        error_ratio = test_error / (train_error + 1e-10)
        has_shift = error_ratio > 2.0 or error_ratio < 0.5
        
        return {
            "has_shift": has_shift,
            "train_reconstruction_error": round(float(train_error), 6),
            "test_reconstruction_error": round(float(test_error), 6),
            "error_ratio": round(float(error_ratio), 4),
            "n_components": n_components,
            "explained_variance_ratio": round(float(sum(pca.explained_variance_ratio_)), 4),
            "method": "PCA Reconstruction Error",
        }
    
    def _check_concept_shift(
        self,
        y_train: Union[np.ndarray, pd.Series],
        y_test: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """Check for concept shift (change in target distribution).
        
        Args:
            y_train: Training target.
            y_test: Test target.
        
        Returns:
            Dictionary with concept shift analysis.
        """
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
        y_test_arr = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)
        
        # Get class distributions
        train_dist = pd.Series(y_train_arr).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(y_test_arr).value_counts(normalize=True).sort_index()
        
        # Align distributions
        all_classes = sorted(set(train_dist.index) | set(test_dist.index))
        train_aligned = np.array([train_dist.get(c, 0) for c in all_classes])
        test_aligned = np.array([test_dist.get(c, 0) for c in all_classes])
        
        # Calculate Jensen-Shannon distance
        js_distance = jensenshannon(train_aligned, test_aligned)
        js_distance = 0 if np.isnan(js_distance) else js_distance
        
        # Perform chi-square test
        train_counts = pd.Series(y_train_arr).value_counts().sort_index()
        test_counts = pd.Series(y_test_arr).value_counts().sort_index()
        
        # Align counts
        train_counts_aligned = np.array([train_counts.get(c, 0) for c in all_classes])
        test_counts_aligned = np.array([test_counts.get(c, 0) for c in all_classes])
        
        try:
            chi2, p_value = stats.chisquare(test_counts_aligned, train_counts_aligned)
        except Exception:
            chi2, p_value = 0, 1.0
        
        has_shift = js_distance > 0.1 or p_value < self.p_value_threshold
        
        return {
            "has_shift": has_shift,
            "js_distance": round(float(js_distance), 4),
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 4),
            "train_distribution": train_dist.to_dict(),
            "test_distribution": test_dist.to_dict(),
            "method": "Target Distribution Comparison",
        }
    
    def _determine_severity(self, results: Dict[str, Any]) -> SeverityLevel:
        """Determine severity based on shift results.
        
        Args:
            results: Dictionary with shift detection results.
        
        Returns:
            SeverityLevel enum value.
        """
        # Count methods detecting shift
        shift_count = sum(1 for r in results.values() if r.get("has_shift", False))
        total_methods = len(results)
        
        if total_methods == 0:
            return SeverityLevel.NONE
        
        shift_ratio = shift_count / total_methods
        
        # Check for critical indicators
        if "adversarial" in results:
            adv = results["adversarial"]
            if adv.get("auc_score", 0) > 0.8:
                return SeverityLevel.CRITICAL
        
        if "psi" in results:
            psi = results["psi"]
            if psi.get("avg_psi", 0) > 0.5:
                return SeverityLevel.CRITICAL
        
        # General severity based on consensus
        if shift_ratio >= 0.75:
            return SeverityLevel.HIGH
        elif shift_ratio >= 0.5:
            return SeverityLevel.MEDIUM
        elif shift_ratio >= 0.25:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE
    
    def _calculate_shift_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall shift score.
        
        Args:
            results: Dictionary with shift detection results.
        
        Returns:
            Score between 0 and 1.
        """
        scores = []
        
        if "ks_test" in results:
            ks = results["ks_test"]
            scores.append(ks.get("shift_ratio", 0))
        
        if "psi" in results:
            psi = results["psi"]
            avg_psi = psi.get("avg_psi", 0)
            scores.append(min(1.0, avg_psi * 2))  # Scale PSI
        
        if "adversarial" in results:
            adv = results["adversarial"]
            auc = adv.get("auc_score", 0.5)
            scores.append(max(0, (auc - 0.5) * 2))  # Scale AUC from 0.5-1.0 to 0-1
        
        if "pca" in results:
            pca = results["pca"]
            error_ratio = pca.get("error_ratio", 1.0)
            scores.append(min(1.0, abs(error_ratio - 1)))
        
        return max(scores) if scores else 0.0
    
    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        details: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for shift.
        
        Args:
            results: Dictionary with shift detection results.
            details: Detection details.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        # Check which methods detected shift
        shifted_methods = [m for m, r in results.items() if r.get("has_shift", False)]
        
        if not shifted_methods:
            recommendations.append("No significant distribution shift detected")
            return recommendations
        
        recommendations.append(
            f"Distribution shift detected by: {', '.join(shifted_methods)}"
        )
        
        # Feature-specific recommendations
        if "ks_test" in results:
            ks = results["ks_test"]
            shifted = ks.get("shifted_features", [])
            if shifted:
                recommendations.append(
                    f"Features with significant shift: {shifted[:10]}"
                )
        
        if "adversarial" in results:
            adv = results["adversarial"]
            top_features = adv.get("top_discriminative_features", [])
            if top_features:
                feature_names = [f["feature"] for f in top_features[:5]]
                recommendations.append(
                    f"Most discriminative features: {feature_names}"
                )
        
        # General recommendations
        recommendations.append(
            "Consider retraining your model on more recent data"
        )
        recommendations.append(
            "Implement monitoring to detect drift in production"
        )
        
        if "concept_shift" in results:
            recommendations.append(
                "Target distribution has changed - review business context"
            )
        
        # Severity-based recommendations
        if len(shifted_methods) >= 3:
            recommendations.append(
                "CRITICAL: Multiple detection methods indicate significant shift. "
                "Model performance may be severely degraded."
            )
            recommendations.append(
                "Consider using domain adaptation techniques"
            )
        
        return recommendations
    
    def get_shifted_features(self) -> List[str]:
        """Get list of features identified as shifted.
        
        Returns:
            List of shifted feature names or empty list.
        """
        if self._last_result is None:
            return []
        
        shifted = set()
        results = self._last_result.details.get("results_by_method", {})
        
        if "ks_test" in results:
            shifted.update(results["ks_test"].get("shifted_features", []))
        
        if "psi" in results:
            shifted.update(results["psi"].get("shifted_features", []))
        
        return sorted(list(shifted))

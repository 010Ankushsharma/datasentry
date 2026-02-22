"""Tests for OutlierDetector."""

import numpy as np
import pytest

from datasentry import OutlierDetector
from datasentry.core.base import SeverityLevel


class TestOutlierDetector:
    """Test suite for OutlierDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = OutlierDetector()
        assert detector.name == "OutlierDetector"
        assert detector.method == "iqr"
        
    def test_detect_iqr(self, data_with_outliers):
        """Test IQR outlier detection."""
        X, y = data_with_outliers
        detector = OutlierDetector(method='iqr')
        result = detector.detect(X, y)
        
        assert result.detector_name == "OutlierDetector"
        assert result.details["method"] == "iqr"
        assert result.details["n_outliers"] >= 2  # We added 2 outliers
        
    def test_detect_zscore(self, data_with_outliers):
        """Test Z-score outlier detection."""
        X, y = data_with_outliers
        detector = OutlierDetector(method='zscore', threshold=2.0)
        result = detector.detect(X, y)
        
        assert result.details["method"] == "zscore"
        assert result.details["n_outliers"] > 0
        
    def test_detect_isolation_forest(self, data_with_outliers):
        """Test Isolation Forest outlier detection."""
        X, y = data_with_outliers
        detector = OutlierDetector(method='isolation_forest')
        result = detector.detect(X, y)
        
        assert result.details["method"] == "isolation_forest"
        
    def test_get_outlier_indices(self, data_with_outliers):
        """Test getting outlier indices."""
        X, y = data_with_outliers
        detector = OutlierDetector()
        detector.detect(X, y)
        
        indices = detector.get_outlier_indices()
        assert indices is not None
        assert len(indices) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

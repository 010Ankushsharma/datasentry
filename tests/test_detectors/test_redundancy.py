"""Tests for RedundancyDetector."""

import numpy as np
import pytest

from datasentry import RedundancyDetector
from datasentry.core.base import SeverityLevel


class TestRedundancyDetector:
    """Test suite for RedundancyDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = RedundancyDetector()
        assert detector.name == "RedundancyDetector"
        assert detector.correlation_threshold == 0.95
        
    def test_detect_no_redundancy(self, sample_data):
        """Test detection on data without redundancy."""
        X, y = sample_data
        detector = RedundancyDetector()
        result = detector.detect(X, y)
        
        assert result.detector_name == "RedundancyDetector"
        
    def test_detect_correlation(self, data_with_redundancy):
        """Test correlation detection."""
        X, y = data_with_redundancy
        detector = RedundancyDetector(correlation_threshold=0.9)
        result = detector.detect(X, y)
        
        assert "correlated_features" in result.details
        corr_info = result.details["correlated_features"]
        assert corr_info["has_correlations"] is True
        
    def test_get_redundant_features(self, data_with_redundancy):
        """Test getting redundant features."""
        X, y = data_with_redundancy
        detector = RedundancyDetector(correlation_threshold=0.9)
        detector.detect(X, y)
        
        redundant = detector.get_redundant_features()
        assert len(redundant) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

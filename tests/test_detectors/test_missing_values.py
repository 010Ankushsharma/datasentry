"""Tests for MissingValueDetector."""

import numpy as np
import pandas as pd
import pytest

from datasentry import MissingValueDetector
from datasentry.core.base import SeverityLevel


class TestMissingValueDetector:
    """Test suite for MissingValueDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = MissingValueDetector()
        assert detector.name == "MissingValueDetector"
        assert detector.missing_threshold == 0.05
        
    def test_detect_no_missing(self, sample_data):
        """Test detection on data without missing values."""
        X, y = sample_data
        detector = MissingValueDetector()
        result = detector.detect(X, y)
        
        assert result.detector_name == "MissingValueDetector"
        assert result.details["total_missing"] == 0
        assert result.details["overall_missing_ratio"] == 0.0
        
    def test_detect_with_missing(self, data_with_missing):
        """Test detection on data with missing values."""
        X, y = data_with_missing
        detector = MissingValueDetector()
        result = detector.detect(X, y)
        
        assert result.details["total_missing"] > 0
        assert result.details["overall_missing_ratio"] > 0
        assert "features_with_missing" in result.details
        
    def test_detect_with_dataframe(self):
        """Test detection with pandas DataFrame."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [None, 2, 3, 4],
        })
        detector = MissingValueDetector()
        result = detector.detect(df)
        
        assert result.details["total_missing"] == 2
        assert result.issue_detected is True
        
    def test_get_missing_summary(self, data_with_missing):
        """Test getting missing summary."""
        X, y = data_with_missing
        detector = MissingValueDetector()
        detector.detect(X, y)
        
        summary = detector.get_missing_summary()
        assert summary is not None
        assert "overall_missing_ratio" in summary
        assert "severity" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

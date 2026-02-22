"""Tests for ShiftDetector."""

import numpy as np
import pytest

from datasentry import ShiftDetector
from datasentry.core.base import SeverityLevel


class TestShiftDetector:
    """Test suite for ShiftDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = ShiftDetector()
        assert detector.name == "ShiftDetector"
        
    def test_detect_requires_test_data(self, sample_data):
        """Test that X_test is required."""
        X, y = sample_data
        detector = ShiftDetector()
        result = detector.detect(X, y)
        
        assert result.issue_detected is False
        assert "error" in result.details
        
    def test_detect_shift(self, shifted_data):
        """Test shift detection."""
        X_train, y_train, X_test, y_test = shifted_data
        detector = ShiftDetector()
        result = detector.detect(X_train, y_train, X_test, y_test)
        
        assert result.detector_name == "ShiftDetector"
        assert "results_by_method" in result.details
        
    def test_get_shifted_features(self, shifted_data):
        """Test getting shifted features."""
        X_train, y_train, X_test, y_test = shifted_data
        detector = ShiftDetector()
        detector.detect(X_train, y_train, X_test, y_test)
        
        shifted = detector.get_shifted_features()
        assert isinstance(shifted, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

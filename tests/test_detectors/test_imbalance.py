"""Tests for ImbalanceDetector."""

import numpy as np
import pytest

from datasentry import ImbalanceDetector
from datasentry.core.base import SeverityLevel


class TestImbalanceDetector:
    """Test suite for ImbalanceDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = ImbalanceDetector()
        assert detector.name == "ImbalanceDetector"
        assert detector.imbalance_threshold == 3.0
        
    def test_detect_balanced_data(self, sample_data):
        """Test detection on balanced data."""
        X, y = sample_data
        detector = ImbalanceDetector()
        result = detector.detect(X, y)
        
        assert result.detector_name == "ImbalanceDetector"
        assert isinstance(bool(result.issue_detected), bool)
        assert isinstance(float(result.score), float)
        assert 0 <= result.score <= 1
        
    def test_detect_imbalanced_data(self, imbalanced_data):
        """Test detection on imbalanced data."""
        X, y = imbalanced_data
        detector = ImbalanceDetector()
        result = detector.detect(X, y)
        
        assert bool(result.issue_detected) is True
        assert result.details["imbalance_ratio"] > 1
        assert "class_distribution" in result.details
        
    def test_severity_levels(self, imbalanced_data):
        """Test severity level assignment."""
        X, y = imbalanced_data
        detector = ImbalanceDetector()
        result = detector.detect(X, y)
        
        assert isinstance(result.severity, SeverityLevel)
        assert result.severity in [
            SeverityLevel.NONE, SeverityLevel.LOW, 
            SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL
        ]
        
    def test_get_summary(self):
        """Test summary method."""
        detector = ImbalanceDetector()
        summary = detector.get_summary()
        
        assert "name" in summary
        assert "parameters" in summary
        assert summary["name"] == "ImbalanceDetector"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

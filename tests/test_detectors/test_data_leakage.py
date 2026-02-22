"""Tests for DataLeakageDetector."""

import numpy as np
import pytest

from datasentry import DataLeakageDetector
from datasentry.core.base import SeverityLevel


class TestDataLeakageDetector:
    """Test suite for DataLeakageDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = DataLeakageDetector()
        assert detector.name == "DataLeakageDetector"
        assert detector.leakage_threshold == 0.95
        
    def test_detect_no_leakage(self, sample_data):
        """Test detection on clean data."""
        X, y = sample_data
        detector = DataLeakageDetector()
        result = detector.detect(X, y)
        
        assert result.detector_name == "DataLeakageDetector"
        assert "checks_performed" in result.details
        
    def test_detect_duplicates(self, data_with_duplicates):
        """Test duplicate detection."""
        X, y = data_with_duplicates
        detector = DataLeakageDetector()
        result = detector.detect(X, y)
        
        assert "duplicates" in result.details
        dup_info = result.details["duplicates"]
        assert bool(dup_info["has_duplicates"]) is True
        assert dup_info["n_duplicates"] >= 10
        
    def test_detect_with_test_data(self, sample_data):
        """Test detection with train/test split."""
        X, y = sample_data
        X_test = X[:20]  # Some overlap
        detector = DataLeakageDetector()
        result = detector.detect(X, y, X_test=X_test)
        
        assert "train_test_contamination" in result.details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

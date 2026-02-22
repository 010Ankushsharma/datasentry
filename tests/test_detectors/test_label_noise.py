"""Tests for LabelNoiseDetector."""

import numpy as np
import pytest

from datasentry import LabelNoiseDetector
from datasentry.core.base import SeverityLevel


class TestLabelNoiseDetector:
    """Test suite for LabelNoiseDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = LabelNoiseDetector()
        assert detector.name == "LabelNoiseDetector"
        assert detector.method == "confident_learning"
        
    def test_detect(self, sample_data):
        """Test noise detection."""
        X, y = sample_data
        detector = LabelNoiseDetector()
        result = detector.detect(X, y)
        
        assert result.detector_name == "LabelNoiseDetector"
        assert "method" in result.details
        assert result.details["method"] in ["confident_learning", "knn_consensus"]
        
    def test_get_noisy_samples(self, sample_data):
        """Test getting noisy sample indices."""
        X, y = sample_data
        detector = LabelNoiseDetector()
        detector.detect(X, y)
        
        noisy = detector.get_noisy_samples()
        assert noisy is not None
        assert isinstance(noisy, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for main DataSentry orchestrator."""

import numpy as np
import pytest

from datasentry import DataSentry


class TestDataSentry:
    """Test suite for DataSentry orchestrator."""
    
    def test_init(self):
        """Test DataSentry initialization."""
        ds = DataSentry(verbose=False)
        assert ds.random_state == 42
        assert ds.verbose is False
        
    def test_detect_all(self, sample_data):
        """Test detect_all method."""
        X, y = sample_data
        ds = DataSentry(verbose=False)
        results = ds.detect_all(X, y)
        
        assert isinstance(results, dict)
        assert "missing_values" in results
        assert "outliers" in results
        assert "imbalance" in results
        
    def test_generate_full_report(self, sample_data):
        """Test generate_full_report method."""
        X, y = sample_data
        ds = DataSentry(verbose=False)
        report = ds.generate_full_report(X, y)
        
        assert isinstance(report, dict)
        assert "report_metadata" in report
        assert "health_score" in report["report_metadata"]
        assert "detailed_results" in report
        
    def test_fix_all(self, data_with_missing, data_with_outliers):
        """Test fix_all method."""
        # Use data with missing values
        X, y = data_with_missing
        ds = DataSentry(verbose=False)
        
        fix_config = {
            'missing_values': {'strategy': 'mean'},
        }
        
        X_fixed, y_fixed = ds.fix_all(X, y, fix_config=fix_config)
        
        assert X_fixed is not None
        # Should have no missing values after fixing
        assert not np.isnan(X_fixed).any()
        
    def test_get_last_detection_results(self, sample_data):
        """Test getting last detection results."""
        X, y = sample_data
        ds = DataSentry(verbose=False)
        ds.detect_all(X, y)
        
        results = ds.get_last_detection_results()
        assert isinstance(results, dict)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

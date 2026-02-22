"""Tests for MissingValueFixer."""

import numpy as np
import pandas as pd
import pytest

from datasentry import MissingValueFixer


class TestMissingValueFixer:
    """Test suite for MissingValueFixer."""
    
    def test_init(self):
        """Test fixer initialization."""
        fixer = MissingValueFixer()
        assert fixer.name == "MissingValueFixer"
        assert fixer.strategy == "mean"
        
    def test_fix_mean(self, data_with_missing):
        """Test mean imputation."""
        X, y = data_with_missing
        fixer = MissingValueFixer(strategy='mean')
        result = fixer.fix(X, y)
        
        assert result.success is True
        assert result.details["missing_after"] == 0
        
    def test_fix_median(self, data_with_missing):
        """Test median imputation."""
        X, y = data_with_missing
        fixer = MissingValueFixer(strategy='median')
        result = fixer.fix(X, y)
        
        assert result.success is True
        assert result.details["missing_after"] == 0
        
    def test_fix_drop(self, data_with_missing):
        """Test dropping rows with missing values."""
        X, y = data_with_missing
        fixer = MissingValueFixer(strategy='drop')
        result = fixer.fix(X, y)
        
        assert result.success is True
        assert result.details["final_samples"] < result.details["original_samples"]
        
    def test_get_imputation_values(self, data_with_missing):
        """Test getting imputation values."""
        X, y = data_with_missing
        fixer = MissingValueFixer(strategy='mean')
        fixer.fix(X, y)
        
        values = fixer.get_imputation_values()
        assert values is not None
        assert isinstance(values, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

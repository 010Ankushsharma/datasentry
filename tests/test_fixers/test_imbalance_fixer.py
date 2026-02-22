"""Tests for ImbalanceFixer."""

import numpy as np
import pytest

from datasentry import ImbalanceFixer


class TestImbalanceFixer:
    """Test suite for ImbalanceFixer."""
    
    def test_init(self):
        """Test fixer initialization."""
        fixer = ImbalanceFixer()
        assert fixer.name == "ImbalanceFixer"
        assert fixer.method == "smote"
        
    def test_fix_class_weights(self, imbalanced_data):
        """Test class weights fix."""
        X, y = imbalanced_data
        fixer = ImbalanceFixer(method='class_weights')
        result = fixer.fix(X, y)
        
        assert result.success is True
        assert "class_weights" in result.details
        assert result.X_transformed is not None
        
    def test_fix_random_over(self, imbalanced_data):
        """Test random oversampling."""
        X, y = imbalanced_data
        fixer = ImbalanceFixer(method='random_over')
        result = fixer.fix(X, y)
        
        assert result.success is True
        # Should have more samples after oversampling
        assert len(result.y_transformed) >= len(y)
        
    def test_get_class_weights(self, imbalanced_data):
        """Test getting class weights."""
        X, y = imbalanced_data
        fixer = ImbalanceFixer(method='class_weights')
        fixer.fix(X, y)
        
        weights = fixer.get_class_weights()
        assert weights is not None
        assert isinstance(weights, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

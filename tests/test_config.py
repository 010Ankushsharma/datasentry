import pytest
from datasentry.config import DataSentryConfig


def test_default_config():
    config = DataSentryConfig()
    assert config.outlier_contamination == 0.05


def test_invalid_contamination():
    with pytest.raises(ValueError):
        DataSentryConfig(outlier_contamination=0.9)

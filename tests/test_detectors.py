import numpy as np
from datasentry.detectors import imbalance


def test_imbalance_detect_balanced():
    y = np.array([0, 1, 0, 1])
    result = imbalance.detect(y, threshold=3.0)
    assert result["imbalance_score"] < 0.5


def test_imbalance_detect_imbalanced():
    y = np.array([0, 0, 0, 0, 1])
    result = imbalance.detect(y, threshold=3.0)
    assert result["is_problematic"] is True

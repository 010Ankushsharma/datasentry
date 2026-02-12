import numpy as np
from datasentry import analyze


def test_analyze_runs():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    report = analyze(X, y)
    assert hasattr(report, "score")
    assert report.score <= 100

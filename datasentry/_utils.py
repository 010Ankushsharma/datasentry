# datasentry/_utils.py
import numpy as np

def validate_X(X):
    X = np.array(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must contain only numeric values.")

    return X


def validate_y(y):
    y = np.array(y)

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional.")

    return y

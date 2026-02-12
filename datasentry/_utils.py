from typing import Union
import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def to_numpy(X: ArrayLike) -> np.ndarray:
    """
    Convert input features to validated NumPy array.

    Ensures:
        - Numeric dtype
        - 2D shape

    Raises
    ------
    ValueError
        If data is non-numeric.
    """

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("All features must be numeric.")

    return X


def validate_y(y: ArrayLike) -> np.ndarray:
    """
    Validate target vector.

    Ensures:
        - 1D
        - Non-empty
    """

    if isinstance(y, pd.Series):
        y = y.values

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("Target y must be 1-dimensional.")

    if len(y) == 0:
        raise ValueError("Target y cannot be empty.")

    return y

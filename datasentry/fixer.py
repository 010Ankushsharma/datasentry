# datasentry/fixer.py
import numpy as np
from ._utils import validate_X, validate_y

class AutoFixer:

    def __init__(self, issues, random_state=42):
        self.issues = issues
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fix(self, X, y):

        X = validate_X(X)
        y = validate_y(y)

        if self.issues.get("imbalance", {}).get("is_imbalanced"):
            X, y = self._oversample(X, y)

        if self.issues.get("outliers", {}).get("is_problematic"):
            X, y = self._remove_outliers(X, y)

        return X, y

    def _oversample(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_new, y_new = [], []

        for cls in classes:
            X_cls = X[y == cls]
            y_cls = y[y == cls]

            repeat = max_count - len(X_cls)

            if repeat > 0:
                idx = np.random.choice(len(X_cls), repeat, replace=True)
                X_cls = np.vstack([X_cls, X_cls[idx]])
                y_cls = np.concatenate([y_cls, y_cls[idx]])

            X_new.append(X_cls)
            y_new.append(y_cls)

        X_final = np.vstack(X_new)
        y_final = np.concatenate(y_new)

        shuffle_idx = np.random.permutation(len(y_final))
        return X_final[shuffle_idx], y_final[shuffle_idx]

    def _remove_outliers(self, X, y):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8

        z_scores = np.abs((X - mean) / std)
        mask = (z_scores < 3).all(axis=1)

        return X[mask], y[mask]

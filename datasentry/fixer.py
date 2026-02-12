from typing import Tuple
import numpy as np
from imblearn.over_sampling import SMOTE
from ._utils import to_numpy, validate_y


class AutoFixer:
    """
    Automatically fix detected issues where possible.
    """

    def __init__(self, issues: dict, config):
        self.issues = issues
        self.config = config

    def fix(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply automatic corrections.
        """

        X = to_numpy(X)
        y = validate_y(y)

        if self.issues.get("imbalance", {}).get("is_problematic"):
            smote = SMOTE(random_state=self.config.random_state)
            X, y = smote.fit_resample(X, y)

        return X, y

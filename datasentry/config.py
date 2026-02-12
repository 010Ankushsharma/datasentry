from dataclasses import dataclass


@dataclass
class DataSentryConfig:
    """
    Configuration object for DataSentry.

    Parameters
    ----------
    imbalance_threshold : float
        Controls sensitivity of imbalance detection.
    outlier_contamination : float
        Expected proportion of outliers (0 < value < 0.5).
    drift_threshold : float
        PSI threshold for distribution shift detection.
    leakage_threshold : float
        Mutual information threshold for leakage detection.
    noise_threshold : float
        Label noise ratio threshold.
    random_state : int
        Random seed for reproducibility.
    """

    imbalance_threshold: float = 3.0
    outlier_contamination: float = 0.05
    drift_threshold: float = 0.2
    leakage_threshold: float = 0.5
    noise_threshold: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        if not (0 < self.outlier_contamination < 0.5):
            raise ValueError("outlier_contamination must be between 0 and 0.5.")

        if self.imbalance_threshold <= 0:
            raise ValueError("imbalance_threshold must be positive.")

        if self.noise_threshold < 0:
            raise ValueError("noise_threshold must be >= 0.")

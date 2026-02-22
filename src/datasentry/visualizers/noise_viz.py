"""Label noise visualizer for DataSentry library.

This module provides visualizations for label noise analysis.
"""

from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from datasentry.core.base import BaseVisualizer


class NoiseVisualizer(BaseVisualizer):
    """Visualize label noise in datasets.
    
    This visualizer creates various plots to analyze label noise:
    - Confusion matrix: Given vs predicted labels
    - Noise score distribution
    - Class-wise noise analysis
    
    Example:
        >>> viz = NoiseVisualizer()
        >>> y_true = np.array([0, 0, 1, 1, 2, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
        >>> fig = viz.plot(y=y_true, y_pred=y_pred, plot_type='confusion')
    """
    
    def __init__(
        self,
        figsize: tuple = (10, 8),
        color_palette: str = "Reds",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the noise visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("NoiseVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "confusion",
        **kwargs
    ) -> plt.Figure:
        """Create noise visualization.
        
        Args:
            X: Feature matrix (not used, for API consistency).
            y: Given labels (target).
            plot_type: Type of plot ('confusion', 'scores', 'class_noise').
            **kwargs: Additional parameters.
                - y_pred: Predicted labels for confusion matrix.
                - noise_scores: Noise scores for each sample.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        
        if y is None:
            raise ValueError("y is required for noise visualization")
        
        if plot_type == "confusion":
            y_pred = kwargs.get('y_pred')
            if y_pred is None:
                raise ValueError("y_pred required for confusion matrix")
            fig = self._plot_confusion(y, y_pred)
        elif plot_type == "scores":
            noise_scores = kwargs.get('noise_scores')
            if noise_scores is None:
                raise ValueError("noise_scores required for score plot")
            fig = self._plot_scores(noise_scores)
        elif plot_type == "class_noise":
            y_pred = kwargs.get('y_pred')
            if y_pred is None:
                raise ValueError("y_pred required for class noise plot")
            fig = self._plot_class_noise(y, y_pred)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_confusion(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> plt.Figure:
        """Plot confusion matrix between given and predicted labels.
        
        Args:
            y_true: True/given labels.
            y_pred: Predicted labels.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=self.color_palette,
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('Given Label', fontsize=12)
        ax.set_title('Label Noise Confusion Matrix\n(Given vs Model-Predicted)', 
                     fontsize=14, fontweight='bold')
        
        return fig
    
    def _plot_scores(self, noise_scores: np.ndarray) -> plt.Figure:
        """Plot distribution of noise scores.
        
        Args:
            noise_scores: Noise scores for each sample.
        
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        axes[0].hist(noise_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(noise_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(noise_scores):.3f}')
        axes[0].axvline(np.median(noise_scores), color='green', linestyle='--',
                       label=f'Median: {np.median(noise_scores):.3f}')
        axes[0].set_xlabel('Noise Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Noise Scores', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        axes[1].boxplot([noise_scores], labels=['All Samples'])
        axes[1].set_ylabel('Noise Score', fontsize=12)
        axes[1].set_title('Noise Score Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        return fig
    
    def _plot_class_noise(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> plt.Figure:
        """Plot noise rate per class.
        
        Args:
            y_true: True/given labels.
            y_pred: Predicted labels.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        
        classes = np.unique(y_true_arr)
        noise_rates = []
        total_counts = []
        
        for cls in classes:
            mask = y_true_arr == cls
            n_total = mask.sum()
            n_noisy = (y_true_arr[mask] != y_pred_arr[mask]).sum()
            noise_rate = n_noisy / n_total if n_total > 0 else 0
            noise_rates.append(noise_rate)
            total_counts.append(n_total)
        
        colors = sns.color_palette("RdYlGn_r", len(classes))
        bars = ax.bar(classes.astype(str), noise_rates, color=colors, edgecolor='black')
        
        # Add count labels
        for bar, count in zip(bars, total_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1%}\n(n={count})',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Noise Rate', fontsize=12)
        ax.set_title('Label Noise Rate by Class', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(noise_rates) * 1.2 if noise_rates else 1)
        ax.grid(axis='y', alpha=0.3)
        
        return fig

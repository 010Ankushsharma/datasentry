"""Class imbalance visualizer for DataSentry library.

This module provides visualizations for class imbalance analysis.
"""

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasentry.core.base import BaseVisualizer
from datasentry.core.utils import get_class_distribution


class ImbalanceVisualizer(BaseVisualizer):
    """Visualize class imbalance in datasets.
    
    This visualizer creates various plots to analyze class distribution:
    - Bar chart: Class counts
    - Pie chart: Class proportions
    - Comparison: Before/after resampling
    
    Example:
        >>> viz = ImbalanceVisualizer()
        >>> y = np.array([0, 0, 0, 0, 1, 1])
        >>> fig = viz.plot(y=y, plot_type='bar')
    """
    
    def __init__(
        self,
        figsize: tuple = (12, 5),
        color_palette: str = "viridis",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the imbalance visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("ImbalanceVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "both",
        **kwargs
    ) -> plt.Figure:
        """Create imbalance visualization.
        
        Args:
            X: Feature matrix (not used, for API consistency).
            y: Target vector.
            plot_type: Type of plot ('bar', 'pie', 'both').
            **kwargs: Additional parameters.
                - y_resampled: Resampled target for comparison.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        
        if y is None:
            raise ValueError("y is required for imbalance visualization")
        
        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
        class_dist = get_class_distribution(y)
        
        if plot_type == "both":
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)
            self._plot_bar(class_dist, axes[0])
            self._plot_pie(class_dist, axes[1])
        elif plot_type == "bar":
            fig, ax = plt.subplots(figsize=(self.figsize[0] // 2, self.figsize[1]))
            self._plot_bar(class_dist, ax)
        elif plot_type == "pie":
            fig, ax = plt.subplots(figsize=(self.figsize[0] // 2, self.figsize[1]))
            self._plot_pie(class_dist, ax)
        elif plot_type == "comparison":
            y_resampled = kwargs.get('y_resampled')
            if y_resampled is None:
                raise ValueError("y_resampled required for comparison plot")
            fig = self._plot_comparison(y, y_resampled)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_bar(self, class_dist: pd.Series, ax: plt.Axes) -> None:
        """Plot class distribution as bar chart.
        
        Args:
            class_dist: Class distribution Series.
            ax: Matplotlib axes.
        """
        colors = sns.color_palette(self.color_palette, len(class_dist))
        bars = ax.bar(class_dist.index.astype(str), class_dist.values, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_pie(self, class_dist: pd.Series, ax: plt.Axes) -> None:
        """Plot class distribution as pie chart.
        
        Args:
            class_dist: Class distribution Series.
            ax: Matplotlib axes.
        """
        colors = sns.color_palette(self.color_palette, len(class_dist))
        proportions = class_dist / class_dist.sum()
        
        wedges, texts, autotexts = ax.pie(
            proportions,
            labels=class_dist.index.astype(str),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(class_dist)
        )
        
        ax.set_title('Class Proportions', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_comparison(
        self,
        y_original: Union[np.ndarray, pd.Series],
        y_resampled: Union[np.ndarray, pd.Series]
    ) -> plt.Figure:
        """Plot before/after comparison.
        
        Args:
            y_original: Original target.
            y_resampled: Resampled target.
        
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        dist_original = get_class_distribution(y_original)
        dist_resampled = get_class_distribution(y_resampled)
        
        colors = sns.color_palette(self.color_palette, max(len(dist_original), len(dist_resampled)))
        
        # Original
        bars1 = axes[0].bar(dist_original.index.astype(str), dist_original.values, color=colors[:len(dist_original)])
        axes[0].set_title('Original Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
        
        # Resampled
        bars2 = axes[1].bar(dist_resampled.index.astype(str), dist_resampled.values, color=colors[:len(dist_resampled)])
        axes[1].set_title('After Resampling', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig

"""Outlier visualizer for DataSentry library.

This module provides visualizations for outlier analysis.
"""

from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasentry.core.base import BaseVisualizer
from datasentry.core.utils import convert_to_dataframe


class OutlierVisualizer(BaseVisualizer):
    """Visualize outliers in datasets.
    
    This visualizer creates various plots to analyze outliers:
    - Box plot: Distribution with outliers
    - Scatter plot: 2D outlier visualization
    - Distribution: Histogram with outlier highlighting
    
    Example:
        >>> viz = OutlierVisualizer()
        >>> X = np.array([[1], [2], [3], [100], [2], [3]])
        >>> fig = viz.plot(X, plot_type='box')
    """
    
    def __init__(
        self,
        figsize: tuple = (12, 8),
        color_palette: str = "Set2",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the outlier visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("OutlierVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "box",
        **kwargs
    ) -> plt.Figure:
        """Create outlier visualization.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional, for colored scatter).
            plot_type: Type of plot ('box', 'scatter', 'distribution').
            **kwargs: Additional parameters.
                - outlier_indices: Indices of detected outliers.
                - feature: Feature name or index to plot.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        df = convert_to_dataframe(X)
        
        if plot_type == "box":
            fig = self._plot_box(df)
        elif plot_type == "scatter":
            feature_x = kwargs.get('feature_x', 0)
            feature_y = kwargs.get('feature_y', 1)
            outlier_indices = kwargs.get('outlier_indices')
            fig = self._plot_scatter(df, y, feature_x, feature_y, outlier_indices)
        elif plot_type == "distribution":
            feature = kwargs.get('feature', 0)
            outlier_indices = kwargs.get('outlier_indices')
            fig = self._plot_distribution(df, feature, outlier_indices)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_box(self, df: pd.DataFrame) -> plt.Figure:
        """Plot box plots for all numeric features.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Matplotlib figure.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numeric features',
                   ha='center', va='center', fontsize=14)
            return fig
        
        n_features = len(numeric_df.columns)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        colors = sns.color_palette(self.color_palette, n_features)
        
        for idx, (col, ax) in enumerate(zip(numeric_df.columns, axes)):
            bp = ax.boxplot(numeric_df[col].dropna(), patch_artist=True)
            bp['boxes'][0].set_facecolor(colors[idx])
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes) if isinstance(axes, np.ndarray) else 1):
            if isinstance(axes, np.ndarray):
                axes[idx].axis('off')
        
        plt.suptitle('Box Plots for Outlier Detection', fontsize=16, fontweight='bold', y=1.02)
        
        return fig
    
    def _plot_scatter(
        self,
        df: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]],
        feature_x: Union[str, int],
        feature_y: Union[str, int],
        outlier_indices: Optional[List[int]]
    ) -> plt.Figure:
        """Plot 2D scatter plot with outlier highlighting.
        
        Args:
            df: Input DataFrame.
            y: Target vector for coloring.
            feature_x: X-axis feature.
            feature_y: Y-axis feature.
            outlier_indices: Indices of outliers to highlight.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Get feature names
        if isinstance(feature_x, int):
            feature_x = numeric_df.columns[feature_x] if feature_x < len(numeric_df.columns) else numeric_df.columns[0]
        if isinstance(feature_y, int):
            feature_y = numeric_df.columns[feature_y] if feature_y < len(numeric_df.columns) else numeric_df.columns[min(1, len(numeric_df.columns)-1)]
        
        x_data = numeric_df[feature_x]
        y_data = numeric_df[feature_y]
        
        # Plot based on target
        if y is not None:
            y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
            classes = np.unique(y_arr)
            colors = sns.color_palette(self.color_palette, len(classes))
            
            for cls, color in zip(classes, colors):
                mask = y_arr == cls
                ax.scatter(x_data[mask], y_data[mask], c=[color], label=f'Class {cls}',
                          alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.legend(title='Class')
        else:
            ax.scatter(x_data, y_data, c='steelblue', alpha=0.6,
                      edgecolors='black', linewidth=0.5)
        
        # Highlight outliers
        if outlier_indices is not None:
            ax.scatter(x_data.iloc[outlier_indices], y_data.iloc[outlier_indices],
                      c='red', s=100, marker='x', linewidths=2,
                      label='Outliers', zorder=5)
            ax.legend()
        
        ax.set_xlabel(feature_x, fontsize=12)
        ax.set_ylabel(feature_y, fontsize=12)
        ax.set_title(f'2D Scatter Plot: {feature_x} vs {feature_y}\n(Outliers marked with X)',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        return fig
    
    def _plot_distribution(
        self,
        df: pd.DataFrame,
        feature: Union[str, int],
        outlier_indices: Optional[List[int]]
    ) -> plt.Figure:
        """Plot distribution histogram with outlier highlighting.
        
        Args:
            df: Input DataFrame.
            feature: Feature to plot.
            outlier_indices: Indices of outliers.
        
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Get feature name
        if isinstance(feature, int):
            feature = numeric_df.columns[feature] if feature < len(numeric_df.columns) else numeric_df.columns[0]
        
        data = numeric_df[feature].dropna()
        
        # Histogram
        axes[0].hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Highlight outlier bins
        if outlier_indices is not None and len(outlier_indices) > 0:
            outlier_data = data.iloc[outlier_indices]
            axes[0].hist(outlier_data, bins=30, color='red', edgecolor='black',
                        alpha=0.5, label='Outliers')
            axes[0].legend()
        
        axes[0].set_xlabel(feature, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution Histogram', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot for the same feature
        bp = axes[1].boxplot(data, patch_artist=True, vert=False)
        bp['boxes'][0].set_facecolor('steelblue')
        
        axes[1].set_xlabel(feature, fontsize=12)
        axes[1].set_title('Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}"
        if outlier_indices is not None:
            stats_text += f"\nOutliers: {len(outlier_indices)}"
        
        axes[1].text(0.98, 0.02, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig

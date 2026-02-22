"""Missing value visualizer for DataSentry library.

This module provides visualizations for missing value analysis.
"""

from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasentry.core.base import BaseVisualizer
from datasentry.core.utils import convert_to_dataframe


class MissingVisualizer(BaseVisualizer):
    """Visualize missing values in datasets.
    
    This visualizer creates various plots to analyze missing values:
    - Missingness matrix: Visual representation of missing pattern
    - Bar chart: Missing counts per feature
    - Heatmap: Missing value correlations
    
    Example:
        >>> viz = MissingVisualizer()
        >>> X = pd.DataFrame({'a': [1, None, 3], 'b': [None, 2, 3]})
        >>> fig = viz.plot(X, plot_type='matrix')
    """
    
    def __init__(
        self,
        figsize: tuple = (12, 8),
        color_palette: str = "viridis",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the missing value visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("MissingVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "matrix",
        **kwargs
    ) -> plt.Figure:
        """Create missing value visualization.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            plot_type: Type of plot ('matrix', 'bar', 'heatmap').
            **kwargs: Additional parameters.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        df = convert_to_dataframe(X)
        
        if plot_type == "matrix":
            fig = self._plot_matrix(df)
        elif plot_type == "bar":
            fig = self._plot_bar(df)
        elif plot_type == "heatmap":
            fig = self._plot_heatmap(df)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_matrix(self, df: pd.DataFrame) -> plt.Figure:
        """Plot missingness matrix.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create missing indicator matrix
        missing_matrix = df.isnull().astype(int)
        
        # Sort by missingness pattern
        missing_counts = missing_matrix.sum(axis=1)
        sorted_idx = missing_counts.sort_values(ascending=False).index
        missing_matrix_sorted = missing_matrix.loc[sorted_idx]
        
        # Plot
        cmap = sns.color_palette(["#E8E8E8", "#D62728"], as_cmap=True)
        sns.heatmap(
            missing_matrix_sorted,
            cmap=cmap,
            cbar_kws={'label': 'Missing', 'ticks': [0, 1]},
            xticklabels=True,
            yticklabels=False,
            ax=ax
        )
        
        ax.set_title('Missing Value Pattern Matrix\n(Sorted by missingness)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Samples', fontsize=12)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        return fig
    
    def _plot_bar(self, df: pd.DataFrame) -> plt.Figure:
        """Plot missing value counts as bar chart.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Missing counts
        missing_counts = df.isnull().sum().sort_values(ascending=True)
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) == 0:
            axes[0].text(0.5, 0.5, 'No Missing Values',
                        ha='center', va='center', fontsize=14)
            axes[1].text(0.5, 0.5, 'No Missing Values',
                        ha='center', va='center', fontsize=14)
            return fig
        
        colors = sns.color_palette("Reds", len(missing_counts))
        
        # Count plot
        bars = axes[0].barh(missing_counts.index.astype(str), missing_counts.values, color=colors)
        axes[0].set_xlabel('Number of Missing Values', fontsize=12)
        axes[0].set_ylabel('Features', fontsize=12)
        axes[0].set_title('Missing Value Counts', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            axes[0].text(width, bar.get_y() + bar.get_height()/2.,
                        f' {int(width)}',
                        ha='left', va='center', fontsize=9)
        
        # Percentage plot
        missing_pct = (df.isnull().mean() * 100).sort_values(ascending=True)
        missing_pct = missing_pct[missing_pct > 0]
        
        bars2 = axes[1].barh(missing_pct.index.astype(str), missing_pct.values, color=colors)
        axes[1].set_xlabel('Percentage Missing (%)', fontsize=12)
        axes[1].set_title('Missing Value Percentages', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for bar in bars2:
            width = bar.get_width()
            axes[1].text(width, bar.get_y() + bar.get_height()/2.,
                        f' {width:.1f}%',
                        ha='left', va='center', fontsize=9)
        
        return fig
    
    def _plot_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """Plot missing value correlation heatmap.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate correlation between missing indicators
        missing_df = df.isnull()
        
        # Only include columns with missing values
        cols_with_missing = missing_df.columns[missing_df.any()].tolist()
        
        if len(cols_with_missing) < 2:
            ax.text(0.5, 0.5, 'Insufficient features with missing values',
                   ha='center', va='center', fontsize=14)
            return fig
        
        missing_subset = missing_df[cols_with_missing]
        corr_matrix = missing_subset.corr()
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Missing Value Correlation\n(Do features miss together?)',
                    fontsize=14, fontweight='bold')
        
        return fig

"""Feature redundancy visualizer for DataSentry library.

This module provides visualizations for feature redundancy analysis.
"""

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasentry.core.base import BaseVisualizer
from datasentry.core.utils import convert_to_dataframe


class RedundancyVisualizer(BaseVisualizer):
    """Visualize feature redundancy in datasets.
    
    This visualizer creates various plots to analyze redundancy:
    - Correlation heatmap: Full correlation matrix
    - Cluster map: Hierarchical clustering of correlations
    - Scatter matrix: Pairwise feature relationships
    
    Example:
        >>> viz = RedundancyVisualizer()
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [4, 5, 6]})
        >>> fig = viz.plot(X, plot_type='heatmap')
    """
    
    def __init__(
        self,
        figsize: tuple = (12, 10),
        color_palette: str = "RdBu_r",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the redundancy visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("RedundancyVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "heatmap",
        **kwargs
    ) -> plt.Figure:
        """Create redundancy visualization.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            plot_type: Type of plot ('heatmap', 'clustermap', 'pairs').
            **kwargs: Additional parameters.
                - high_corr_pairs: List of high correlation pairs to highlight.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        df = convert_to_dataframe(X)
        
        if plot_type == "heatmap":
            high_corr_pairs = kwargs.get('high_corr_pairs')
            fig = self._plot_heatmap(df, high_corr_pairs)
        elif plot_type == "clustermap":
            fig = self._plot_clustermap(df)
        elif plot_type == "pairs":
            features = kwargs.get('features')
            fig = self._plot_pairs(df, features)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_heatmap(
        self,
        df: pd.DataFrame,
        high_corr_pairs: Optional[List[Dict]] = None
    ) -> plt.Figure:
        """Plot correlation heatmap.
        
        Args:
            df: Input DataFrame.
            high_corr_pairs: Optional list of high correlation pairs.
        
        Returns:
            Matplotlib figure.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient numeric features',
                   ha='center', va='center', fontsize=14)
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        corr_matrix = numeric_df.corr()
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap=self.color_palette,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        title = 'Feature Correlation Matrix'
        if high_corr_pairs:
            title += f'\n({len(high_corr_pairs)} high correlations detected)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig
    
    def _plot_clustermap(self, df: pd.DataFrame) -> plt.Figure:
        """Plot hierarchical clustering of correlations.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Matplotlib figure.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 3:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient numeric features for clustering',
                   ha='center', va='center', fontsize=14)
            return fig
        
        corr_matrix = numeric_df.corr()
        
        g = sns.clustermap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap=self.color_palette,
            center=0,
            vmin=-1,
            vmax=1,
            figsize=(self.figsize[0], self.figsize[1]),
            dendrogram_ratio=0.2,
            cbar_pos=(0.02, 0.8, 0.05, 0.18)
        )
        
        g.fig.suptitle('Hierarchical Clustering of Feature Correlations', 
                      fontsize=14, fontweight='bold', y=1.02)
        
        return g.fig
    
    def _plot_pairs(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot scatter matrix for pairwise relationships.
        
        Args:
            df: Input DataFrame.
            features: List of features to plot (if None, select top correlated).
        
        Returns:
            Matplotlib figure.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient numeric features',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Select features
        if features is None:
            # Select top 4 features with highest variance
            variances = numeric_df.var().sort_values(ascending=False)
            features = variances.head(4).index.tolist()
        
        features = [f for f in features if f in numeric_df.columns][:4]
        
        if len(features) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient features selected',
                   ha='center', va='center', fontsize=14)
            return fig
        
        n = len(features)
        fig, axes = plt.subplots(n, n, figsize=(self.figsize[0], self.figsize[0]))
        
        for i, feat_i in enumerate(features):
            for j, feat_j in enumerate(features):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(numeric_df[feat_i].dropna(), bins=20, color='steelblue', edgecolor='black')
                    ax.set_title(feat_i, fontsize=10, fontweight='bold')
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(numeric_df[feat_j], numeric_df[feat_i],
                              alpha=0.5, c='steelblue', edgecolors='black', linewidth=0.5)
                    
                    # Add correlation coefficient
                    corr = numeric_df[feat_i].corr(numeric_df[feat_j])
                    ax.text(0.05, 0.95, f'r={corr:.2f}',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                if i == n - 1:
                    ax.set_xlabel(feat_j, fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(feat_i, fontsize=9)
                else:
                    ax.set_yticklabels([])
        
        plt.suptitle('Pairwise Feature Relationships', fontsize=16, fontweight='bold', y=1.02)
        
        return fig

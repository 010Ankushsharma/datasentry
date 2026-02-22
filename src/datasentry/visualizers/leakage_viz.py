"""Data leakage visualizer for DataSentry library.

This module provides visualizations for data leakage analysis.
"""

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasentry.core.base import BaseVisualizer
from datasentry.core.utils import convert_to_dataframe


class LeakageVisualizer(BaseVisualizer):
    """Visualize data leakage in datasets.
    
    This visualizer creates various plots to analyze data leakage:
    - Correlation heatmap: Feature correlations
    - Feature importance: From adversarial validation
    - Distribution comparison: Train vs test
    
    Example:
        >>> viz = LeakageVisualizer()
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> fig = viz.plot(X, plot_type='correlation')
    """
    
    def __init__(
        self,
        figsize: tuple = (12, 10),
        color_palette: str = "RdBu_r",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the leakage visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("LeakageVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "correlation",
        **kwargs
    ) -> plt.Figure:
        """Create leakage visualization.
        
        Args:
            X: Feature matrix.
            y: Target vector (optional).
            plot_type: Type of plot ('correlation', 'feature_importance', 'distribution').
            **kwargs: Additional parameters.
                - feature_importances: Dict of feature importances.
                - X_test: Test data for distribution comparison.
                - high_corr_pairs: List of highly correlated feature pairs.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        df = convert_to_dataframe(X)
        
        if plot_type == "correlation":
            high_corr_pairs = kwargs.get('high_corr_pairs')
            fig = self._plot_correlation(df, high_corr_pairs)
        elif plot_type == "feature_importance":
            feature_importances = kwargs.get('feature_importances')
            if feature_importances is None:
                raise ValueError("feature_importances required")
            fig = self._plot_feature_importance(feature_importances)
        elif plot_type == "distribution":
            X_test = kwargs.get('X_test')
            if X_test is None:
                raise ValueError("X_test required for distribution plot")
            feature = kwargs.get('feature', 0)
            fig = self._plot_distribution_comparison(df, X_test, feature)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_correlation(
        self,
        df: pd.DataFrame,
        high_corr_pairs: Optional[List[Dict]] = None
    ) -> plt.Figure:
        """Plot correlation heatmap.
        
        Args:
            df: Input DataFrame.
            high_corr_pairs: Optional list of high correlation pairs to highlight.
        
        Returns:
            Matplotlib figure.
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numeric features for correlation',
                   ha='center', va='center', fontsize=14)
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        corr_matrix = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
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
        
        ax.set_title('Feature Correlation Matrix\n(Lower Triangle)', 
                    fontsize=14, fontweight='bold')
        
        # Highlight high correlations if provided
        if high_corr_pairs:
            title_suffix = f"\n{len(high_corr_pairs)} high correlations detected"
            ax.set_title(ax.get_title() + title_suffix, fontsize=14, fontweight='bold')
        
        return fig
    
    def _plot_feature_importance(self, feature_importances: Dict[str, float]) -> plt.Figure:
        """Plot feature importances from adversarial validation.
        
        Args:
            feature_importances: Dictionary mapping feature names to importance scores.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by importance
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features[:20]]  # Top 20
        importances = [f[1] for f in sorted_features[:20]]
        
        colors = sns.color_palette("viridis", len(features))
        bars = ax.barh(features, importances, color=colors)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance for Train/Test Discrimination\n(Higher = More Shifted)',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        return fig
    
    def _plot_distribution_comparison(
        self,
        X_train: pd.DataFrame,
        X_test: Union[np.ndarray, pd.DataFrame],
        feature: Union[str, int]
    ) -> plt.Figure:
        """Plot distribution comparison between train and test.
        
        Args:
            X_train: Training DataFrame.
            X_test: Test data.
            feature: Feature name or index to plot.
        
        Returns:
            Matplotlib figure.
        """
        df_test = convert_to_dataframe(X_test)
        
        # Get feature name
        if isinstance(feature, int):
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            if feature >= len(numeric_cols):
                feature = numeric_cols[0] if len(numeric_cols) > 0 else X_train.columns[0]
            else:
                feature = numeric_cols[feature]
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram comparison
        train_data = X_train[feature].dropna()
        test_data = df_test[feature].dropna() if feature in df_test.columns else []
        
        axes[0].hist(train_data, bins=30, alpha=0.6, label='Train', color='steelblue', density=True)
        if len(test_data) > 0:
            axes[0].hist(test_data, bins=30, alpha=0.6, label='Test', color='coral', density=True)
        axes[0].set_xlabel(feature, fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [train_data]
        labels = ['Train']
        if len(test_data) > 0:
            data_to_plot.append(test_data)
            labels.append('Test')
        
        axes[1].boxplot(data_to_plot, labels=labels)
        axes[1].set_ylabel(feature, fontsize=12)
        axes[1].set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        return fig

"""Data shift visualizer for DataSentry library.

This module provides visualizations for data shift analysis.
"""

from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasentry.core.base import BaseVisualizer
from datasentry.core.utils import convert_to_dataframe


class ShiftVisualizer(BaseVisualizer):
    """Visualize data distribution shift between datasets.
    
    This visualizer creates various plots to analyze shift:
    - Distribution comparison: Overlaid histograms
    - QQ plot: Quantile-quantile comparison
    - Drift timeline: Shift over time (if timestamps provided)
    
    Example:
        >>> viz = ShiftVisualizer()
        >>> X_train = np.random.randn(100, 5)
        >>> X_test = np.random.randn(50, 5) * 2
        >>> fig = viz.plot(X_train, X_test=X_test, plot_type='comparison')
    """
    
    def __init__(
        self,
        figsize: tuple = (14, 8),
        color_palette: str = "Set2",
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """Initialize the shift visualizer.
        
        Args:
            figsize: Figure size for plots.
            color_palette: Color palette for plots.
            style: Matplotlib style to use.
        """
        super().__init__("ShiftVisualizer")
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style
    
    def plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        plot_type: str = "comparison",
        **kwargs
    ) -> plt.Figure:
        """Create shift visualization.
        
        Args:
            X: Training feature matrix.
            y: Target vector (optional).
            plot_type: Type of plot ('comparison', 'qq', 'ks').
            **kwargs: Additional parameters.
                - X_test: Test data for comparison (required).
                - feature: Feature name or index to plot.
        
        Returns:
            Matplotlib figure object.
        """
        plt.style.use(self.style)
        
        X_test = kwargs.get('X_test')
        if X_test is None:
            raise ValueError("X_test is required for shift visualization")
        
        df_train = convert_to_dataframe(X)
        df_test = convert_to_dataframe(X_test)
        
        if plot_type == "comparison":
            fig = self._plot_comparison(df_train, df_test, kwargs.get('feature', 0))
        elif plot_type == "qq":
            fig = self._plot_qq(df_train, df_test, kwargs.get('feature', 0))
        elif plot_type == "ks":
            fig = self._plot_ks_statistic(df_train, df_test)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        plt.tight_layout()
        return fig
    
    def _plot_comparison(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature: Union[str, int]
    ) -> plt.Figure:
        """Plot distribution comparison between train and test.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
            feature: Feature to plot.
        
        Returns:
            Matplotlib figure.
        """
        numeric_train = df_train.select_dtypes(include=[np.number])
        numeric_test = df_test.select_dtypes(include=[np.number])
        
        # Get feature name
        if isinstance(feature, int):
            common_cols = list(set(numeric_train.columns) & set(numeric_test.columns))
            if feature >= len(common_cols):
                feature = common_cols[0] if common_cols else numeric_train.columns[0]
            else:
                feature = common_cols[feature]
        
        if feature not in numeric_train.columns or feature not in numeric_test.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Feature {feature} not found in both datasets',
                   ha='center', va='center', fontsize=14)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        train_data = numeric_train[feature].dropna()
        test_data = numeric_test[feature].dropna()
        
        # Histogram comparison
        axes[0, 0].hist(train_data, bins=30, alpha=0.6, label='Train', 
                       color='steelblue', edgecolor='black', density=True)
        axes[0, 0].hist(test_data, bins=30, alpha=0.6, label='Test',
                       color='coral', edgecolor='black', density=True)
        axes[0, 0].set_xlabel(feature, fontsize=12)
        axes[0, 0].set_ylabel('Density', fontsize=12)
        axes[0, 0].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Box plot
        bp = axes[0, 1].boxplot([train_data, test_data], labels=['Train', 'Test'],
                                patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        axes[0, 1].set_ylabel(feature, fontsize=12)
        axes[0, 1].set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # CDF comparison
        train_sorted = np.sort(train_data)
        test_sorted = np.sort(test_data)
        train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
        test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)
        
        axes[1, 0].plot(train_sorted, train_cdf, label='Train', color='steelblue', linewidth=2)
        axes[1, 0].plot(test_sorted, test_cdf, label='Test', color='coral', linewidth=2)
        axes[1, 0].set_xlabel(feature, fontsize=12)
        axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1, 0].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Statistics comparison
        stats_train = {
            'Mean': train_data.mean(),
            'Median': train_data.median(),
            'Std': train_data.std(),
            'Min': train_data.min(),
            'Max': train_data.max(),
        }
        stats_test = {
            'Mean': test_data.mean(),
            'Median': test_data.median(),
            'Std': test_data.std(),
            'Min': test_data.min(),
            'Max': test_data.max(),
        }
        
        axes[1, 1].axis('off')
        
        table_data = []
        for stat in stats_train.keys():
            table_data.append([stat, f"{stats_train[stat]:.3f}", f"{stats_test[stat]:.3f}"])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Statistic', 'Train', 'Test'],
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.2, 0.8, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('Statistics Comparison', fontsize=14, fontweight='bold', pad=20)
        
        return fig
    
    def _plot_qq(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature: Union[str, int]
    ) -> plt.Figure:
        """Plot QQ plot for quantile comparison.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
            feature: Feature to plot.
        
        Returns:
            Matplotlib figure.
        """
        from scipy import stats
        
        numeric_train = df_train.select_dtypes(include=[np.number])
        numeric_test = df_test.select_dtypes(include=[np.number])
        
        # Get feature name
        if isinstance(feature, int):
            common_cols = list(set(numeric_train.columns) & set(numeric_test.columns))
            if feature >= len(common_cols):
                feature = common_cols[0] if common_cols else numeric_train.columns[0]
            else:
                feature = common_cols[feature]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        train_data = numeric_train[feature].dropna()
        test_data = numeric_test[feature].dropna()
        
        # Calculate quantiles
        quantiles = np.linspace(0, 100, 100)
        train_quantiles = np.percentile(train_data, quantiles)
        test_quantiles = np.percentile(test_data, quantiles)
        
        # Plot
        ax.scatter(train_quantiles, test_quantiles, alpha=0.6, c='steelblue', edgecolors='black')
        
        # Add diagonal line
        min_val = min(train_quantiles.min(), test_quantiles.min())
        max_val = max(train_quantiles.max(), test_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
        
        ax.set_xlabel(f'Train: {feature}', fontsize=12)
        ax.set_ylabel(f'Test: {feature}', fontsize=12)
        ax.set_title(f'Q-Q Plot: {feature}\n(Points on diagonal indicate no shift)',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig
    
    def _plot_ks_statistic(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> plt.Figure:
        """Plot KS statistics for all features.
        
        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
        
        Returns:
            Matplotlib figure.
        """
        from scipy import stats
        
        numeric_train = df_train.select_dtypes(include=[np.number])
        numeric_test = df_test.select_dtypes(include=[np.number])
        
        common_cols = list(set(numeric_train.columns) & set(numeric_test.columns))
        
        if not common_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No common numeric features',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Calculate KS statistics
        ks_results = []
        for col in common_cols:
            train_vals = numeric_train[col].dropna()
            test_vals = numeric_test[col].dropna()
            
            if len(train_vals) > 0 and len(test_vals) > 0:
                ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)
                ks_results.append({
                    'feature': col,
                    'ks_statistic': ks_stat,
                    'p_value': p_value
                })
        
        # Sort by KS statistic
        ks_results.sort(key=lambda x: x['ks_statistic'], reverse=True)
        
        # Plot top features
        top_n = min(15, len(ks_results))
        top_results = ks_results[:top_n]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        features = [r['feature'] for r in top_results]
        ks_stats = [r['ks_statistic'] for r in top_results]
        colors = ['red' if r['p_value'] < 0.05 else 'steelblue' for r in top_results]
        
        bars = ax.barh(features[::-1], ks_stats[::-1], color=colors[::-1])
        
        ax.set_xlabel('KS Statistic', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Kolmogorov-Smirnov Statistics by Feature\n(Red = Significant shift, p < 0.05)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {width:.3f}',
                   ha='left', va='center', fontsize=9)
        
        return fig

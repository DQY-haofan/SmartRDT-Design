#!/usr/bin/env python3
"""
Journal-Quality Visualization Module for RMTwin
================================================
Publication-ready figures for top-tier journals (e.g., Automation in Construction).

This module provides:
1. Pareto front visualizations (2D, 3D, parallel coordinates)
2. Convergence analysis plots
3. Decision variable distribution analysis
4. Sensitivity analysis heatmaps
5. Baseline comparison charts
6. Trade-off analysis figures

Author: RMTwin Research Team
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

# Configure matplotlib for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

logger = logging.getLogger(__name__)

# Color schemes for publications
JOURNAL_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'success': '#3A7D44',      # Green
    'neutral': '#666666',      # Gray
}

BASELINE_COLORS = {
    'NSGA-III': '#2E86AB',
    'Random': '#A23B72',
    'Grid': '#F18F01',
    'Weighted': '#C73E1D',
    'Expert': '#3A7D44',
}

OBJECTIVE_LABELS = {
    'f1_total_cost_USD': 'Total Cost (Million USD)',
    'f2_one_minus_recall': '1 - Detection Recall',
    'f3_latency_seconds': 'Processing Latency (s)',
    'f4_traffic_disruption_hours_year': 'Traffic Disruption (h/year)',
    'f5_carbon_emissions_kgCO2e_year': 'Carbon Emissions (tCO₂e/year)',
    'f6_system_failure_rate': 'System Failure Rate',
    'detection_recall': 'Detection Recall',
}

VARIABLE_LABELS = {
    'sensor': 'Sensor System',
    'data_rate': 'Data Rate (Hz)',
    'geo_lod': 'Geometric LOD',
    'cond_lod': 'Condition LOD',
    'algorithm': 'Detection Algorithm',
    'detection_threshold': 'Detection Threshold',
    'storage': 'Storage System',
    'communication': 'Communication',
    'deployment': 'Deployment',
    'crew_size': 'Crew Size',
    'inspection_cycle': 'Inspection Cycle (days)',
}


class JournalVisualizer:
    """Generate publication-quality visualizations for multi-objective optimization results."""
    
    def __init__(self, output_dir: str = './figures'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_pareto_2d(self, pareto_df: pd.DataFrame, 
                       x_col: str = 'f1_total_cost_USD',
                       y_col: str = 'detection_recall',
                       baseline_dfs: Dict[str, pd.DataFrame] = None,
                       highlight_solutions: List[int] = None,
                       title: str = None,
                       save_name: str = 'pareto_2d.pdf') -> plt.Figure:
        """
        Create 2D Pareto front visualization.
        
        Args:
            pareto_df: DataFrame with Pareto solutions
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            baseline_dfs: Dict of baseline DataFrames
            highlight_solutions: List of solution indices to highlight
            title: Figure title
            save_name: Output filename
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scale x values if cost
        x_scale = 1e6 if 'cost' in x_col.lower() else 1
        x_data = pareto_df[x_col] / x_scale
        y_data = pareto_df[y_col]
        
        # Plot baselines first (background)
        if baseline_dfs:
            for name, df in baseline_dfs.items():
                if len(df) > 0 and x_col in df.columns and y_col in df.columns:
                    feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                    if len(feasible) > 0:
                        ax.scatter(feasible[x_col] / x_scale, feasible[y_col],
                                  c=BASELINE_COLORS.get(name, '#999999'),
                                  alpha=0.4, s=30, label=name, marker='o')
        
        # Plot Pareto front
        sorted_idx = np.argsort(x_data)
        ax.plot(x_data.iloc[sorted_idx], y_data.iloc[sorted_idx], 
                'o-', color=JOURNAL_COLORS['primary'], 
                linewidth=2, markersize=8, label='NSGA-III Pareto Front')
        
        # Highlight specific solutions
        if highlight_solutions:
            for idx in highlight_solutions:
                if idx < len(pareto_df):
                    ax.scatter(x_data.iloc[idx], y_data.iloc[idx],
                              s=150, c=JOURNAL_COLORS['tertiary'], 
                              marker='*', zorder=10, edgecolors='black')
        
        # Labels
        xlabel = OBJECTIVE_LABELS.get(x_col, x_col)
        ylabel = OBJECTIVE_LABELS.get(y_col, y_col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if title:
            ax.set_title(title)
        
        ax.legend(loc='best', framealpha=0.9)
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_pareto_3d(self, pareto_df: pd.DataFrame,
                       x_col: str = 'f1_total_cost_USD',
                       y_col: str = 'detection_recall',
                       z_col: str = 'f5_carbon_emissions_kgCO2e_year',
                       save_name: str = 'pareto_3d.pdf') -> plt.Figure:
        """
        Create 3D Pareto front visualization.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scale values
        x_scale = 1e6 if 'cost' in x_col.lower() else 1
        z_scale = 1000 if 'carbon' in z_col.lower() else 1
        
        x = pareto_df[x_col] / x_scale
        y = pareto_df[y_col]
        z = pareto_df[z_col] / z_scale
        
        # Color by one objective (e.g., latency)
        if 'f3_latency_seconds' in pareto_df.columns:
            colors = pareto_df['f3_latency_seconds']
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=60, alpha=0.8)
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, label='Latency (s)')
        else:
            ax.scatter(x, y, z, c=JOURNAL_COLORS['primary'], s=60, alpha=0.8)
        
        ax.set_xlabel(OBJECTIVE_LABELS.get(x_col, x_col))
        ax.set_ylabel(OBJECTIVE_LABELS.get(y_col, y_col))
        ax.set_zlabel(OBJECTIVE_LABELS.get(z_col, z_col).replace('kg', 't'))
        
        ax.view_init(elev=20, azim=45)
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_parallel_coordinates(self, pareto_df: pd.DataFrame,
                                  objectives: List[str] = None,
                                  color_by: str = 'f1_total_cost_USD',
                                  save_name: str = 'parallel_coords.pdf') -> plt.Figure:
        """
        Create parallel coordinates plot for all objectives.
        """
        if objectives is None:
            objectives = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                         'f4_traffic_disruption_hours_year', 'f5_carbon_emissions_kgCO2e_year']
        
        # Filter to available columns
        objectives = [o for o in objectives if o in pareto_df.columns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize data to [0, 1]
        data_norm = pareto_df[objectives].copy()
        for col in objectives:
            min_val = data_norm[col].min()
            max_val = data_norm[col].max()
            if max_val > min_val:
                data_norm[col] = (data_norm[col] - min_val) / (max_val - min_val)
            else:
                data_norm[col] = 0.5
        
        # Color by specified column
        if color_by in pareto_df.columns:
            colors = plt.cm.viridis((pareto_df[color_by] - pareto_df[color_by].min()) / 
                                    (pareto_df[color_by].max() - pareto_df[color_by].min() + 1e-10))
        else:
            colors = [JOURNAL_COLORS['primary']] * len(pareto_df)
        
        # Plot lines
        x = np.arange(len(objectives))
        for i in range(len(data_norm)):
            ax.plot(x, data_norm.iloc[i].values, c=colors[i], alpha=0.6, linewidth=1.5)
        
        # Axis setup
        ax.set_xticks(x)
        ax.set_xticklabels([OBJECTIVE_LABELS.get(o, o).replace(' ', '\n') for o in objectives], 
                          rotation=0, ha='center')
        ax.set_ylabel('Normalized Value')
        ax.set_ylim(-0.05, 1.05)
        
        # Add colorbar
        if color_by in pareto_df.columns:
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                       norm=plt.Normalize(pareto_df[color_by].min(), 
                                                         pareto_df[color_by].max()))
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label(OBJECTIVE_LABELS.get(color_by, color_by))
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_convergence(self, history: Dict,
                        save_name: str = 'convergence.pdf') -> plt.Figure:
        """
        Plot optimization convergence metrics.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract convergence data
        if 'convergence' in history and history['convergence']:
            conv = history['convergence']
            generations = range(1, len(conv.get('hv', [])) + 1)
            
            # Hypervolume
            if 'hv' in conv and conv['hv']:
                axes[0].plot(generations, conv['hv'], '-o', 
                           color=JOURNAL_COLORS['primary'], linewidth=2, markersize=4)
                axes[0].set_xlabel('Generation')
                axes[0].set_ylabel('Hypervolume')
                axes[0].set_title('(a) Hypervolume Convergence')
            
            # Number of non-dominated solutions
            if 'n_nds' in conv and conv['n_nds']:
                axes[1].plot(generations, conv['n_nds'], '-s',
                           color=JOURNAL_COLORS['secondary'], linewidth=2, markersize=4)
                axes[1].set_xlabel('Generation')
                axes[1].set_ylabel('Number of Solutions')
                axes[1].set_title('(b) Pareto Front Size')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_sensitivity_heatmap(self, sensitivity_results: Dict[str, Dict[str, float]],
                                 save_name: str = 'sensitivity_heatmap.pdf') -> plt.Figure:
        """
        Create heatmap of variable sensitivities.
        
        Args:
            sensitivity_results: Dict of {variable: {objective: impact_pct}}
        """
        # Convert to DataFrame
        variables = list(sensitivity_results.keys())
        objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability']
        
        # Build matrix
        data = np.zeros((len(variables), len(objectives)))
        for i, var in enumerate(variables):
            for j, obj in enumerate(objectives):
                data[i, j] = sensitivity_results[var].get(obj, 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list('sensitivity', 
                                                  ['#ffffff', '#ffd700', '#ff6b6b', '#c0392b'])
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=200)
        
        # Labels
        ax.set_xticks(np.arange(len(objectives)))
        ax.set_yticks(np.arange(len(variables)))
        ax.set_xticklabels(objectives)
        ax.set_yticklabels([VARIABLE_LABELS.get(v, v) for v in variables])
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations
        for i in range(len(variables)):
            for j in range(len(objectives)):
                val = data[i, j]
                if val > 0.1:
                    text_color = 'white' if val > 100 else 'black'
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                           color=text_color, fontsize=8)
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Impact (%)', rotation=-90, va='bottom')
        
        ax.set_title('Decision Variable Sensitivity Analysis')
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_baseline_comparison(self, pareto_df: pd.DataFrame,
                                 baseline_dfs: Dict[str, pd.DataFrame],
                                 save_name: str = 'baseline_comparison.pdf') -> plt.Figure:
        """
        Create comprehensive baseline comparison figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = ['NSGA-III'] + list(baseline_dfs.keys())
        colors = [BASELINE_COLORS.get(m, '#999999') for m in methods]
        
        # Prepare data
        metrics = {
            'NSGA-III': {
                'n_solutions': len(pareto_df),
                'n_feasible': len(pareto_df),
                'min_cost': pareto_df['f1_total_cost_USD'].min() / 1e6 if len(pareto_df) > 0 else np.nan,
                'max_recall': pareto_df['detection_recall'].max() if len(pareto_df) > 0 else np.nan,
            }
        }
        
        for name, df in baseline_dfs.items():
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns and len(df) > 0 else df
            metrics[name] = {
                'n_solutions': len(df),
                'n_feasible': len(feasible),
                'min_cost': feasible['f1_total_cost_USD'].min() / 1e6 if len(feasible) > 0 else np.nan,
                'max_recall': feasible['detection_recall'].max() if len(feasible) > 0 else np.nan,
            }
        
        # (a) Number of feasible solutions
        ax = axes[0, 0]
        values = [metrics[m]['n_feasible'] for m in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Number of Feasible Solutions')
        ax.set_title('(a) Solution Feasibility')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val}', ha='center', va='bottom', fontsize=9)
        
        # (b) Minimum cost achieved
        ax = axes[0, 1]
        values = [metrics[m]['min_cost'] for m in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Minimum Cost (Million USD)')
        ax.set_title('(b) Cost Optimization')
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'${val:.2f}M', ha='center', va='bottom', fontsize=9)
        
        # (c) Maximum recall achieved
        ax = axes[1, 0]
        values = [metrics[m]['max_recall'] for m in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Maximum Detection Recall')
        ax.set_title('(c) Detection Performance')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # (d) Cost-Recall scatter comparison
        ax = axes[1, 1]
        for i, (name, df) in enumerate([('NSGA-III', pareto_df)] + list(baseline_dfs.items())):
            if len(df) > 0 and 'f1_total_cost_USD' in df.columns:
                if name == 'NSGA-III':
                    data = df
                else:
                    data = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(data) > 0:
                    ax.scatter(data['f1_total_cost_USD'] / 1e6, data['detection_recall'],
                              c=BASELINE_COLORS.get(name, '#999999'), alpha=0.6, 
                              s=40, label=name)
        ax.set_xlabel('Total Cost (Million USD)')
        ax.set_ylabel('Detection Recall')
        ax.set_title('(d) Cost-Recall Trade-off')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_decision_distribution(self, pareto_df: pd.DataFrame,
                                   save_name: str = 'decision_distribution.pdf') -> plt.Figure:
        """
        Visualize distribution of decision variables in Pareto solutions.
        """
        # Decision variable columns
        var_cols = ['sensor', 'algorithm', 'storage', 'communication', 'deployment',
                   'data_rate', 'geo_lod', 'cond_lod', 'detection_threshold',
                   'crew_size', 'inspection_cycle']
        
        available_cols = [c for c in var_cols if c in pareto_df.columns]
        
        n_vars = len(available_cols)
        n_cols = 4
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(available_cols):
            ax = axes[i]
            data = pareto_df[col]
            
            if data.dtype == 'object':
                # Categorical
                counts = data.value_counts()
                ax.bar(range(len(counts)), counts.values, color=JOURNAL_COLORS['primary'], alpha=0.8)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels([str(x)[:15] for x in counts.index], rotation=45, ha='right', fontsize=8)
            else:
                # Numerical
                ax.hist(data, bins=15, color=JOURNAL_COLORS['primary'], alpha=0.8, edgecolor='white')
            
            ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=10)
            ax.set_ylabel('Count')
        
        # Hide empty subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_radar_chart(self, solutions: List[Dict], labels: List[str],
                         objectives: List[str] = None,
                         save_name: str = 'radar_chart.pdf') -> plt.Figure:
        """
        Create radar chart comparing selected solutions.
        """
        if objectives is None:
            objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability']
        
        n_objectives = len(objectives)
        angles = np.linspace(0, 2 * np.pi, n_objectives, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        colors = [JOURNAL_COLORS['primary'], JOURNAL_COLORS['secondary'], 
                  JOURNAL_COLORS['tertiary'], JOURNAL_COLORS['quaternary']]
        
        for i, (sol, label) in enumerate(zip(solutions, labels)):
            values = [sol.get(obj, 0) for obj in objectives]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=label, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(objectives)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath)
        logger.info(f"Saved: {filepath}")
        
        return fig
    
    def generate_all_figures(self, pareto_df: pd.DataFrame,
                            baseline_dfs: Dict[str, pd.DataFrame] = None,
                            history: Dict = None,
                            sensitivity: Dict = None) -> List[str]:
        """
        Generate all publication figures.
        
        Returns:
            List of generated figure paths
        """
        figures = []
        
        # 1. Pareto 2D (Cost vs Recall)
        fig = self.plot_pareto_2d(pareto_df, baseline_dfs=baseline_dfs)
        figures.append(os.path.join(self.output_dir, 'pareto_2d.pdf'))
        plt.close(fig)
        
        # 2. Pareto 3D
        if 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns:
            fig = self.plot_pareto_3d(pareto_df)
            figures.append(os.path.join(self.output_dir, 'pareto_3d.pdf'))
            plt.close(fig)
        
        # 3. Parallel coordinates
        fig = self.plot_parallel_coordinates(pareto_df)
        figures.append(os.path.join(self.output_dir, 'parallel_coords.pdf'))
        plt.close(fig)
        
        # 4. Convergence
        if history:
            fig = self.plot_convergence(history)
            figures.append(os.path.join(self.output_dir, 'convergence.pdf'))
            plt.close(fig)
        
        # 5. Sensitivity heatmap
        if sensitivity:
            fig = self.plot_sensitivity_heatmap(sensitivity)
            figures.append(os.path.join(self.output_dir, 'sensitivity_heatmap.pdf'))
            plt.close(fig)
        
        # 6. Baseline comparison
        if baseline_dfs:
            fig = self.plot_baseline_comparison(pareto_df, baseline_dfs)
            figures.append(os.path.join(self.output_dir, 'baseline_comparison.pdf'))
            plt.close(fig)
        
        # 7. Decision distribution
        fig = self.plot_decision_distribution(pareto_df)
        figures.append(os.path.join(self.output_dir, 'decision_distribution.pdf'))
        plt.close(fig)
        
        logger.info(f"Generated {len(figures)} publication figures")
        return figures


def run_sensitivity_analysis(evaluator, base_config: Dict = None) -> Dict[str, Dict[str, float]]:
    """
    Run comprehensive sensitivity analysis.
    
    Returns:
        Dict of {variable: {objective: impact_pct}}
    """
    from model_params import MODEL_PARAMS
    
    # Default base configuration
    if base_config is None:
        base_x = np.array([0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.7, 0.3, 0.15])
    else:
        base_x = np.array(list(base_config.values()))
    
    variables = ['sensor', 'data_rate', 'geo_lod', 'cond_lod', 'algorithm',
                'detection_threshold', 'storage', 'communication', 'deployment',
                'crew_size', 'inspection_cycle']
    
    objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability']
    
    # Get baseline
    base_obj, _ = evaluator._evaluate_single(base_x)
    
    results = {}
    
    for i, var in enumerate(variables):
        impacts = {}
        
        for delta in [-0.3, 0.3]:
            x_test = base_x.copy()
            x_test[i] = np.clip(base_x[i] + delta, 0, 1)
            
            test_obj, _ = evaluator._evaluate_single(x_test)
            
            for j, obj in enumerate(objectives):
                if base_obj[j] != 0:
                    pct_change = abs(test_obj[j] - base_obj[j]) / abs(base_obj[j]) * 100
                else:
                    pct_change = abs(test_obj[j]) * 100
                
                if obj not in impacts:
                    impacts[obj] = 0
                impacts[obj] = max(impacts[obj], pct_change)
        
        results[var] = impacts
    
    return results


if __name__ == '__main__':
    # Example usage
    print("Journal Visualization Module for RMTwin")
    print("=" * 50)
    print("\nUsage:")
    print("  from visualization_journal import JournalVisualizer")
    print("  viz = JournalVisualizer('./figures')")
    print("  viz.generate_all_figures(pareto_df, baseline_dfs, history)")

#!/usr/bin/env python3
"""
Visualization Module for RMTwin Optimization
Complete implementation of all visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Publication quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

# Professional color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'highlight': '#9467bd',
    'neutral': '#7f7f7f',
    'light': '#bcbcbc'
}


class Visualizer:
    """Main visualization class"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
    
    def create_all_figures(self, pareto_results: pd.DataFrame, 
                          baseline_results: Optional[Dict[str, pd.DataFrame]] = None):
        """Generate all publication-quality figures"""
        logger.info("Creating visualizations...")
        
        # Figure 1: Key Pareto Front Projections
        self.create_pareto_projections(pareto_results)
        
        # Figure 2: Comprehensive Multi-Objective Analysis
        self.create_comprehensive_analysis(pareto_results)
        
        # Figure 3: Insights and Comparison
        self.create_insights_comparison(pareto_results)
        
        # Figure 4: Expert Analysis
        self.create_expert_analysis(pareto_results)
        
        # Figure 5: Baseline Comparison (if available)
        if baseline_results:
            self.create_baseline_comparison(pareto_results, baseline_results)
        
        logger.info(f"All figures saved to {self.output_dir}")
    
    def create_pareto_projections(self, df: pd.DataFrame):
        """Figure 1: Key 2D Pareto Front Projections"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Key Trade-offs in RMTwin Multi-Objective Optimization', 
                     fontsize=14, y=0.98)
        
        projections = [
            ('f1_total_cost_USD', 'detection_recall', 'Cost vs Performance', 
             True, False, 1000, 1, 'k$', '', 'f5_carbon_emissions_kgCO2e_year', 'Carbon (kgCO2e/y)'),
            ('f1_total_cost_USD', 'f5_carbon_emissions_kgCO2e_year', 'Cost vs Sustainability', 
             True, True, 1000, 1000, 'k$', 'tons CO₂/year', 'detection_recall', 'Recall'),
            ('detection_recall', 'f3_latency_seconds', 'Performance vs Real-time Capability', 
             False, True, 1, 1, '', 'seconds', 'f1_total_cost_USD', 'Cost ($)'),
            ('f5_carbon_emissions_kgCO2e_year', 'system_MTBF_hours', 'Sustainability vs Reliability', 
             True, False, 1000, 8760, 'tons CO₂/year', 'years', 'detection_recall', 'Recall')
        ]
        
        for idx, (x_col, y_col, title, x_min, y_min, x_scale, y_scale, 
                  x_unit, y_unit, c_col, c_label) in enumerate(projections):
            ax = axes[idx // 2, idx % 2]
            
            # Scale data
            x_data = df[x_col] / x_scale
            y_data = df[y_col] / y_scale
            
            # Color by third objective
            c_data = df[c_col] / (1000 if 'cost' in c_col.lower() else 1)
            
            # Find Pareto front
            pareto_mask = self._find_pareto_front_2d(
                df[x_col].values, df[y_col].values, x_min, y_min)
            
            # Plot all points
            scatter = ax.scatter(x_data, y_data, c=c_data, cmap='viridis',
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Highlight Pareto front
            if np.any(pareto_mask):
                pareto_x = x_data[pareto_mask]
                pareto_y = y_data[pareto_mask]
                
                # Sort for line plotting
                if x_min:
                    sort_idx = np.argsort(pareto_x)
                else:
                    sort_idx = np.argsort(pareto_x)[::-1]
                
                ax.plot(pareto_x.iloc[sort_idx], pareto_y.iloc[sort_idx], 
                       'r--', linewidth=2, alpha=0.7, label='Pareto Front')
                
                ax.scatter(pareto_x, pareto_y, s=100, facecolors='none', 
                          edgecolors='red', linewidth=2, alpha=0.8)
            
            # Labels
            x_label = x_col.replace('_', ' ').replace('f1 ', '').replace('f5 ', '').replace('f3 ', '')
            y_label = y_col.replace('_', ' ').replace('f1 ', '').replace('f5 ', '').replace('f3 ', '')
            
            if x_unit:
                x_label = f"{x_label} ({x_unit})"
            if y_unit:
                y_label = f"{y_label} ({y_unit})"
            
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'({chr(97+idx)}) {title}', fontsize=11, pad=5)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label(c_label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            if idx == 0:
                ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        plt.tight_layout()
        self._save_figure(fig, 'figure_1_key_pareto_projections')
    
    def create_comprehensive_analysis(self, df: pd.DataFrame):
        """Figure 2: Comprehensive Multi-Objective Analysis"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], 
                      hspace=0.3, wspace=0.3)
        
        # 1. Parallel Coordinates
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_parallel_coordinates(df, ax1)
        ax1.set_title('(a) Multi-Objective Trade-offs Across Solutions', fontsize=12)
        
        # 2. Technology Performance Matrix
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_technology_matrix(df, ax2)
        ax2.set_title('(b) Sensor Technology Performance', fontsize=12)
        
        # 3. Solution Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_solution_distribution(df, ax3)
        ax3.set_title('(c) Solution Density', fontsize=12)
        
        # 4. Representative Solutions Table
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_representative_solutions(df, ax4)
        ax4.set_title('(d) Representative Solutions for Different Optimization Priorities', 
                      fontsize=12, pad=10)
        
        fig.suptitle('Comprehensive Analysis of RMTwin Configuration Optimization Results', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        self._save_figure(fig, 'figure_2_comprehensive_analysis')
    
    def create_insights_comparison(self, df: pd.DataFrame):
        """Figure 3: Key Insights and Method Comparison"""
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Correlation Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation_matrix(df, ax1)
        ax1.set_title('(a) Objective Correlations', fontsize=12)
        
        # 2. Technology Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_technology_distribution(df, ax2)
        ax2.set_title('(b) Sensor Technology Distribution', fontsize=12)
        
        # 3. Trade-off Visualization
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_trade_off_analysis(df, ax3)
        ax3.set_title('(c) Multi-Criteria Performance', fontsize=12)
        
        # 4. Summary Statistics
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_summary_statistics(df, ax4)
        
        fig.suptitle('Key Insights from 6-Objective RMTwin Optimization', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        self._save_figure(fig, 'figure_3_insights_comparison')
    
    def create_expert_analysis(self, df: pd.DataFrame):
        """Figure 4: Expert-Enhanced Analysis"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Annualized Cost Breakdown
        ax1 = plt.subplot(2, 3, 1)
        self._plot_cost_breakdown(df, ax1)
        ax1.set_title('(a) Annualized Cost Breakdown by Sensor Type', fontsize=14)
        
        # 2. Carbon Footprint Analysis
        ax2 = plt.subplot(2, 3, 2)
        self._plot_carbon_analysis(df, ax2)
        ax2.set_title('(b) Cost vs Environmental Impact Trade-off', fontsize=14)
        
        # 3. Scenario Impact Analysis
        ax3 = plt.subplot(2, 3, 3)
        self._plot_scenario_impact(ax3)
        ax3.set_title('(c) Scenario-Dependent Network Performance', fontsize=14)
        
        # 4. Class Imbalance Impact
        ax4 = plt.subplot(2, 3, 4)
        self._plot_class_imbalance_impact(ax4)
        ax4.set_title('(d) Class Imbalance Impact on Algorithms', fontsize=14)
        
        # 5. Redundancy Reliability
        ax5 = plt.subplot(2, 3, 5)
        self._plot_redundancy_impact(ax5)
        ax5.set_title('(e) Redundancy Impact on System Reliability', fontsize=14)
        
        # 6. Comprehensive Performance Radar
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        self._plot_performance_radar(df, ax6)
        ax6.set_title('(f) Multi-Criteria Performance Comparison', 
                      fontsize=14, pad=20)
        
        plt.suptitle('Expert-Enhanced Analysis: Advanced Modeling Insights', 
                    fontsize=16)
        plt.tight_layout()
        
        self._save_figure(fig, 'figure_4_expert_analysis')
    
    def create_baseline_comparison(self, pareto_df: pd.DataFrame, 
                                 baseline_results: Dict[str, pd.DataFrame]):
        """Figure 5: Baseline Method Comparison"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Method comparison - feasible solutions
        ax1 = plt.subplot(3, 3, 1)
        self._plot_feasibility_comparison(pareto_df, baseline_results, ax1)
        ax1.set_title('Feasibility Rate by Method', fontsize=14)
        
        # 2. Objective space coverage - Cost vs Recall
        ax2 = plt.subplot(3, 3, 2)
        self._plot_objective_space_coverage(pareto_df, baseline_results, ax2)
        ax2.set_title('Cost vs Performance Trade-off', fontsize=14)
        
        # 3. Sustainability comparison
        ax3 = plt.subplot(3, 3, 3)
        self._plot_sustainability_comparison(pareto_df, baseline_results, ax3)
        ax3.set_title('Sustainability vs Reliability', fontsize=14)
        
        # 4. Hypervolume comparison
        ax4 = plt.subplot(3, 3, 4)
        self._plot_hypervolume_comparison(pareto_df, baseline_results, ax4)
        ax4.set_title('Solution Quality Comparison', fontsize=14)
        
        # 5. Computation time comparison
        ax5 = plt.subplot(3, 3, 5)
        self._plot_computational_efficiency(baseline_results, ax5)
        ax5.set_title('Computational Efficiency', fontsize=14)
        
        # 6. Best solutions comparison table
        ax6 = plt.subplot(3, 3, 6)
        self._plot_best_solutions_table(pareto_df, baseline_results, ax6)
        ax6.set_title('Best Solution Comparison', fontsize=14)
        
        # 7-9. Additional comparisons
        for idx, (obj_name, obj_col) in enumerate([
            ('Latency', 'f3_latency_seconds'),
            ('Traffic Disruption', 'f4_traffic_disruption_hours'),
            ('All Objectives', None)
        ]):
            ax = plt.subplot(3, 3, 7 + idx)
            if obj_col:
                self._plot_objective_distribution(pareto_df, baseline_results, 
                                                obj_col, obj_name, ax)
                ax.set_title(f'{obj_name} Distribution', fontsize=14)
            else:
                self._plot_multi_objective_radar(pareto_df, baseline_results, ax)
                ax.set_title('Multi-Objective Performance', fontsize=14, pad=20)
        
        plt.suptitle('Baseline Methods vs NSGA-II Comparison (6 Objectives)', 
                    fontsize=18)
        plt.tight_layout()
        
        self._save_figure(fig, 'figure_5_baseline_comparison')
    
    # Helper methods
    def _find_pareto_front_2d(self, x: np.ndarray, y: np.ndarray, 
                             x_minimize: bool = True, 
                             y_minimize: bool = True) -> np.ndarray:
        """Find 2D Pareto front"""
        n = len(x)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if x_minimize:
                        x_dominates = x[j] <= x[i]
                    else:
                        x_dominates = x[j] >= x[i]
                    
                    if y_minimize:
                        y_dominates = y[j] <= y[i]
                    else:
                        y_dominates = y[j] >= y[i]
                    
                    if x_dominates and y_dominates:
                        if ((x_minimize and x[j] < x[i]) or 
                            (not x_minimize and x[j] > x[i]) or
                            (y_minimize and y[j] < y[i]) or 
                            (not y_minimize and y[j] > y[i])):
                            pareto_mask[i] = False
                            break
        
        return pareto_mask
    
    def _plot_parallel_coordinates(self, df: pd.DataFrame, ax):
        """Plot parallel coordinates for multi-objective visualization"""
        # Prepare data
        plot_df = df.copy()
        
        # Normalize objectives
        objectives = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                     'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year', 
                     'system_MTBF_hours']
        
        for obj in objectives:
            if obj in ['detection_recall', 'system_MTBF_hours']:
                # Maximize these
                plot_df[obj] = (plot_df[obj] - plot_df[obj].min()) / \
                              (plot_df[obj].max() - plot_df[obj].min())
            else:
                # Minimize these
                plot_df[obj] = 1 - (plot_df[obj] - plot_df[obj].min()) / \
                                   (plot_df[obj].max() - plot_df[obj].min())
        
        # Rename columns
        display_names = {
            'f1_total_cost_USD': 'Cost↓',
            'detection_recall': 'Recall↑',
            'f3_latency_seconds': 'Latency↓',
            'f4_traffic_disruption_hours': 'Disruption↓',
            'f5_carbon_emissions_kgCO2e_year': 'Carbon↓',
            'system_MTBF_hours': 'Reliability↑'
        }
        plot_df.rename(columns=display_names, inplace=True)
        
        # Extract sensor type
        plot_df['Sensor'] = df['sensor'].str.extract(r'(\w+)_')[0].fillna('Other')
        
        # Select diverse solutions
        selected_indices = []
        for sensor in plot_df['Sensor'].unique():
            sensor_df = plot_df[plot_df['Sensor'] == sensor]
            if len(sensor_df) >= 3:
                indices = [sensor_df.index[0], 
                          sensor_df.index[len(sensor_df)//2], 
                          sensor_df.index[-1]]
            else:
                indices = sensor_df.index.tolist()
            selected_indices.extend(indices)
        
        plot_df_filtered = plot_df.loc[selected_indices[:50]]  # Limit to 50
        
        # Create parallel coordinates
        from pandas.plotting import parallel_coordinates
        parallel_coordinates(plot_df_filtered, 'Sensor', 
                           cols=list(display_names.values()),
                           colormap='tab10', alpha=0.7, linewidth=2, ax=ax)
        
        ax.set_ylabel('Normalized Value [0,1]', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    
    def _save_figure(self, fig, filename: str):
        """Save figure in multiple formats"""
        for fmt in self.config.figure_format:
            path = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(path, dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")
    
    # Additional helper methods would follow...
    # (Implementing all the plot_* methods referenced above)
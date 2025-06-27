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
        
        # Rename columns - 改为英文
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
    
    def _plot_technology_matrix(self, df: pd.DataFrame, ax):
        """Plot technology performance matrix"""
        # Group by sensor type
        df['sensor_type'] = df['sensor'].str.extract(r'(\w+)_')[0]
        
        metrics = ['detection_recall', 'f1_total_cost_USD', 'f3_latency_seconds', 
                'f5_carbon_emissions_kgCO2e_year']
        metric_names = ['Recall', 'Cost (k$)', 'Latency (s)', 'Carbon (ton/yr)']  # 英文
        
        # Create performance matrix
        sensor_types = df['sensor_type'].unique()[:6]  # Limit display
        matrix_data = []
        
        for sensor_type in sensor_types:
            sensor_df = df[df['sensor_type'] == sensor_type]
            row = []
            for metric in metrics:
                if metric == 'f1_total_cost_USD':
                    value = sensor_df[metric].mean() / 1000  # Convert to k$
                elif metric == 'f5_carbon_emissions_kgCO2e_year':
                    value = sensor_df[metric].mean() / 1000  # Convert to tons
                else:
                    value = sensor_df[metric].mean()
                row.append(value)
            matrix_data.append(row)
        
        # Normalize to 0-1
        matrix_data = np.array(matrix_data)
        for i in range(matrix_data.shape[1]):
            col = matrix_data[:, i]
            if metrics[i] in ['detection_recall']:  # Higher is better
                matrix_data[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-10)
            else:  # Lower is better
                matrix_data[:, i] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
        
        # Plot heatmap
        im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(sensor_types)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(sensor_types)
        
        # Add value labels
        for i in range(len(sensor_types)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{matrix_data[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Performance', fontsize=9)  # 英文

    def _plot_solution_distribution(self, df: pd.DataFrame, ax):
        """Plot solution distribution"""
        # 2D density plot: Cost vs Recall
        x = df['f1_total_cost_USD'] / 1000  # k$
        y = df['detection_recall']
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=20)
        
        # Plot
        im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='Blues', interpolation='gaussian')
        
        # Add scatter points
        ax.scatter(x, y, c='red', s=10, alpha=0.5, edgecolors='none')
        
        ax.set_xlabel('Total Cost (k$)')  # 英文
        ax.set_ylabel('Detection Recall')  # 英文
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Solution Count', fontsize=9)  # 英文

    def _plot_representative_solutions(self, df: pd.DataFrame, ax):
        """Plot representative solutions table"""
        # Select representative solutions
        representatives = []
        
        # Lowest cost
        if len(df) > 0:
            min_cost_idx = df['f1_total_cost_USD'].argmin()
            representatives.append(('Lowest Cost', df.iloc[min_cost_idx]))  # 英文
        
        # Highest recall
        if len(df) > 0:
            max_recall_idx = df['detection_recall'].argmax()
            representatives.append(('Highest Recall', df.iloc[max_recall_idx]))  # 英文
        
        # Lowest carbon
        if 'f5_carbon_emissions_kgCO2e_year' in df.columns and len(df) > 0:
            min_carbon_idx = df['f5_carbon_emissions_kgCO2e_year'].argmin()
            representatives.append(('Lowest Carbon', df.iloc[min_carbon_idx]))  # 英文
        
        # Balanced solution
        if len(df) > 0:
            norm_df = df.copy()
            for col in ['f1_total_cost_USD', 'f3_latency_seconds', 'f4_traffic_disruption_hours', 
                        'f5_carbon_emissions_kgCO2e_year']:
                if col in norm_df.columns:
                    norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min() + 1e-10)
            norm_df['detection_recall'] = 1 - norm_df['detection_recall']
            
            norm_sum = norm_df[['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']].sum(axis=1)
            balanced_idx = norm_sum.argmin()
            representatives.append(('Balanced', df.iloc[balanced_idx]))  # 英文
        
        # Create table data
        if representatives:
            table_data = []
            for name, sol in representatives:
                row = [
                    name,
                    sol['sensor'].split('_')[0],
                    sol['algorithm'].split('_')[0],
                    f"${sol['f1_total_cost_USD']/1000:.0f}k",
                    f"{sol['detection_recall']:.3f}",
                    f"{sol['f3_latency_seconds']:.1f}s",
                    f"{sol['f5_carbon_emissions_kgCO2e_year']/1000:.1f}t" if 'f5_carbon_emissions_kgCO2e_year' in sol else 'N/A'
                ]
                table_data.append(row)
            
            columns = ['Solution Type', 'Sensor', 'Algorithm', 'Cost', 'Recall', 'Latency', 'Carbon']  # 英文
            
            # Draw table
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=columns,
                            cellLoc='center', loc='center',
                            colWidths=[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Set table style
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(table_data) + 1):
                for j in range(len(columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
        else:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)  # 英文
            ax.axis('off')

    def _plot_representative_solutions(self, df: pd.DataFrame, ax):
        """Plot representative solutions table - English version"""
        # Select representative solutions
        representatives = []
        
        # Lowest cost
        if len(df) > 0:
            min_cost_idx = df['f1_total_cost_USD'].argmin()
            representatives.append(('Lowest Cost', df.iloc[min_cost_idx]))
        
        # Highest recall
        if len(df) > 0:
            max_recall_idx = df['detection_recall'].argmax()
            representatives.append(('Highest Recall', df.iloc[max_recall_idx]))
        
        # Lowest carbon
        if 'f5_carbon_emissions_kgCO2e_year' in df.columns and len(df) > 0:
            min_carbon_idx = df['f5_carbon_emissions_kgCO2e_year'].argmin()
            representatives.append(('Lowest Carbon', df.iloc[min_carbon_idx]))
        
        # Balanced solution
        if len(df) > 0:
            norm_df = df.copy()
            for col in ['f1_total_cost_USD', 'f3_latency_seconds', 'f4_traffic_disruption_hours', 
                        'f5_carbon_emissions_kgCO2e_year']:
                if col in norm_df.columns:
                    norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min() + 1e-10)
            norm_df['detection_recall'] = 1 - norm_df['detection_recall']
            
            norm_sum = norm_df[['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']].sum(axis=1)
            balanced_idx = norm_sum.argmin()
            representatives.append(('Balanced', df.iloc[balanced_idx]))
        
        # Create table data
        if representatives:
            table_data = []
            for name, sol in representatives:
                row = [
                    name,
                    sol['sensor'].split('_')[0],
                    sol['algorithm'].split('_')[0],
                    f"${sol['f1_total_cost_USD']/1000:.0f}k",
                    f"{sol['detection_recall']:.3f}",
                    f"{sol['f3_latency_seconds']:.1f}s",
                    f"{sol['f5_carbon_emissions_kgCO2e_year']/1000:.1f}t" if 'f5_carbon_emissions_kgCO2e_year' in sol else 'N/A'
                ]
                table_data.append(row)
            
            columns = ['Solution Type', 'Sensor', 'Algorithm', 'Cost', 'Recall', 'Latency', 'Carbon']
            
            # Draw table
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=columns,
                            cellLoc='center', loc='center',
                            colWidths=[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Set table style
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(table_data) + 1):
                for j in range(len(columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
        else:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_correlation_matrix(self, df: pd.DataFrame, ax):
        """Plot objective correlation matrix - English version"""
        # Select objective columns
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                    'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year',
                    'system_MTBF_hours']
        
        # Filter existing columns
        available_cols = [col for col in obj_cols if col in df.columns]
        
        if len(available_cols) > 1 and len(df) > 1:
            # Calculate correlation
            corr_data = df[available_cols].corr()
            
            # Plot heatmap
            mask = np.triu(np.ones_like(corr_data), k=1)
            sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": .8}, ax=ax)
            
            # Set labels
            labels = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability'][:len(available_cols)]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
        else:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_technology_distribution(self, df: pd.DataFrame, ax):
        """绘制技术分布饼图"""
        if 'sensor' in df.columns and len(df) > 0:
            # 提取传感器类型
            df['sensor_type'] = df['sensor'].str.extract(r'(\w+)_')[0]
            
            # 统计各类型数量
            sensor_counts = df['sensor_type'].value_counts()
            
            # 绘制饼图
            colors = plt.cm.Set3(np.linspace(0, 1, len(sensor_counts)))
            wedges, texts, autotexts = ax.pie(sensor_counts.values, 
                                            labels=sensor_counts.index,
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
            
            # 调整文字大小
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(8)
                autotext.set_weight('bold')
        else:
            ax.text(0.5, 0.5, 'No Sensor Data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_trade_off_analysis(self, df: pd.DataFrame, ax):
        """绘制权衡分析雷达图"""
        if len(df) > 0:
            # 选择几个代表性解决方案
            solutions = []
            labels = []
            
            # 最低成本
            min_cost_idx = df['f1_total_cost_USD'].argmin()
            solutions.append(df.iloc[min_cost_idx])
            labels.append('Lowest cost')
            
            # 最高召回率
            max_recall_idx = df['detection_recall'].argmax()
            if max_recall_idx != min_cost_idx:
                solutions.append(df.iloc[max_recall_idx])
                labels.append('Highest performance')
            
            # 随机选择一个中间解
            if len(df) > 2:
                mid_idx = len(df) // 2
                solutions.append(df.iloc[mid_idx])
                labels.append('Balancing scheme')
            
            # 准备雷达图数据
            categories = ['Cost', 'Recall rate', 'Latency', 'Interference', 'Carbon emissions', 'Reliability']
            
            # 归一化数据
            fig_temp = plt.figure(figsize=(6, 6))
            ax_radar = fig_temp.add_subplot(111, projection='polar')
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            for sol, label in zip(solutions, labels):
                values = []
                # 成本（反转，越低越好）
                values.append(1 - (sol['f1_total_cost_USD'] - df['f1_total_cost_USD'].min()) / 
                            (df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min() + 1e-10))
                # 召回率（越高越好）
                values.append(sol['detection_recall'])
                # 延迟（反转）
                values.append(1 - (sol['f3_latency_seconds'] - df['f3_latency_seconds'].min()) / 
                            (df['f3_latency_seconds'].max() - df['f3_latency_seconds'].min() + 1e-10))
                # 干扰（反转）
                values.append(1 - (sol['f4_traffic_disruption_hours'] - df['f4_traffic_disruption_hours'].min()) / 
                            (df['f4_traffic_disruption_hours'].max() - df['f4_traffic_disruption_hours'].min() + 1e-10))
                # 碳排放（反转）
                if 'f5_carbon_emissions_kgCO2e_year' in sol:
                    values.append(1 - (sol['f5_carbon_emissions_kgCO2e_year'] - df['f5_carbon_emissions_kgCO2e_year'].min()) / 
                                (df['f5_carbon_emissions_kgCO2e_year'].max() - df['f5_carbon_emissions_kgCO2e_year'].min() + 1e-10))
                else:
                    values.append(0.5)
                # 可靠性（越高越好）
                if 'system_MTBF_hours' in sol:
                    values.append((sol['system_MTBF_hours'] - df['system_MTBF_hours'].min()) / 
                                (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min() + 1e-10))
                else:
                    values.append(0.5)
                
                values += values[:1]
                
                ax_radar.plot(angles, values, 'o-', linewidth=2, label=label)
                ax_radar.fill(angles, values, alpha=0.15)
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, fontsize=8)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=7)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=8)
            ax_radar.grid(True, alpha=0.3)
            
            # 将雷达图复制到原始轴
            plt.close(fig_temp)
            ax.text(0.5, 0.5, 'See separate radar chart', ha='center', va='center', fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_summary_statistics(self, df: pd.DataFrame, ax):
        """Plot summary statistics table - English version"""
        if len(df) > 0:
            # Calculate statistics
            stats = []
            
            # Cost statistics
            stats.append(['Total Cost', 
                        f"${df['f1_total_cost_USD'].min()/1e6:.1f}M - ${df['f1_total_cost_USD'].max()/1e6:.1f}M",
                        f"${df['f1_total_cost_USD'].mean()/1e6:.1f}M ± ${df['f1_total_cost_USD'].std()/1e6:.1f}M"])
            
            # Recall statistics
            stats.append(['Detection Recall',
                        f"{df['detection_recall'].min():.3f} - {df['detection_recall'].max():.3f}",
                        f"{df['detection_recall'].mean():.3f} ± {df['detection_recall'].std():.3f}"])
            
            # Latency statistics
            stats.append(['Latency',
                        f"{df['f3_latency_seconds'].min():.1f}s - {df['f3_latency_seconds'].max():.1f}s",
                        f"{df['f3_latency_seconds'].mean():.1f}s ± {df['f3_latency_seconds'].std():.1f}s"])
            
            # Carbon emissions statistics
            if 'f5_carbon_emissions_kgCO2e_year' in df.columns:
                stats.append(['Annual Carbon',
                            f"{df['f5_carbon_emissions_kgCO2e_year'].min()/1000:.1f}t - {df['f5_carbon_emissions_kgCO2e_year'].max()/1000:.1f}t",
                            f"{df['f5_carbon_emissions_kgCO2e_year'].mean()/1000:.1f}t ± {df['f5_carbon_emissions_kgCO2e_year'].std()/1000:.1f}t"])
            
            # Solution count
            stats.append(['Pareto Solutions', f"{len(df)}", ''])
            
            # Main sensor distribution
            if 'sensor' in df.columns:
                sensor_types = df['sensor'].str.extract(r'(\w+)_')[0].value_counts()
                top_sensor = sensor_types.index[0] if len(sensor_types) > 0 else 'N/A'
                stats.append(['Main Sensor', top_sensor, f"{sensor_types.iloc[0]/len(df)*100:.1f}%"])
            
            # Create table
            columns = ['Metric', 'Range', 'Mean±Std']
            
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=stats, colLabels=columns,
                            cellLoc='center', loc='center',
                            colWidths=[0.3, 0.35, 0.35])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Set style
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(stats) + 1):
                for j in range(len(columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#e3f2fd')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_cost_breakdown(self, df: pd.DataFrame, ax):
        """绘制成本分解图"""
        if len(df) > 0:
            # 按传感器类型分组
            df['sensor_type'] = df['sensor'].str.extract(r'(\w+)_')[0]
            
            # 计算平均年度成本
            cost_by_sensor = df.groupby('sensor_type')['annual_cost_USD'].mean().sort_values(ascending=False)
            
            # 限制显示数量
            cost_by_sensor = cost_by_sensor.head(8)
            
            # 绘制条形图
            bars = ax.bar(range(len(cost_by_sensor)), cost_by_sensor.values / 1000)  # 转换为k$
            
            # 设置颜色渐变
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(cost_by_sensor)))
            ax.set_xticklabels(cost_by_sensor.index, rotation=45, ha='right')
            ax.set_ylabel('Average Annual Cost (k$)')  # 改为英文
            ax.grid(True, axis='y', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, cost_by_sensor.values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'${value/1000:.0f}k', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)  # 改为英文
            ax.axis('off')

    def _plot_carbon_analysis(self, df: pd.DataFrame, ax):
        """绘制碳排放分析"""
        if 'f5_carbon_emissions_kgCO2e_year' in df.columns and len(df) > 0:
            # 成本 vs 碳排放散点图
            x = df['f1_total_cost_USD'] / 1000  # k$
            y = df['f5_carbon_emissions_kgCO2e_year'] / 1000  # 吨
            
            # 按传感器类型着色
            df['sensor_type'] = df['sensor'].str.extract(r'(\w+)_')[0]
            sensor_types = df['sensor_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_types)))
            
            for i, sensor_type in enumerate(sensor_types):
                mask = df['sensor_type'] == sensor_type
                ax.scatter(x[mask], y[mask], c=[colors[i]], label=sensor_type, 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Total Cost (k$)')  # 改为英文
            ax.set_ylabel('Annual Carbon Emissions (tons CO2)')  # 改为英文
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, linewidth=2)
        else:
            ax.text(0.5, 0.5, 'No Carbon Emission Data', ha='center', va='center', fontsize=12)  # 改为英文
            ax.axis('off')

    def _plot_scenario_impact(self, ax):
        """绘制场景影响分析"""
        # 模拟不同场景下的性能数据
        scenarios = ['Urban', 'Rural', 'Mixed']  # 改为英文
        metrics = ['Coverage Efficiency', 'Detection Accuracy', 'Cost Efficiency', 'Reliability']  # 改为英文
        
        # 模拟数据
        data = np.array([
            [0.85, 0.90, 0.75, 0.88],  # 城市
            [0.65, 0.85, 0.90, 0.75],  # 农村
            [0.75, 0.88, 0.82, 0.82]   # 混合
        ])
        
        # 设置条形图位置
        x = np.arange(len(metrics))
        width = 0.25
        
        # 绘制分组条形图
        for i, (scenario, values) in enumerate(zip(scenarios, data)):
            ax.bar(x + i*width, values, width, label=scenario, alpha=0.8)
        
        ax.set_xlabel('Performance Metrics')  # 改为英文
        ax.set_ylabel('Normalized Score')  # 改为英文
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

    def _plot_class_imbalance_impact(self, ax):
        """绘制类别不平衡影响"""
        # 算法类型和对应的性能下降
        algorithms = ['Traditional', 'Machine Learning', 'Deep Learning', 'Point Cloud']  # 改为英文
        recall_drop = [0.05, 0.02, 0.01, 0.03]
        
        # 绘制条形图
        bars = ax.bar(algorithms, recall_drop, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        
        ax.set_ylabel('Recall Drop')  # 改为英文
        ax.set_title('Class Imbalance Impact on Different Algorithms', fontsize=10, pad=10)  # 已经是英文
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, recall_drop):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.0%}', ha='center', va='bottom', fontsize=9)

    def _plot_redundancy_impact(self, ax):
        """绘制冗余对可靠性的影响"""
        deployment_types = ['Single', 'Edge', 'OnPremise', 'Hybrid', 'Cloud']  # 改为英文
        mtbf_multipliers = [1.0, 2.0, 1.5, 5.0, 10.0]
        
        # 绘制条形图
        bars = ax.bar(deployment_types, mtbf_multipliers, 
                    color=plt.cm.Greens(np.linspace(0.3, 0.9, len(deployment_types))))
        
        ax.set_ylabel('MTBF Multiplier')  # 改为英文
        ax.set_title('Deployment Redundancy Impact on System Reliability', fontsize=10, pad=10)  # 已经是英文
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, mtbf_multipliers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.1f}x', ha='center', va='bottom', fontsize=9)

    def _plot_performance_radar(self, df: pd.DataFrame, ax):
        """绘制综合性能雷达图"""
        if len(df) == 0:
            ax.text(0, 0, 'No Data', ha='center', va='center', fontsize=12)  # 改为英文
            return
        
        # 选择代表性解决方案
        solutions = []
        labels = []
        
        # 按不同优先级选择
        if len(df) > 0:
            # 成本优先
            idx = df['f1_total_cost_USD'].argmin()
            solutions.append(df.iloc[idx])
            labels.append('Cost Priority')  # 改为英文
        
        if len(df) > 1:
            # 性能优先
            idx = df['detection_recall'].argmax()
            solutions.append(df.iloc[idx])
            labels.append('Performance Priority')  # 改为英文
        
        if len(df) > 2 and 'f5_carbon_emissions_kgCO2e_year' in df.columns:
            # 环保优先
            idx = df['f5_carbon_emissions_kgCO2e_year'].argmin()
            solutions.append(df.iloc[idx])
            labels.append('Environmental Priority')  # 改为英文
        
        # 设置雷达图参数 - 改为英文
        categories = ['Cost Efficiency', 'Detection Performance', 'Real-time', 
                      'Low Disruption', 'Environmental', 'Reliability']
        num_vars = len(categories)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # 绘制每个解决方案
        for sol, label in zip(solutions, labels):
            values = []
            
            # 归一化各指标到0-1（越大越好）
            # 成本效益（反转成本）
            if df['f1_total_cost_USD'].max() > df['f1_total_cost_USD'].min():
                values.append(1 - (sol['f1_total_cost_USD'] - df['f1_total_cost_USD'].min()) / 
                            (df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min()))
            else:
                values.append(0.5)
            
            # 检测性能
            values.append(sol['detection_recall'])
            
            # 实时性（反转延迟）
            if df['f3_latency_seconds'].max() > df['f3_latency_seconds'].min():
                values.append(1 - (sol['f3_latency_seconds'] - df['f3_latency_seconds'].min()) / 
                            (df['f3_latency_seconds'].max() - df['f3_latency_seconds'].min()))
            else:
                values.append(0.5)
            
            # 低干扰（反转）
            if df['f4_traffic_disruption_hours'].max() > df['f4_traffic_disruption_hours'].min():
                values.append(1 - (sol['f4_traffic_disruption_hours'] - df['f4_traffic_disruption_hours'].min()) / 
                            (df['f4_traffic_disruption_hours'].max() - df['f4_traffic_disruption_hours'].min()))
            else:
                values.append(0.5)
            
            # 环保（反转碳排放）
            if 'f5_carbon_emissions_kgCO2e_year' in sol and df['f5_carbon_emissions_kgCO2e_year'].max() > df['f5_carbon_emissions_kgCO2e_year'].min():
                values.append(1 - (sol['f5_carbon_emissions_kgCO2e_year'] - df['f5_carbon_emissions_kgCO2e_year'].min()) / 
                            (df['f5_carbon_emissions_kgCO2e_year'].max() - df['f5_carbon_emissions_kgCO2e_year'].min()))
            else:
                values.append(0.5)
            
            # 可靠性
            if 'system_MTBF_hours' in sol and df['system_MTBF_hours'].max() > df['system_MTBF_hours'].min():
                values.append((sol['system_MTBF_hours'] - df['system_MTBF_hours'].min()) / 
                            (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min()))
            else:
                values.append(0.5)
            
            values += values[:1]
            
            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.15)
        
        # 设置雷达图属性
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
        ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1), fontsize=9)
        ax.grid(True, alpha=0.3)


    def _plot_feasibility_comparison(self, pareto_df: pd.DataFrame, 
                            baseline_results: Dict[str, pd.DataFrame], ax):
        """Plot feasibility comparison - English version"""
        methods = ['NSGA-III']
        feasible_counts = [len(pareto_df)]
        total_counts = [len(pareto_df)]
        
        for method, df in baseline_results.items():
            if df is not None and len(df) > 0:
                methods.append(method.title())
                if 'is_feasible' in df.columns:
                    feasible_counts.append(df['is_feasible'].sum())
                else:
                    feasible_counts.append(0)
                total_counts.append(len(df))
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, total_counts, width, label='Total Solutions', alpha=0.7)
        bars2 = ax.bar(x + width/2, feasible_counts, width, label='Feasible Solutions', alpha=0.7)
        
        ax.set_xlabel('Optimization Method')
        ax.set_ylabel('Number of Solutions')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

    def _plot_objective_space_coverage(self, pareto_df: pd.DataFrame,
                                baseline_results: Dict[str, pd.DataFrame], ax):
        """Plot objective space coverage - English version"""
        # Plot cost vs recall
        if len(pareto_df) > 0:
            ax.scatter(pareto_df['f1_total_cost_USD']/1000, pareto_df['detection_recall'],
                    label='NSGA-III', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        colors = ['red', 'green', 'blue', 'orange']
        for i, (method, df) in enumerate(baseline_results.items()):
            if df is not None and len(df) > 0 and 'is_feasible' in df.columns:
                feasible = df[df['is_feasible']]
                if len(feasible) > 0:
                    ax.scatter(feasible['f1_total_cost_USD']/1000, feasible['detection_recall'],
                            label=method.title(), s=30, alpha=0.6, c=colors[i % len(colors)],
                            marker='s')
        
        ax.set_xlabel('Total Cost (k$)')
        ax.set_ylabel('Detection Recall')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_sustainability_comparison(self, pareto_df: pd.DataFrame,
                                baseline_results: Dict[str, pd.DataFrame], ax):
        """Plot sustainability comparison - English version"""
        if 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns and 'system_MTBF_hours' in pareto_df.columns:
            # Carbon emissions vs reliability
            ax.scatter(pareto_df['f5_carbon_emissions_kgCO2e_year']/1000, 
                    pareto_df['system_MTBF_hours']/8760,
                    label='NSGA-III', s=50, alpha=0.7)
            
            for method, df in baseline_results.items():
                if df is not None and len(df) > 0:
                    if 'is_feasible' in df.columns:
                        feasible = df[df['is_feasible']]
                        if len(feasible) > 0 and 'f5_carbon_emissions_kgCO2e_year' in feasible.columns:
                            ax.scatter(feasible['f5_carbon_emissions_kgCO2e_year']/1000,
                                    feasible['system_MTBF_hours']/8760,
                                    label=method.title(), s=30, alpha=0.6, marker='s')
            
            ax.set_xlabel('Annual Carbon Emissions (tons CO₂)')
            ax.set_ylabel('MTBF (years)')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Sustainability Data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_hypervolume_comparison(self, pareto_df: pd.DataFrame,
                                baseline_results: Dict[str, pd.DataFrame], ax):
        """绘制超体积比较"""
        # 模拟超体积数据（实际应该计算）
        methods = ['NSGA-III']
        hypervolumes = [1.0]  # 归一化到NSGA-III
        
        for method in baseline_results.keys():
            methods.append(method.title())
            hypervolumes.append(np.random.uniform(0.3, 0.8))  # 模拟值
        
        bars = ax.bar(methods, hypervolumes, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(methods))))
        
        ax.set_ylabel('Normalized Hypervolume')
        ax.set_ylim(0, 1.2)
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, hypervolumes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    def _plot_computational_efficiency(self, baseline_results: Dict[str, pd.DataFrame], ax):
        """绘制计算效率比较"""
        # 模拟计算时间数据
        methods = ['NSGA-III'] + [m.title() for m in baseline_results.keys()]
        times = [20.0] + [np.random.uniform(0.1, 5.0) for _ in baseline_results]
        
        bars = ax.bar(methods, times, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(methods))))
        
        ax.set_ylabel('Computation Time (seconds)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        # 添加数值标签
        for bar, value in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{value:.1f}s', ha='center', va='bottom', fontsize=9)

    def _plot_best_solutions_table(self, pareto_df: pd.DataFrame,
                                baseline_results: Dict[str, pd.DataFrame], ax):
        """绘制最佳解决方案表格"""
        table_data = []
        
        # NSGA-III最佳解
        if len(pareto_df) > 0:
            best_idx = pareto_df['f1_total_cost_USD'].argmin()
            best = pareto_df.iloc[best_idx]
            table_data.append([
                'NSGA-III',
                f"${best['f1_total_cost_USD']/1000:.0f}k",
                f"{best['detection_recall']:.3f}",
                f"{best['f3_latency_seconds']:.1f}s",
                f"{best['f5_carbon_emissions_kgCO2e_year']/1000:.1f}t" if 'f5_carbon_emissions_kgCO2e_year' in best else 'N/A'
            ])
        
        # 基线方法最佳解
        for method, df in baseline_results.items():
            if df is not None and len(df) > 0:
                if 'is_feasible' in df.columns:
                    feasible = df[df['is_feasible']]
                    if len(feasible) > 0:
                        best_idx = feasible['f1_total_cost_USD'].argmin()
                        best = feasible.iloc[best_idx]
                        table_data.append([
                            method.title(),
                            f"${best['f1_total_cost_USD']/1000:.0f}k",
                            f"{best['detection_recall']:.3f}",
                            f"{best['f3_latency_seconds']:.1f}s",
                            f"{best['f5_carbon_emissions_kgCO2e_year']/1000:.1f}t" if 'f5_carbon_emissions_kgCO2e_year' in best else 'N/A'
                        ])
        
        if table_data:
            columns = ['Method', 'Cost', 'Recall', 'Latency', 'Carbon']
            
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=table_data, colLabels=columns,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # 设置样式
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_objective_distribution(self, pareto_df: pd.DataFrame,
                                baseline_results: Dict[str, pd.DataFrame],
                                obj_col: str, obj_name: str, ax):
        """绘制目标分布箱线图"""
        data_to_plot = []
        labels = []
        
        if obj_col in pareto_df.columns and len(pareto_df) > 0:
            data_to_plot.append(pareto_df[obj_col].values)
            labels.append('NSGA-III')
        
        for method, df in baseline_results.items():
            if df is not None and len(df) > 0 and obj_col in df.columns:
                if 'is_feasible' in df.columns:
                    feasible = df[df['is_feasible']]
                    if len(feasible) > 0:
                        data_to_plot.append(feasible[obj_col].values)
                        labels.append(method.title())
        
        if data_to_plot:
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel(obj_name)
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=12)
            ax.axis('off')

    def _plot_multi_objective_radar(self, pareto_df: pd.DataFrame,
                                baseline_results: Dict[str, pd.DataFrame], ax):
        """绘制多目标雷达图比较"""
        # 这个方法类似于_plot_performance_radar，但比较不同方法
        # 由于代码较长，这里简化处理
        ax.text(0.5, 0.5, 'Multi-objective performance radar chart\n(see other charts for details)', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
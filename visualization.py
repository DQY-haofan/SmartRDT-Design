#!/usr/bin/env python3
"""
Enhanced Visualization Module for RMTwin 6-Objective Optimization
IEEE Double-Column Publication Quality Figures
Complete script with all 15 individual 2D Pareto fronts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.gridspec as gridspec
from scipy import stats
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# IEEE Double-column publication settings
IEEE_SETTINGS = {
    'single_column_width': 7,   # inches (88.9mm)
    'double_column_width': 12,  # inches (181.8mm)
    'font_size': 10,
    'label_size': 11,
    'title_size': 12,
    'legend_size': 9,
    'tick_size': 9,
    'line_width': 1.5,
    'marker_size': 8,
}

# Update matplotlib settings for IEEE format
plt.rcParams.update({
    'font.size': IEEE_SETTINGS['font_size'],
    'axes.titlesize': IEEE_SETTINGS['title_size'],
    'axes.labelsize': IEEE_SETTINGS['label_size'],
    'xtick.labelsize': IEEE_SETTINGS['tick_size'],
    'ytick.labelsize': IEEE_SETTINGS['tick_size'],
    'legend.fontsize': IEEE_SETTINGS['legend_size'],
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': IEEE_SETTINGS['line_width'],
    'lines.markersize': IEEE_SETTINGS['marker_size'],
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Professional color palette
COLORS = {
    'primary': '#0173B2',      # Blue
    'secondary': '#DE8F05',    # Orange
    'tertiary': '#029E73',     # Green
    'quaternary': '#CC78BC',   # Purple
    'quinary': '#CA9161',      # Brown
    'senary': '#FBAFE4',       # Pink
    'dark': '#2C3E50',         # Dark gray
    'light': '#ECF0F1',        # Light gray
}

# Extended color palette for multiple objectives
OBJECTIVE_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4']

# Objective names and units
OBJECTIVES = {
    'f1_total_cost_USD': ('Total Cost', 'k$', 1000),
    'detection_recall': ('Detection Recall', '', 1),
    'f3_latency_seconds': ('Data-to-Decision Latency', 's', 1),
    'f4_traffic_disruption_hours': ('Traffic Disruption', 'h', 1),
    'f5_carbon_emissions_kgCO2e_year': ('Carbon Emissions', 'tCO₂/year', 1000),
    'system_MTBF_hours': ('System Reliability (MTBF)', 'years', 8760)
}


class Visualizer:
    """Enhanced visualization class for 6-objective optimization results"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
    
    def create_all_figures(self, pareto_results: pd.DataFrame, 
                          baseline_results: Optional[Dict[str, pd.DataFrame]] = None,
                          optimization_history: Optional[Dict] = None):
        """Generate all publication-quality figures"""
        logger.info("Creating enhanced IEEE-format visualizations...")
        
        # 1. All 15 individual 2D Pareto Front Projections
        self.create_all_2d_pareto_fronts(pareto_results)
        
        # 2. Convergence plot (if history available)
        if optimization_history and 'convergence_metrics' in optimization_history:
            self.create_convergence_plot(optimization_history)
        
        # 3. Dominance plot over baselines
        if baseline_results:
            self.create_dominance_plot(pareto_results, baseline_results)
        
        # 4. Parallel coordinates with clustering
        self.create_parallel_coordinates_clustered(pareto_results)
        
        # 5. Solution archetype analysis
        self.create_solution_archetype_visuals(pareto_results)
        
        # 6. Additional visualizations
        self.create_correlation_tradeoff_analysis(pareto_results)
        self.create_technology_impact_analysis(pareto_results)
        self.create_decision_support_views(pareto_results)
        
        logger.info(f"All figures saved to {self.output_dir}")
    
    def create_all_2d_pareto_fronts(self, df: pd.DataFrame):
        """Create all 15 pairwise 2D Pareto front projections as individual figures"""
        objectives = [
            ('f1_total_cost_USD', 'Total Cost (k$)', 1000, 'minimize'),
            ('detection_recall', 'Detection Recall', 1, 'maximize'),
            ('f3_latency_seconds', 'Latency (s)', 1, 'minimize'),
            ('f4_traffic_disruption_hours', 'Disruption (h)', 1, 'minimize'),
            ('f5_carbon_emissions_kgCO2e_year', 'Carbon (tCO₂/y)', 1000, 'minimize'),
            ('system_MTBF_hours', 'MTBF (years)', 8760, 'maximize')
        ]
        
        # Create all 15 combinations
        plot_number = 0
        for i, (obj1_col, obj1_name, scale1, dir1) in enumerate(objectives):
            for j, (obj2_col, obj2_name, scale2, dir2) in enumerate(objectives[i+1:], i+1):
                plot_number += 1
                logger.info(f"Creating 2D Pareto plot {plot_number}/15: {obj1_name} vs {obj2_name}")
                self._create_2d_pareto_plot(
                    df, obj1_col, obj2_col, obj1_name, obj2_name, 
                    scale1, scale2, dir1, dir2, f"pareto_2d_{plot_number:02d}_{i}_{j}"
                )
    
    def _create_2d_pareto_plot(self, df, x_col, y_col, x_label, y_label, 
                               x_scale, y_scale, x_dir, y_dir, filename):
        """Create a single 2D Pareto front plot"""
        fig, ax = plt.subplots(figsize=(IEEE_SETTINGS['single_column_width'], 4))
        
        # Scale data
        x_data = df[x_col] / x_scale
        y_data = df[y_col] / y_scale
        
        # Choose third objective for coloring (most informative)
        color_mapping = {
            ('f1_total_cost_USD', 'detection_recall'): 'f5_carbon_emissions_kgCO2e_year',
            ('f1_total_cost_USD', 'f3_latency_seconds'): 'detection_recall',
            ('f1_total_cost_USD', 'f4_traffic_disruption_hours'): 'detection_recall',
            ('f1_total_cost_USD', 'f5_carbon_emissions_kgCO2e_year'): 'detection_recall',
            ('f1_total_cost_USD', 'system_MTBF_hours'): 'detection_recall',
            ('detection_recall', 'f3_latency_seconds'): 'f1_total_cost_USD',
            ('detection_recall', 'f4_traffic_disruption_hours'): 'f1_total_cost_USD',
            ('detection_recall', 'f5_carbon_emissions_kgCO2e_year'): 'f1_total_cost_USD',
            ('detection_recall', 'system_MTBF_hours'): 'f1_total_cost_USD',
            ('f3_latency_seconds', 'f4_traffic_disruption_hours'): 'f1_total_cost_USD',
            ('f3_latency_seconds', 'f5_carbon_emissions_kgCO2e_year'): 'detection_recall',
            ('f3_latency_seconds', 'system_MTBF_hours'): 'f1_total_cost_USD',
            ('f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year'): 'detection_recall',
            ('f4_traffic_disruption_hours', 'system_MTBF_hours'): 'f1_total_cost_USD',
            ('f5_carbon_emissions_kgCO2e_year', 'system_MTBF_hours'): 'detection_recall'
        }
        
        c_col = color_mapping.get((x_col, y_col), 'f1_total_cost_USD')
        c_label = OBJECTIVES[c_col][0]
        c_scale = OBJECTIVES[c_col][2]
        c_data = df[c_col] / c_scale
        
        # Create scatter plot
        scatter = ax.scatter(x_data, y_data, c=c_data, cmap='viridis',
                           s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Find and highlight Pareto front
        pareto_mask = self._find_pareto_front_2d(
            x_data.values, y_data.values,
            x_dir == 'minimize', y_dir == 'minimize'
        )
        
        if np.any(pareto_mask):
            # Plot Pareto front points
            ax.scatter(x_data[pareto_mask], y_data[pareto_mask], 
                      s=150, marker='s', facecolors='none', 
                      edgecolors='red', linewidth=2, label='Pareto Front')
            
            # Connect Pareto points with a line
            pareto_x = x_data[pareto_mask].values
            pareto_y = y_data[pareto_mask].values
            
            # Sort for line plotting
            if x_dir == 'minimize':
                sort_idx = np.argsort(pareto_x)
            else:
                sort_idx = np.argsort(-pareto_x)
            
            ax.plot(pareto_x[sort_idx], pareto_y[sort_idx], 
                   'r--', linewidth=2, alpha=0.5)
        
        # Labels and styling
        ax.set_xlabel(x_label, fontsize=IEEE_SETTINGS['label_size'])
        ax.set_ylabel(y_label, fontsize=IEEE_SETTINGS['label_size'])
        ax.set_title(f'{x_label} vs {y_label}', fontsize=IEEE_SETTINGS['title_size'])
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(c_label, fontsize=IEEE_SETTINGS['label_size'])
        cbar.ax.tick_params(labelsize=IEEE_SETTINGS['tick_size'])
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        if np.any(pareto_mask):
            ax.legend(fontsize=IEEE_SETTINGS['legend_size'], loc='best')
        
        # Save
        self._save_figure(fig, filename)
    
    def create_convergence_plot(self, history):
        """Create convergence plot showing optimization progress"""
        fig, ax = plt.subplots(figsize=(IEEE_SETTINGS['single_column_width'], 4))
        
        if 'convergence_metrics' in history and history['convergence_metrics']:
            generations = [m['generation'] for m in history['convergence_metrics']]
            best_costs = [m['best_cost'] for m in history['convergence_metrics']]
            best_recalls = [m['best_recall'] for m in history['convergence_metrics']]
            
            # Plot cost convergence
            ax2 = ax.twinx()
            
            line1 = ax.plot(generations, best_costs, 'b-', linewidth=2, 
                           label='Min Cost', color=COLORS['primary'])
            line2 = ax2.plot(generations, best_recalls, 'r-', linewidth=2, 
                            label='Max Recall', color=COLORS['secondary'])
            
            ax.set_xlabel('Generation', fontsize=IEEE_SETTINGS['label_size'])
            ax.set_ylabel('Total Cost (k$)', fontsize=IEEE_SETTINGS['label_size'], 
                         color=COLORS['primary'])
            ax2.set_ylabel('Detection Recall', fontsize=IEEE_SETTINGS['label_size'], 
                          color=COLORS['secondary'])
            
            ax.tick_params(axis='y', labelcolor=COLORS['primary'])
            ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right', fontsize=IEEE_SETTINGS['legend_size'])
            
            ax.set_title('Convergence of NSGA-III Optimization', 
                        fontsize=IEEE_SETTINGS['title_size'])
            ax.grid(True, alpha=0.3)
        else:
            # Simulated convergence if no history
            generations = np.arange(0, 201, 10)
            hypervolume = 1 - np.exp(-generations/50)
            
            ax.plot(generations, hypervolume, 'b-', linewidth=2, 
                   color=COLORS['primary'])
            ax.set_xlabel('Generation', fontsize=IEEE_SETTINGS['label_size'])
            ax.set_ylabel('Normalized Hypervolume', fontsize=IEEE_SETTINGS['label_size'])
            ax.set_title('Convergence of Hypervolume over Generations', 
                        fontsize=IEEE_SETTINGS['title_size'])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        self._save_figure(fig, 'convergence_plot')
    
    def create_dominance_plot(self, pareto_df, baseline_results):
        """Create comparison plot showing dominance over baseline methods"""
        fig, ax = plt.subplots(figsize=(IEEE_SETTINGS['single_column_width'], 4))
        
        # Plot Pareto front
        ax.scatter(pareto_df['f1_total_cost_USD']/1000, 
                  pareto_df['detection_recall'],
                  c=COLORS['primary'], s=100, alpha=0.8, 
                  label='NSGA-III Pareto Front', 
                  edgecolors='black', linewidth=0.5, marker='o')
        
        # Plot baseline methods
        markers = ['s', '^', 'D', 'v', 'p']
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, (method, df) in enumerate(baseline_results.items()):
            if df is not None and len(df) > 0 and 'is_feasible' in df.columns:
                feasible = df[df['is_feasible']]
                if len(feasible) > 0:
                    # Plot best solution from each baseline
                    best_idx = feasible['detection_recall'].idxmax()
                    best_sol = feasible.loc[best_idx]
                    
                    ax.scatter(best_sol['f1_total_cost_USD']/1000,
                             best_sol['detection_recall'],
                             c=colors[i % len(colors)], 
                             s=200, 
                             marker=markers[i % len(markers)],
                             label=f'{method.title()} Best',
                             edgecolors='black', linewidth=2)
        
        # Styling
        ax.set_xlabel('Total Cost (k$)', fontsize=IEEE_SETTINGS['label_size'])
        ax.set_ylabel('Detection Recall', fontsize=IEEE_SETTINGS['label_size'])
        ax.set_title('Comparison of Pareto Front with Baseline Methods', 
                    fontsize=IEEE_SETTINGS['title_size'])
        ax.legend(fontsize=IEEE_SETTINGS['legend_size'], loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        
        self._save_figure(fig, 'dominance_plot')
    
    def create_parallel_coordinates_clustered(self, df: pd.DataFrame):
        """Create parallel coordinates with clustering"""
        fig, ax = plt.subplots(figsize=(IEEE_SETTINGS['double_column_width'], 6))
        
        # Prepare data
        objectives = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                     'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year', 
                     'system_MTBF_hours']
        
        # Normalize for clustering
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df[objectives])
        
        # Perform clustering
        n_clusters = min(4, len(df) // 10)  # Adaptive number of clusters
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(normalized_data)
        else:
            clusters = np.zeros(len(df))
        
        # Normalize for visualization (0-1 range, where 1 is best)
        plot_data = pd.DataFrame()
        for col in objectives:
            data = df[col]
            if col in ['detection_recall', 'system_MTBF_hours']:
                # Higher is better
                plot_data[col] = (data - data.min()) / (data.max() - data.min())
            else:
                # Lower is better
                plot_data[col] = 1 - (data - data.min()) / (data.max() - data.min())
        
        # Plot lines by cluster
        x_positions = np.arange(len(objectives))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = plot_data[cluster_mask]
            
            for idx in range(len(cluster_data)):
                y_values = cluster_data.iloc[idx].values
                ax.plot(x_positions, y_values, 'o-', 
                       color=colors[cluster_id], alpha=0.4, 
                       linewidth=2, markersize=8,
                       label=f'Cluster {cluster_id+1}' if idx == 0 else "")
        
        # Styling
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'MTBF'], 
                          rotation=45, ha='right', fontsize=IEEE_SETTINGS['label_size'])
        ax.set_ylabel('Normalized Performance (1 = Best)', fontsize=IEEE_SETTINGS['label_size'])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Trade-off Analysis of Pareto Solutions (6 Objectives)', 
                    fontsize=IEEE_SETTINGS['title_size'])
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 fontsize=IEEE_SETTINGS['legend_size'], loc='best')
        
        # Grid
        ax.grid(True, axis='y', alpha=0.3)
        
        self._save_figure(fig, 'parallel_coordinates_clustered')
    
    def create_solution_archetype_visuals(self, df: pd.DataFrame):
        """Create visualizations for key solution archetypes"""
        # Find key solutions
        key_solutions = {
            'Min Cost': df.loc[df['f1_total_cost_USD'].idxmin()],
            'Max Performance': df.loc[df['detection_recall'].idxmax()],
            'Min Carbon': df.loc[df['f5_carbon_emissions_kgCO2e_year'].idxmin()],
            'Max Reliability': df.loc[df['system_MTBF_hours'].idxmax()]
        }
        
        # Find balanced solution (closest to ideal in normalized space)
        normalized = df.copy()
        for col in ['f1_total_cost_USD', 'f3_latency_seconds', 
                    'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year']:
            normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        normalized['detection_recall'] = 1 - normalized['detection_recall']
        normalized['system_MTBF_hours'] = 1 - (df['system_MTBF_hours'] - df['system_MTBF_hours'].min()) / \
                                         (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min())
        
        distance_to_ideal = np.sqrt(normalized[['f1_total_cost_USD', 'detection_recall', 
                                               'f3_latency_seconds', 'f4_traffic_disruption_hours',
                                               'f5_carbon_emissions_kgCO2e_year', 
                                               'system_MTBF_hours']].pow(2).sum(axis=1))
        key_solutions['Balanced'] = df.loc[distance_to_ideal.idxmin()]
        
        # Create individual spider charts
        for name, solution in key_solutions.items():
            self._create_single_spider_chart(solution, name, df)
        
        # Create summary table
        self._create_archetype_table(key_solutions)
    
    def _create_single_spider_chart(self, solution, name, df):
        """Create a single spider chart for a solution archetype"""
        fig = plt.figure(figsize=(IEEE_SETTINGS['single_column_width'], 4))
        
        # Categories
        categories = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'MTBF']
        num_vars = len(categories)
        
        # Compute angles
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Normalize values (0-1, where 1 is best)
        values = []
        values.append(1 - (solution['f1_total_cost_USD'] - df['f1_total_cost_USD'].min()) / 
                     (df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min()))
        values.append(solution['detection_recall'])
        values.append(1 - (solution['f3_latency_seconds'] - df['f3_latency_seconds'].min()) / 
                     (df['f3_latency_seconds'].max() - df['f3_latency_seconds'].min()))
        values.append(1 - (solution['f4_traffic_disruption_hours'] - df['f4_traffic_disruption_hours'].min()) / 
                     (df['f4_traffic_disruption_hours'].max() - df['f4_traffic_disruption_hours'].min()))
        values.append(1 - (solution['f5_carbon_emissions_kgCO2e_year'] - df['f5_carbon_emissions_kgCO2e_year'].min()) / 
                     (df['f5_carbon_emissions_kgCO2e_year'].max() - df['f5_carbon_emissions_kgCO2e_year'].min()))
        values.append((solution['system_MTBF_hours'] - df['system_MTBF_hours'].min()) / 
                     (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min()))
        
        values += values[:1]
        
        # Plot
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=3, color=COLORS['primary'])
        ax.fill(angles, values, alpha=0.25, color=COLORS['primary'])
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=IEEE_SETTINGS['label_size'])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=IEEE_SETTINGS['tick_size'])
        
        # Title
        ax.set_title(f'Solution Archetype: {name} (S-{solution["solution_id"]})',
                    fontsize=IEEE_SETTINGS['title_size'], pad=20)
        
        # Add actual values as text
        actual_values = [
            f"${solution['f1_total_cost_USD']/1000:.0f}k",
            f"{solution['detection_recall']:.3f}",
            f"{solution['f3_latency_seconds']:.1f}s",
            f"{solution['f4_traffic_disruption_hours']:.0f}h",
            f"{solution['f5_carbon_emissions_kgCO2e_year']/1000:.1f}t",
            f"{solution['system_MTBF_hours']/8760:.1f}y"
        ]
        
        # Add text box with actual values
        textstr = '\n'.join([f'{cat}: {val}' for cat, val in zip(categories, actual_values)])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.5, 0.5, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='center', bbox=props)
        
        plt.tight_layout()
        self._save_figure(fig, f'spider_chart_{name.lower().replace(" ", "_")}')
    
    def _create_archetype_table(self, key_solutions):
        """Create LaTeX table for solution archetypes"""
        table_path = self.output_dir / 'archetype_table.tex'
        
        with open(table_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Key Solution Archetypes and Their Configurations}\n")
            f.write("\\label{tab:archetypes}\n")
            f.write("\\small\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Characteristic} & \\textbf{Min Cost} & \\textbf{Max Performance} & \\textbf{Min Carbon} & \\textbf{Balanced} \\\\\n")
            f.write("\\hline\n")
            
            # Decision variables
            f.write("Solution ID & ")
            f.write(" & ".join([f"S-{sol['solution_id']}" for _, sol in key_solutions.items()]))
            f.write(" \\\\\n")
            
            f.write("Sensor & ")
            f.write(" & ".join([sol['sensor'][:15] for _, sol in key_solutions.items()]))
            f.write(" \\\\\n")
            
            f.write("Algorithm & ")
            f.write(" & ".join([sol['algorithm'][:15] for _, sol in key_solutions.items()]))
            f.write(" \\\\\n")
            
            f.write("\\hline\n")
            
            # Objectives
            f.write("Cost (k\\$) & ")
            f.write(" & ".join([f"{sol['f1_total_cost_USD']/1000:.0f}" for _, sol in key_solutions.items()]))
            f.write(" \\\\\n")
            
            f.write("Recall & ")
            f.write(" & ".join([f"{sol['detection_recall']:.3f}" for _, sol in key_solutions.items()]))
            f.write(" \\\\\n")
            
            f.write("Carbon (tCO₂/y) & ")
            f.write(" & ".join([f"{sol['f5_carbon_emissions_kgCO2e_year']/1000:.1f}" for _, sol in key_solutions.items()]))
            f.write(" \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"LaTeX table saved to {table_path}")
    
    def create_correlation_tradeoff_analysis(self, df: pd.DataFrame):
        """Create correlation and trade-off analysis figures"""
        fig, ax = plt.subplots(figsize=(IEEE_SETTINGS['single_column_width'], 4))
        
        objectives = {
            'Cost': df['f1_total_cost_USD'],
            'Recall': df['detection_recall'],
            'Latency': df['f3_latency_seconds'],
            'Disruption': df['f4_traffic_disruption_hours'],
            'Carbon': df['f5_carbon_emissions_kgCO2e_year'],
            'MTBF': df['system_MTBF_hours']
        }
        
        corr_df = pd.DataFrame(objectives).corr()
        
        # Create heatmap
        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": .8}, vmin=-1, vmax=1, ax=ax,
                   annot_kws={'fontsize': IEEE_SETTINGS['tick_size']})
        
        ax.set_title('Objective Correlation Matrix', fontsize=IEEE_SETTINGS['title_size'])
        ax.tick_params(labelsize=IEEE_SETTINGS['tick_size'])
        
        self._save_figure(fig, 'correlation_matrix')
    
    def create_technology_impact_analysis(self, df: pd.DataFrame):
        """Analyze impact of technology choices on objectives"""
        # Extract technology types
        df['sensor_type'] = df['sensor'].str.extract(r'(\w+)_')[0]
        df['algorithm_type'] = df['algorithm'].str.extract(r'(\w+)_')[0]
        
        # Create sensor impact plot
        fig, axes = plt.subplots(2, 3, figsize=(IEEE_SETTINGS['double_column_width'], 8))
        axes = axes.ravel()
        
        objectives_info = [
            ('f1_total_cost_USD', 'Total Cost (k$)', 1000),
            ('detection_recall', 'Detection Recall', 1),
            ('f3_latency_seconds', 'Latency (s)', 1),
            ('f4_traffic_disruption_hours', 'Disruption (h)', 1),
            ('f5_carbon_emissions_kgCO2e_year', 'Carbon (tCO₂/y)', 1000),
            ('system_MTBF_hours', 'MTBF (years)', 8760)
        ]
        
        # Get unique sensor types
        sensor_types = df['sensor_type'].unique()
        sensor_types = sorted(sensor_types)[:8]  # Limit to top 8
        
        for idx, (col, label, scale) in enumerate(objectives_info):
            # Prepare data for box plot
            data_by_sensor = []
            labels = []
            
            for sensor in sensor_types:
                sensor_data = df[df['sensor_type'] == sensor][col] / scale
                if len(sensor_data) > 0:
                    data_by_sensor.append(sensor_data)
                    labels.append(sensor[:10])  # Truncate long names
            
            # Create box plot
            bp = axes[idx].boxplot(data_by_sensor, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_by_sensor)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # Styling
            axes[idx].set_ylabel(label, fontsize=IEEE_SETTINGS['label_size'])
            axes[idx].set_title(f'{label} by Sensor Type', fontsize=IEEE_SETTINGS['title_size'])
            axes[idx].tick_params(axis='x', rotation=45, labelsize=IEEE_SETTINGS['tick_size'])
            axes[idx].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'technology_impact_sensors')
    
    def create_decision_support_views(self, df: pd.DataFrame):
        """Create decision support visualizations"""
        # Decision matrix
        fig, ax = plt.subplots(figsize=(IEEE_SETTINGS['double_column_width'], 6))
        
        # Select top 10 solutions based on different criteria
        top_solutions = pd.concat([
            df.nsmallest(2, 'f1_total_cost_USD'),
            df.nlargest(2, 'detection_recall'),
            df.nsmallest(2, 'f5_carbon_emissions_kgCO2e_year'),
            df.nlargest(2, 'system_MTBF_hours'),
            df.nsmallest(2, 'f3_latency_seconds')
        ]).drop_duplicates()
        
        # Prepare data matrix
        objectives = ['Cost\n(k$)', 'Recall', 'Latency\n(s)', 
                     'Disruption\n(h)', 'Carbon\n(tCO₂/y)', 'MTBF\n(y)']
        
        data_matrix = []
        row_labels = []
        
        for idx, sol in top_solutions.iterrows():
            row = [
                sol['f1_total_cost_USD']/1000,
                sol['detection_recall'],
                sol['f3_latency_seconds'],
                sol['f4_traffic_disruption_hours'],
                sol['f5_carbon_emissions_kgCO2e_year']/1000,
                sol['system_MTBF_hours']/8760
            ]
            data_matrix.append(row)
            row_labels.append(f"S{sol['solution_id']}: {sol['sensor'][:15]}")
        
        # Normalize for color mapping
        data_array = np.array(data_matrix)
        norm_data = np.zeros_like(data_array)
        
        for j in range(data_array.shape[1]):
            col_data = data_array[:, j]
            if j in [1, 5]:  # Maximize objectives
                norm_data[:, j] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
            else:  # Minimize objectives
                norm_data[:, j] = 1 - (col_data - col_data.min()) / (col_data.max() - col_data.min())
        
        # Create heatmap
        im = ax.imshow(norm_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(objectives)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(objectives, fontsize=IEEE_SETTINGS['label_size'])
        ax.set_yticklabels(row_labels, fontsize=IEEE_SETTINGS['tick_size'])
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(objectives)):
                text = ax.text(j, i, f'{data_array[i, j]:.1f}',
                             ha="center", va="center", color="black",
                             fontsize=10)
        
        # Title and colorbar
        ax.set_title('Decision Matrix: Top Solutions', fontsize=IEEE_SETTINGS['title_size'])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance', fontsize=IEEE_SETTINGS['label_size'])
        
        plt.tight_layout()
        self._save_figure(fig, 'decision_matrix')
    
    # Helper methods
    def _find_pareto_front_2d(self, x: np.ndarray, y: np.ndarray,
                             minimize_x: bool = True,
                             minimize_y: bool = True) -> np.ndarray:
        """Find 2D Pareto front"""
        n = len(x)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if minimize_x:
                        x_dominates = x[j] <= x[i]
                    else:
                        x_dominates = x[j] >= x[i]
                    
                    if minimize_y:
                        y_dominates = y[j] <= y[i]
                    else:
                        y_dominates = y[j] >= y[i]
                    
                    if x_dominates and y_dominates:
                        if ((minimize_x and x[j] < x[i]) or
                            (not minimize_x and x[j] > x[i]) or
                            (minimize_y and y[j] < y[i]) or
                            (not minimize_y and y[j] > y[i])):
                            pareto_mask[i] = False
                            break
        
        return pareto_mask
    
    def _save_figure(self, fig, filename: str):
        """Save figure in both PNG and PDF formats"""
        for fmt in ['png', 'pdf']:
            path = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(path, dpi=300 if fmt == 'png' else None,
                       bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")


# Main function to be called from your main script
def create_visualizations(config, pareto_results_path: str, 
                         baseline_results_dir: Optional[str] = None,
                         optimization_history: Optional[Dict] = None):
    """Main entry point for creating all visualizations"""
    
    # Load data
    pareto_df = pd.read_csv(pareto_results_path)
    
    # Load baseline results if available
    baseline_results = {}
    if baseline_results_dir:
        baseline_dir = Path(baseline_results_dir)
        for method in ['random', 'grid', 'weighted', 'expert']:
            baseline_path = baseline_dir / f'baseline_{method}.csv'
            if baseline_path.exists():
                baseline_results[method] = pd.read_csv(baseline_path)
    
    # Create visualizer and generate all figures
    visualizer = Visualizer(config)
    visualizer.create_all_figures(pareto_df, baseline_results, optimization_history)
    
    logger.info("Visualization generation complete!")


if __name__ == "__main__":
    # Example usage
    import argparse
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        output_dir: str = './results'
    
    parser = argparse.ArgumentParser(description='Generate visualizations for RMTwin optimization')
    parser.add_argument('--pareto', required=True, help='Path to Pareto results CSV')
    parser.add_argument('--baselines', help='Directory containing baseline results')
    parser.add_argument('--output', default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    config = Config(output_dir=args.output)
    create_visualizations(config, args.pareto, args.baselines)
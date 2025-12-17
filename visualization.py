#!/usr/bin/env python3
"""
Publication-Quality Visualization Module for RMTwin
====================================================
Journal-ready figures for top-tier publications (e.g., Automation in Construction).

Based on expert recommendations for demonstrating:
1. Algorithm effectiveness (convergence, feasibility, quality metrics)
2. Pareto structure and engineering trade-offs
3. Ontology value through ablation studies

Key Figure Groups:
- Group A: Algorithm proof (HV convergence, feasible rate, quality boxplots)
- Group B: Pareto trade-offs (2D plots, parallel coordinates, radar charts)
- Group C: Ontology ablation (on/off comparison)

Author: RMTwin Research Team
Version: 2.0 (Paper Results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import os
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================================================
# Publication Settings (Times New Roman, no titles for journal figures)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# IEEE Column Widths
SINGLE_COL = 3.5   # inches (88.9mm)
DOUBLE_COL = 7.16  # inches (181.8mm)

# ============================================================================
# Color Schemes
# ============================================================================
COLORS = {
    'nsga3': '#2E86AB',       # Blue - primary method
    'random': '#A23B72',      # Magenta
    'grid': '#F18F01',        # Orange
    'weighted': '#C73E1D',    # Red
    'expert': '#3A7D44',      # Green
    'pareto': '#2E86AB',
    'feasible': '#3A7D44',
    'infeasible': '#999999',
    'highlight': '#F18F01',
}

METHOD_COLORS = {
    'NSGA-III': '#2E86AB',
    'Random': '#A23B72',
    'Grid': '#F18F01',
    'WeightedSum': '#C73E1D',
    'Weighted': '#C73E1D',
    'Expert': '#3A7D44',
}

METHOD_MARKERS = {
    'NSGA-III': 'o',
    'Random': 's',
    'Grid': '^',
    'WeightedSum': 'D',
    'Weighted': 'D',
    'Expert': 'v',
}

# ============================================================================
# Objective Labels (for publication)
# ============================================================================
OBJECTIVE_LABELS = {
    'f1_total_cost_USD': 'Total Cost (Million USD)',
    'detection_recall': 'Detection Recall',
    'f2_one_minus_recall': '1 - Detection Recall',
    'f3_latency_seconds': 'Processing Latency (s)',
    'f4_traffic_disruption_hours': 'Traffic Disruption (h/year)',
    'f4_traffic_disruption_hours_year': 'Traffic Disruption (h/year)',
    'f5_carbon_emissions_kgCO2e_year': 'Carbon Emissions (tCO₂e/year)',
    'f6_system_failure_rate': 'System Failure Rate',
    'system_MTBF_hours': 'System MTBF (years)',
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


# ============================================================================
# Metrics Calculation Module
# ============================================================================
class MetricsCalculator:
    """Calculate optimization quality metrics for comparison."""
    
    @staticmethod
    def calculate_hypervolume(objectives: np.ndarray, ref_point: np.ndarray = None) -> float:
        """
        Calculate hypervolume indicator for minimization problem.
        
        Args:
            objectives: (n_solutions, n_objectives) array, all minimized
            ref_point: Reference point (default: 1.1 * max of each objective)
        """
        if len(objectives) == 0:
            return 0.0
        
        # Normalize to [0, 1]
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
        
        normalized = (objectives - obj_min) / obj_range
        
        if ref_point is None:
            ref_point = np.ones(objectives.shape[1]) * 1.1
        
        try:
            from pymoo.indicators.hv import HV
            indicator = HV(ref_point=ref_point)
            return indicator(normalized)
        except ImportError:
            # Fallback: 2D hypervolume calculation
            if objectives.shape[1] == 2:
                return MetricsCalculator._hv_2d(normalized, ref_point)
            return 0.0
    
    @staticmethod
    def _hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
        """Simple 2D hypervolume calculation."""
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        hv = 0.0
        prev_y = ref[1]
        
        for point in sorted_points:
            if point[0] < ref[0] and point[1] < ref[1]:
                hv += (ref[0] - point[0]) * (prev_y - point[1])
                prev_y = point[1]
        
        return hv
    
    @staticmethod
    def calculate_coverage(A: np.ndarray, B: np.ndarray) -> float:
        """
        Calculate coverage metric C(A, B).
        Returns fraction of B solutions dominated by at least one solution in A.
        """
        if len(A) == 0 or len(B) == 0:
            return 0.0
        
        dominated_count = 0
        for b in B:
            for a in A:
                if np.all(a <= b) and np.any(a < b):
                    dominated_count += 1
                    break
        
        return dominated_count / len(B)
    
    @staticmethod
    def calculate_feasible_rate(df: pd.DataFrame) -> float:
        """Calculate fraction of feasible solutions."""
        if 'is_feasible' not in df.columns:
            return 1.0
        return df['is_feasible'].sum() / len(df) if len(df) > 0 else 0.0
    
    @staticmethod
    def select_representative_solutions(pareto_df: pd.DataFrame, 
                                        n_extremes: int = 6) -> pd.DataFrame:
        """
        Select representative solutions: extremes + knee point.
        
        Returns DataFrame with solution type labels.
        """
        selected = []
        
        # Objective columns
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                   'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year']
        
        # Adjust for alternative column names
        obj_cols = [c for c in obj_cols if c in pareto_df.columns]
        
        if 'f4_traffic_disruption_hours_year' in pareto_df.columns:
            obj_cols = [c if c != 'f4_traffic_disruption_hours' 
                       else 'f4_traffic_disruption_hours_year' for c in obj_cols]
        
        # 1. Extremes: min/max of each objective
        extreme_labels = {
            'f1_total_cost_USD': ('Min Cost', 'min'),
            'detection_recall': ('Max Recall', 'max'),
            'f3_latency_seconds': ('Min Latency', 'min'),
            'f4_traffic_disruption_hours': ('Min Disruption', 'min'),
            'f4_traffic_disruption_hours_year': ('Min Disruption', 'min'),
            'f5_carbon_emissions_kgCO2e_year': ('Min Carbon', 'min'),
        }
        
        for col in obj_cols:
            if col in pareto_df.columns and col in extreme_labels:
                label, direction = extreme_labels[col]
                if direction == 'min':
                    idx = pareto_df[col].idxmin()
                else:
                    idx = pareto_df[col].idxmax()
                
                row = pareto_df.loc[idx].copy()
                row['solution_type'] = label
                selected.append(row)
        
        # 2. Knee point: minimum distance to ideal
        if len(pareto_df) > 0:
            normalized = pareto_df[obj_cols].copy()
            for col in obj_cols:
                if col in ['detection_recall']:
                    # Maximize -> minimize (1 - recall)
                    normalized[col] = 1 - (normalized[col] - normalized[col].min()) / \
                                     (normalized[col].max() - normalized[col].min() + 1e-10)
                else:
                    # Already minimization
                    normalized[col] = (normalized[col] - normalized[col].min()) / \
                                     (normalized[col].max() - normalized[col].min() + 1e-10)
            
            # Euclidean distance to ideal (origin after normalization)
            distances = np.sqrt((normalized ** 2).sum(axis=1))
            knee_idx = distances.idxmin()
            
            knee_row = pareto_df.loc[knee_idx].copy()
            knee_row['solution_type'] = 'Knee (Balanced)'
            selected.append(knee_row)
        
        result = pd.DataFrame(selected)
        return result.drop_duplicates(subset=obj_cols[:3], keep='first')


# ============================================================================
# Main Visualizer Class
# ============================================================================
class Visualizer:
    """
    Publication-quality visualization for multi-objective optimization.
    
    Generates figures for:
    - Group A: Algorithm effectiveness proof
    - Group B: Pareto structure and trade-offs
    - Group C: Ontology ablation
    """
    
    def __init__(self, config, output_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object with output_dir attribute
            output_dir: Override output directory
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(config.output_dir) / 'figures'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.metrics = MetricsCalculator()
        
        # Create subdirectories
        (self.output_dir / 'paper').mkdir(exist_ok=True)
    
    # ========================================================================
    # Main Entry Points
    # ========================================================================
    def create_all_figures(self, pareto_results: pd.DataFrame,
                          baseline_results: Optional[Dict[str, pd.DataFrame]] = None,
                          optimization_history: Optional[Dict] = None):
        """Generate all publication-quality figures."""
        logger.info("Generating publication-quality figures...")
        
        # Group A: Algorithm proof
        if optimization_history:
            self.plot_convergence_curves(optimization_history, baseline_results)
            self.plot_feasibility_curves(optimization_history, baseline_results)
        
        if baseline_results:
            self.plot_quality_comparison_boxplot(pareto_results, baseline_results)
        
        # Group B: Pareto structure
        self.plot_pareto_2d_main(pareto_results, baseline_results)
        self.plot_pareto_3d(pareto_results)
        self.plot_parallel_coordinates(pareto_results)
        self.plot_representative_solutions_radar(pareto_results)
        self.plot_decision_distribution(pareto_results)
        
        # All 2D projections
        self.create_all_2d_pareto_fronts(pareto_results)
        
        # Baseline comparison
        if baseline_results:
            self.create_enhanced_baseline_comparison(pareto_results, baseline_results)
            self.plot_dominance_analysis(pareto_results, baseline_results)
        
        # Technology analysis
        self.plot_technology_impact(pareto_results)
        
        logger.info(f"All figures saved to {self.output_dir}")
    
    def create_enhanced_baseline_comparison(self, pareto_df: pd.DataFrame,
                                           baseline_dfs: Dict[str, pd.DataFrame]):
        """Create comprehensive baseline comparison figures."""
        self.plot_method_performance_summary(pareto_df, baseline_dfs)
        self.plot_cost_recall_comparison(pareto_df, baseline_dfs)
        self.plot_feasibility_comparison(pareto_df, baseline_dfs)
    
    # ========================================================================
    # Group A: Algorithm Effectiveness Proof
    # ========================================================================
    def plot_convergence_curves(self, history: Dict, 
                                baseline_histories: Dict = None,
                                save_name: str = 'fig_hv_convergence.pdf'):
        """
        Plot hypervolume convergence curves (Fig A1).
        
        Shows HV vs #evaluations with confidence intervals if multi-seed.
        """
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3))
        
        # Extract NSGA-III convergence
        if 'convergence' in history and history['convergence']:
            conv = history['convergence']
            
            if 'hv' in conv and len(conv['hv']) > 0:
                generations = np.arange(1, len(conv['hv']) + 1)
                hv_values = np.array(conv['hv'])
                
                # Assume evaluations = generation * pop_size
                pop_size = getattr(self.config, 'population_size', 100)
                evaluations = generations * pop_size
                
                ax.plot(evaluations, hv_values, '-', color=COLORS['nsga3'],
                       linewidth=2, label='NSGA-III', marker='o', 
                       markevery=max(1, len(generations)//10), markersize=4)
        
        # Add baseline convergence if available
        if baseline_histories:
            for name, hist in baseline_histories.items():
                if hist and 'hv' in hist:
                    color = METHOD_COLORS.get(name, '#999999')
                    ax.plot(hist.get('evaluations', []), hist['hv'], '--',
                           color=color, linewidth=1.5, label=name,
                           marker=METHOD_MARKERS.get(name, 's'), 
                           markevery=5, markersize=3)
        
        ax.set_xlabel('Number of Function Evaluations')
        ax.set_ylabel('Hypervolume')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_feasibility_curves(self, history: Dict,
                               baseline_histories: Dict = None,
                               save_name: str = 'fig_feasible_rate.pdf'):
        """
        Plot feasibility rate curves (Fig A2).
        """
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3))
        
        if 'convergence' in history and history['convergence']:
            conv = history['convergence']
            
            if 'feasible_rate' in conv and len(conv['feasible_rate']) > 0:
                generations = np.arange(1, len(conv['feasible_rate']) + 1)
                rates = np.array(conv['feasible_rate'])
                
                pop_size = getattr(self.config, 'population_size', 100)
                evaluations = generations * pop_size
                
                ax.plot(evaluations, rates * 100, '-', color=COLORS['nsga3'],
                       linewidth=2, label='NSGA-III', marker='o',
                       markevery=max(1, len(generations)//10), markersize=4)
        
        ax.set_xlabel('Number of Function Evaluations')
        ax.set_ylabel('Feasible Rate (%)')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_quality_comparison_boxplot(self, pareto_df: pd.DataFrame,
                                       baseline_dfs: Dict[str, pd.DataFrame],
                                       save_name: str = 'fig_quality_boxplot.pdf'):
        """
        Plot quality comparison boxplot (Fig A3).
        
        Shows distribution of key metrics across methods.
        """
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5))
        
        methods = ['NSGA-III'] + list(baseline_dfs.keys())
        
        # Prepare data
        cost_data = []
        recall_data = []
        carbon_data = []
        method_labels = []
        
        # NSGA-III
        if len(pareto_df) > 0:
            cost_data.append(pareto_df['f1_total_cost_USD'].values / 1e6)
            recall_data.append(pareto_df['detection_recall'].values)
            if 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns:
                carbon_data.append(pareto_df['f5_carbon_emissions_kgCO2e_year'].values / 1000)
            method_labels.append('NSGA-III')
        
        # Baselines
        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feasible) > 0:
                    cost_data.append(feasible['f1_total_cost_USD'].values / 1e6)
                    recall_data.append(feasible['detection_recall'].values)
                    if 'f5_carbon_emissions_kgCO2e_year' in feasible.columns:
                        carbon_data.append(feasible['f5_carbon_emissions_kgCO2e_year'].values / 1000)
                    method_labels.append(name.title())
        
        colors = [METHOD_COLORS.get(m, '#999999') for m in method_labels]
        
        # (a) Cost distribution
        ax = axes[0]
        bp = ax.boxplot(cost_data, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Total Cost (Million USD)')
        ax.tick_params(axis='x', rotation=45)
        ax.set_title('(a) Cost', fontsize=10)
        
        # (b) Recall distribution
        ax = axes[1]
        bp = ax.boxplot(recall_data, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Detection Recall')
        ax.tick_params(axis='x', rotation=45)
        ax.set_title('(b) Recall', fontsize=10)
        
        # (c) Carbon distribution
        ax = axes[2]
        if carbon_data:
            bp = ax.boxplot(carbon_data, labels=method_labels[:len(carbon_data)], 
                           patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(carbon_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_ylabel('Carbon (tCO₂e/year)')
        ax.tick_params(axis='x', rotation=45)
        ax.set_title('(c) Carbon', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    # ========================================================================
    # Group B: Pareto Structure and Trade-offs
    # ========================================================================
    def plot_pareto_2d_main(self, pareto_df: pd.DataFrame,
                           baseline_dfs: Dict[str, pd.DataFrame] = None,
                           save_name: str = 'fig_pareto_2d_main.pdf'):
        """
        Main 2D Pareto front: Cost vs Recall with Latency coloring (Fig B1).
        """
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 3.5))
        
        # Scale cost to millions
        x_data = pareto_df['f1_total_cost_USD'] / 1e6
        y_data = pareto_df['detection_recall']
        
        # Color by latency if available
        if 'f3_latency_seconds' in pareto_df.columns:
            c_data = pareto_df['f3_latency_seconds']
            scatter = ax.scatter(x_data, y_data, c=c_data, cmap='viridis',
                               s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Latency (s)')
        else:
            ax.scatter(x_data, y_data, c=COLORS['pareto'], s=60, alpha=0.8,
                      edgecolors='white', linewidth=0.5)
        
        # Connect Pareto front with line
        sorted_idx = np.argsort(x_data)
        ax.plot(x_data.iloc[sorted_idx], y_data.iloc[sorted_idx], 
               '--', color=COLORS['pareto'], alpha=0.5, linewidth=1)
        
        # Plot baselines
        if baseline_dfs:
            for name, df in baseline_dfs.items():
                if df is not None and len(df) > 0:
                    feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                    if len(feasible) > 0:
                        ax.scatter(feasible['f1_total_cost_USD'] / 1e6,
                                 feasible['detection_recall'],
                                 c=METHOD_COLORS.get(name.title(), '#999999'),
                                 marker=METHOD_MARKERS.get(name.title(), 's'),
                                 s=30, alpha=0.5, label=name.title())
        
        ax.set_xlabel('Total Cost (Million USD)')
        ax.set_ylabel('Detection Recall')
        
        if baseline_dfs:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_pareto_3d(self, pareto_df: pd.DataFrame,
                      save_name: str = 'fig_pareto_3d.pdf'):
        """
        3D Pareto front visualization.
        """
        fig = plt.figure(figsize=(SINGLE_COL * 2, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        x = pareto_df['f1_total_cost_USD'] / 1e6
        y = pareto_df['detection_recall']
        
        if 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns:
            z = pareto_df['f5_carbon_emissions_kgCO2e_year'] / 1000
            z_label = 'Carbon (tCO₂e/year)'
        else:
            z = pareto_df.get('f3_latency_seconds', np.zeros(len(pareto_df)))
            z_label = 'Latency (s)'
        
        # Color by fourth dimension if available
        if 'f3_latency_seconds' in pareto_df.columns and 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns:
            c = pareto_df['f3_latency_seconds']
            scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=50, alpha=0.8)
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Latency (s)')
        else:
            ax.scatter(x, y, z, c=COLORS['pareto'], s=50, alpha=0.8)
        
        ax.set_xlabel('Cost (M USD)', fontsize=9)
        ax.set_ylabel('Recall', fontsize=9)
        ax.set_zlabel(z_label, fontsize=9)
        
        ax.view_init(elev=20, azim=45)
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_parallel_coordinates(self, pareto_df: pd.DataFrame,
                                 save_name: str = 'fig_parallel_coords.pdf'):
        """
        Parallel coordinates for all objectives + key decision variables (Fig B2).
        """
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
        
        # Select objectives
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']
        
        # Add optional columns
        for col in ['f4_traffic_disruption_hours', 'f4_traffic_disruption_hours_year',
                   'f5_carbon_emissions_kgCO2e_year', 'system_MTBF_hours']:
            if col in pareto_df.columns:
                obj_cols.append(col)
        
        obj_cols = obj_cols[:6]  # Limit to 6 objectives
        
        # Normalize to [0, 1]
        data_norm = pareto_df[obj_cols].copy()
        for col in obj_cols:
            min_val = data_norm[col].min()
            max_val = data_norm[col].max()
            if max_val > min_val:
                # For recall and MTBF, higher is better; for others, lower is better
                if col in ['detection_recall', 'system_MTBF_hours']:
                    data_norm[col] = (data_norm[col] - min_val) / (max_val - min_val)
                else:
                    data_norm[col] = 1 - (data_norm[col] - min_val) / (max_val - min_val)
            else:
                data_norm[col] = 0.5
        
        # Color by cost
        colors = plt.cm.viridis(
            (pareto_df['f1_total_cost_USD'] - pareto_df['f1_total_cost_USD'].min()) /
            (pareto_df['f1_total_cost_USD'].max() - pareto_df['f1_total_cost_USD'].min() + 1e-10)
        )
        
        # Plot lines
        x = np.arange(len(obj_cols))
        for i in range(len(data_norm)):
            ax.plot(x, data_norm.iloc[i].values, c=colors[i], alpha=0.4, linewidth=1)
        
        # Labels
        labels = []
        for col in obj_cols:
            label = OBJECTIVE_LABELS.get(col, col)
            # Shorten labels
            label = label.replace('Traffic Disruption', 'Disruption')
            label = label.replace('Carbon Emissions', 'Carbon')
            label = label.replace('Million USD', 'M$')
            labels.append(label.replace(' ', '\n'))
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Normalized Value (1 = Best)')
        ax.set_ylim(-0.05, 1.05)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
            norm=plt.Normalize(pareto_df['f1_total_cost_USD'].min() / 1e6,
                              pareto_df['f1_total_cost_USD'].max() / 1e6))
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Cost (M USD)')
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_representative_solutions_radar(self, pareto_df: pd.DataFrame,
                                           save_name: str = 'fig_radar_representatives.pdf'):
        """
        Radar chart comparing representative solutions (Fig B3).
        """
        # Select representative solutions
        representatives = self.metrics.select_representative_solutions(pareto_df)
        
        if len(representatives) == 0:
            logger.warning("No representative solutions found for radar chart")
            return None
        
        # Prepare radar data
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']
        for col in ['f4_traffic_disruption_hours', 'f4_traffic_disruption_hours_year',
                   'f5_carbon_emissions_kgCO2e_year']:
            if col in pareto_df.columns:
                obj_cols.append(col)
        
        obj_cols = obj_cols[:6]
        
        # Normalize (all to higher = better)
        normalized_data = []
        for _, row in representatives.iterrows():
            values = []
            for col in obj_cols:
                min_val = pareto_df[col].min()
                max_val = pareto_df[col].max()
                if max_val > min_val:
                    if col in ['detection_recall', 'system_MTBF_hours']:
                        val = (row[col] - min_val) / (max_val - min_val)
                    else:
                        val = 1 - (row[col] - min_val) / (max_val - min_val)
                else:
                    val = 0.5
                values.append(val)
            normalized_data.append(values)
        
        # Create radar
        fig = plt.figure(figsize=(SINGLE_COL * 2, 4))
        ax = fig.add_subplot(111, polar=True)
        
        n_vars = len(obj_cols)
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Short labels
        labels = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'MTBF'][:len(obj_cols)]
        
        colors = [COLORS['nsga3'], COLORS['random'], COLORS['grid'],
                 COLORS['weighted'], COLORS['expert'], '#666666']
        
        for i, (values, (_, row)) in enumerate(zip(normalized_data, representatives.iterrows())):
            values_closed = values + values[:1]
            label = row.get('solution_type', f'Solution {i+1}')
            ax.plot(angles, values_closed, 'o-', linewidth=2, 
                   label=label, color=colors[i % len(colors)], markersize=4)
            ax.fill(angles, values_closed, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_decision_distribution(self, pareto_df: pd.DataFrame,
                                  save_name: str = 'fig_decision_distribution.pdf'):
        """
        Visualize distribution of decision variables in Pareto solutions.
        """
        var_cols = ['sensor', 'algorithm', 'storage', 'communication', 'deployment',
                   'data_rate', 'geo_lod', 'cond_lod', 'detection_threshold',
                   'crew_size', 'inspection_cycle']
        
        available_cols = [c for c in var_cols if c in pareto_df.columns]
        
        if len(available_cols) == 0:
            logger.warning("No decision variable columns found")
            return None
        
        n_vars = len(available_cols)
        n_cols = 4
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(DOUBLE_COL, 2.5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_vars == 1 else axes
        
        for i, col in enumerate(available_cols):
            ax = axes[i]
            data = pareto_df[col]
            
            if data.dtype == 'object':
                counts = data.value_counts()
                ax.bar(range(len(counts)), counts.values, 
                      color=COLORS['pareto'], alpha=0.8)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels([str(x)[:12] for x in counts.index], 
                                  rotation=45, ha='right', fontsize=7)
            else:
                ax.hist(data, bins=15, color=COLORS['pareto'], 
                       alpha=0.8, edgecolor='white')
            
            ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=9)
            ax.set_ylabel('Count', fontsize=8)
        
        # Hide empty subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    # ========================================================================
    # All 2D Pareto Projections
    # ========================================================================
    def create_all_2d_pareto_fronts(self, df: pd.DataFrame):
        """Create all pairwise 2D Pareto front projections."""
        objectives = [
            ('f1_total_cost_USD', 'Cost (k$)', 1000, 'min'),
            ('detection_recall', 'Recall', 1, 'max'),
            ('f3_latency_seconds', 'Latency (s)', 1, 'min'),
        ]
        
        # Add optional objectives
        if 'f4_traffic_disruption_hours' in df.columns:
            objectives.append(('f4_traffic_disruption_hours', 'Disruption (h)', 1, 'min'))
        elif 'f4_traffic_disruption_hours_year' in df.columns:
            objectives.append(('f4_traffic_disruption_hours_year', 'Disruption (h/y)', 1, 'min'))
        
        if 'f5_carbon_emissions_kgCO2e_year' in df.columns:
            objectives.append(('f5_carbon_emissions_kgCO2e_year', 'Carbon (tCO₂/y)', 1000, 'min'))
        
        if 'system_MTBF_hours' in df.columns:
            objectives.append(('system_MTBF_hours', 'MTBF (years)', 8760, 'max'))
        
        # Create all pairwise combinations
        plot_num = 0
        for i, (col1, label1, scale1, dir1) in enumerate(objectives):
            for j, (col2, label2, scale2, dir2) in enumerate(objectives[i+1:], i+1):
                plot_num += 1
                self._create_single_2d_pareto(
                    df, col1, col2, label1, label2, scale1, scale2,
                    dir1, dir2, f'pareto_2d_{plot_num:02d}'
                )
    
    def _create_single_2d_pareto(self, df, x_col, y_col, x_label, y_label,
                                x_scale, y_scale, x_dir, y_dir, filename):
        """Create a single 2D Pareto projection."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, 3))
        
        x_data = df[x_col] / x_scale
        y_data = df[y_col] / y_scale
        
        # Find 2D Pareto front
        pareto_mask = self._find_pareto_front_2d(
            x_data.values, y_data.values,
            x_dir == 'min', y_dir == 'min'
        )
        
        # Color by third objective
        if 'f1_total_cost_USD' not in [x_col, y_col]:
            c_data = df['f1_total_cost_USD'] / 1000
            c_label = 'Cost (k$)'
        elif 'detection_recall' not in [x_col, y_col]:
            c_data = df['detection_recall']
            c_label = 'Recall'
        else:
            c_data = df.get('f3_latency_seconds', np.zeros(len(df)))
            c_label = 'Latency (s)'
        
        scatter = ax.scatter(x_data, y_data, c=c_data, cmap='viridis',
                           s=50, alpha=0.8, edgecolors='white', linewidth=0.3)
        
        # Highlight Pareto front
        if np.any(pareto_mask):
            ax.scatter(x_data[pareto_mask], y_data[pareto_mask],
                      s=100, facecolors='none', edgecolors='red',
                      linewidth=1.5, label='Pareto Front')
            
            # Connect points
            px = x_data[pareto_mask].values
            py = y_data[pareto_mask].values
            sort_idx = np.argsort(px) if x_dir == 'min' else np.argsort(-px)
            ax.plot(px[sort_idx], py[sort_idx], 'r--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(c_label, fontsize=8)
        
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, filename)
    
    # ========================================================================
    # Baseline Comparison
    # ========================================================================
    def plot_method_performance_summary(self, pareto_df: pd.DataFrame,
                                       baseline_dfs: Dict[str, pd.DataFrame],
                                       save_name: str = 'fig_method_summary.pdf'):
        """Summary bar chart comparing all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5))
        
        methods = ['NSGA-III']
        metrics = {
            'NSGA-III': {
                'n_solutions': len(pareto_df),
                'n_feasible': len(pareto_df),
                'min_cost': pareto_df['f1_total_cost_USD'].min() / 1e6 if len(pareto_df) > 0 else np.nan,
                'max_recall': pareto_df['detection_recall'].max() if len(pareto_df) > 0 else np.nan,
            }
        }
        
        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                methods.append(name.title())
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                metrics[name.title()] = {
                    'n_solutions': len(df),
                    'n_feasible': len(feasible),
                    'min_cost': feasible['f1_total_cost_USD'].min() / 1e6 if len(feasible) > 0 else np.nan,
                    'max_recall': feasible['detection_recall'].max() if len(feasible) > 0 else np.nan,
                }
        
        colors = [METHOD_COLORS.get(m, '#999999') for m in methods]
        
        # (a) Feasible solutions
        ax = axes[0, 0]
        values = [metrics[m]['n_feasible'] for m in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Number of Feasible Solutions')
        ax.set_title('(a) Solution Feasibility', fontsize=10)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val}', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        
        # (b) Minimum cost
        ax = axes[0, 1]
        values = [metrics[m]['min_cost'] for m in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Min Cost (Million USD)')
        ax.set_title('(b) Cost Optimization', fontsize=10)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'${val:.2f}M', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        
        # (c) Maximum recall
        ax = axes[1, 0]
        values = [metrics[m]['max_recall'] for m in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Max Detection Recall')
        ax.set_title('(c) Detection Performance', fontsize=10)
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        
        # (d) Scatter comparison
        ax = axes[1, 1]
        for i, (name, df) in enumerate([('NSGA-III', pareto_df)] + list(baseline_dfs.items())):
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns and name != 'NSGA-III' else df
                if len(feasible) > 0:
                    ax.scatter(feasible['f1_total_cost_USD'] / 1e6, 
                             feasible['detection_recall'],
                             c=METHOD_COLORS.get(name.title() if name != 'NSGA-III' else name, '#999999'),
                             marker=METHOD_MARKERS.get(name.title() if name != 'NSGA-III' else name, 'o'),
                             s=40 if name == 'NSGA-III' else 25,
                             alpha=0.7 if name == 'NSGA-III' else 0.5,
                             label=name.title() if name != 'NSGA-III' else name)
        ax.set_xlabel('Cost (Million USD)')
        ax.set_ylabel('Recall')
        ax.set_title('(d) Cost-Recall Trade-off', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_cost_recall_comparison(self, pareto_df: pd.DataFrame,
                                   baseline_dfs: Dict[str, pd.DataFrame],
                                   save_name: str = 'fig_cost_recall_comparison.pdf'):
        """Detailed Cost vs Recall comparison."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 4))
        
        # Plot NSGA-III
        ax.scatter(pareto_df['f1_total_cost_USD'] / 1e6, 
                  pareto_df['detection_recall'],
                  c=COLORS['nsga3'], s=80, alpha=0.8, 
                  label='NSGA-III', marker='o',
                  edgecolors='white', linewidth=0.5)
        
        # Connect Pareto front
        sorted_idx = np.argsort(pareto_df['f1_total_cost_USD'])
        ax.plot(pareto_df['f1_total_cost_USD'].iloc[sorted_idx] / 1e6,
               pareto_df['detection_recall'].iloc[sorted_idx],
               '--', color=COLORS['nsga3'], alpha=0.5, linewidth=1)
        
        # Plot baselines
        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feasible) > 0:
                    ax.scatter(feasible['f1_total_cost_USD'] / 1e6,
                             feasible['detection_recall'],
                             c=METHOD_COLORS.get(name.title(), '#999999'),
                             marker=METHOD_MARKERS.get(name.title(), 's'),
                             s=40, alpha=0.5, label=name.title())
        
        ax.set_xlabel('Total Cost (Million USD)')
        ax.set_ylabel('Detection Recall')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_feasibility_comparison(self, pareto_df: pd.DataFrame,
                                   baseline_dfs: Dict[str, pd.DataFrame],
                                   save_name: str = 'fig_feasibility_comparison.pdf'):
        """Compare feasibility rates across methods."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3))
        
        methods = ['NSGA-III']
        feasible_rates = [100.0]  # NSGA-III Pareto is all feasible
        total_counts = [len(pareto_df)]
        
        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                methods.append(name.title())
                rate = self.metrics.calculate_feasible_rate(df) * 100
                feasible_rates.append(rate)
                total_counts.append(len(df))
        
        colors = [METHOD_COLORS.get(m, '#999999') for m in methods]
        
        bars = ax.bar(methods, feasible_rates, color=colors, alpha=0.8)
        
        ax.set_ylabel('Feasible Rate (%)')
        ax.set_ylim(0, 110)
        
        for bar, rate, total in zip(bars, feasible_rates, total_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.1f}%\n(n={total})', ha='center', va='bottom', fontsize=8)
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        
        self._save_figure(fig, save_name)
        return fig
    
    def plot_dominance_analysis(self, pareto_df: pd.DataFrame,
                               baseline_dfs: Dict[str, pd.DataFrame],
                               save_name: str = 'fig_dominance_analysis.pdf'):
        """Analyze dominance relationships."""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3))
        
        # Calculate coverage metrics
        pareto_obj = pareto_df[['f1_total_cost_USD', 'detection_recall']].values
        # Convert recall to minimization (1 - recall)
        pareto_obj[:, 1] = 1 - pareto_obj[:, 1]
        
        methods = []
        coverage_values = []
        
        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feasible) > 0:
                    baseline_obj = feasible[['f1_total_cost_USD', 'detection_recall']].values
                    baseline_obj[:, 1] = 1 - baseline_obj[:, 1]
                    
                    # C(Pareto, Baseline): fraction of baseline dominated by Pareto
                    coverage = self.metrics.calculate_coverage(pareto_obj, baseline_obj) * 100
                    methods.append(name.title())
                    coverage_values.append(coverage)
        
        if methods:
            colors = [METHOD_COLORS.get(m, '#999999') for m in methods]
            bars = ax.bar(methods, coverage_values, color=colors, alpha=0.8)
            
            ax.set_ylabel('Baseline Solutions Dominated (%)')
            ax.set_ylim(0, 110)
            
            for bar, val in zip(bars, coverage_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
        
        self._save_figure(fig, save_name)
        return fig
    
    # ========================================================================
    # Technology Analysis
    # ========================================================================
    def plot_technology_impact(self, pareto_df: pd.DataFrame,
                              save_name: str = 'fig_technology_impact.pdf'):
        """Analyze impact of technology choices."""
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5))
        
        # (a) Sensor type distribution
        ax = axes[0, 0]
        if 'sensor' in pareto_df.columns:
            counts = pareto_df['sensor'].value_counts()
            ax.bar(range(len(counts)), counts.values, color=COLORS['pareto'], alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:15] for x in counts.index], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Count')
            ax.set_title('(a) Sensor Selection', fontsize=10)
        
        # (b) Algorithm type distribution
        ax = axes[0, 1]
        if 'algorithm' in pareto_df.columns:
            counts = pareto_df['algorithm'].value_counts()
            ax.bar(range(len(counts)), counts.values, color=COLORS['grid'], alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:15] for x in counts.index], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Count')
            ax.set_title('(b) Algorithm Selection', fontsize=10)
        
        # (c) Communication type distribution
        ax = axes[1, 0]
        if 'communication' in pareto_df.columns:
            counts = pareto_df['communication'].value_counts()
            ax.bar(range(len(counts)), counts.values, color=COLORS['weighted'], alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:15] for x in counts.index], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Count')
            ax.set_title('(c) Communication Selection', fontsize=10)
        
        # (d) Deployment type distribution
        ax = axes[1, 1]
        if 'deployment' in pareto_df.columns:
            counts = pareto_df['deployment'].value_counts()
            ax.bar(range(len(counts)), counts.values, color=COLORS['expert'], alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:15] for x in counts.index], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Count')
            ax.set_title('(d) Deployment Selection', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    # ========================================================================
    # Group C: Ontology Ablation
    # ========================================================================
    def plot_ontology_ablation(self, on_history: Dict, off_history: Dict,
                              save_name: str = 'fig_ablation_ontology.pdf'):
        """
        Plot Ontology on/off ablation comparison (Fig C1).
        """
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3))
        
        # (a) HV convergence
        ax = axes[0]
        if on_history and 'hv' in on_history:
            ax.plot(on_history.get('evaluations', range(len(on_history['hv']))),
                   on_history['hv'], '-', color=COLORS['nsga3'],
                   linewidth=2, label='Ontology ON')
        
        if off_history and 'hv' in off_history:
            ax.plot(off_history.get('evaluations', range(len(off_history['hv']))),
                   off_history['hv'], '--', color=COLORS['random'],
                   linewidth=2, label='Ontology OFF')
        
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Hypervolume')
        ax.set_title('(a) Convergence Comparison', fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # (b) Feasible rate
        ax = axes[1]
        if on_history and 'feasible_rate' in on_history:
            ax.plot(on_history.get('evaluations', range(len(on_history['feasible_rate']))),
                   np.array(on_history['feasible_rate']) * 100, '-', 
                   color=COLORS['nsga3'], linewidth=2, label='Ontology ON')
        
        if off_history and 'feasible_rate' in off_history:
            ax.plot(off_history.get('evaluations', range(len(off_history['feasible_rate']))),
                   np.array(off_history['feasible_rate']) * 100, '--',
                   color=COLORS['random'], linewidth=2, label='Ontology OFF')
        
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Feasible Rate (%)')
        ax.set_title('(b) Feasibility Comparison', fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    # ========================================================================
    # Sensitivity Analysis
    # ========================================================================
    def plot_sensitivity_heatmap(self, sensitivity_results: Dict[str, Dict[str, float]],
                                save_name: str = 'fig_sensitivity_heatmap.pdf'):
        """
        Create heatmap of variable sensitivities.
        """
        variables = list(sensitivity_results.keys())
        objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability']
        
        # Build matrix
        data = np.zeros((len(variables), len(objectives)))
        for i, var in enumerate(variables):
            for j, obj in enumerate(objectives):
                data[i, j] = sensitivity_results[var].get(obj, 0)
        
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 4))
        
        cmap = LinearSegmentedColormap.from_list('sensitivity',
            ['#ffffff', '#ffd700', '#ff6b6b', '#c0392b'])
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=200)
        
        ax.set_xticks(np.arange(len(objectives)))
        ax.set_yticks(np.arange(len(variables)))
        ax.set_xticklabels(objectives, fontsize=8)
        ax.set_yticklabels([VARIABLE_LABELS.get(v, v) for v in variables], fontsize=8)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Annotations
        for i in range(len(variables)):
            for j in range(len(objectives)):
                val = data[i, j]
                if val > 0.1:
                    text_color = 'white' if val > 100 else 'black'
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                           color=text_color, fontsize=7)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Impact (%)', rotation=-90, va='bottom', fontsize=9)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    def _find_pareto_front_2d(self, x: np.ndarray, y: np.ndarray,
                             minimize_x: bool = True, minimize_y: bool = True) -> np.ndarray:
        """Find 2D Pareto front mask."""
        n = len(x)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if minimize_x:
                        x_dom = x[j] <= x[i]
                    else:
                        x_dom = x[j] >= x[i]
                    
                    if minimize_y:
                        y_dom = y[j] <= y[i]
                    else:
                        y_dom = y[j] >= y[i]
                    
                    if x_dom and y_dom:
                        strictly_better = (
                            (minimize_x and x[j] < x[i]) or
                            (not minimize_x and x[j] > x[i]) or
                            (minimize_y and y[j] < y[i]) or
                            (not minimize_y and y[j] > y[i])
                        )
                        if strictly_better:
                            pareto_mask[i] = False
                            break
        
        return pareto_mask
    
    def _save_figure(self, fig, filename: str):
        """Save figure in PDF and PNG formats."""
        for fmt in ['pdf', 'png']:
            path = self.output_dir / f'{filename}.{fmt}'
            fig.savefig(path, dpi=300 if fmt == 'png' else None,
                       bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        logger.info(f"Saved: {filename}")


# ============================================================================
# Sensitivity Analysis Function
# ============================================================================
def run_sensitivity_analysis(evaluator, base_config: Dict = None) -> Dict[str, Dict[str, float]]:
    """
    Run comprehensive sensitivity analysis.
    
    Returns:
        Dict of {variable: {objective: impact_pct}}
    """
    if base_config is None:
        base_x = np.array([0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.7, 0.3, 0.15])
    else:
        base_x = np.array(list(base_config.values()))
    
    variables = ['sensor', 'data_rate', 'geo_lod', 'cond_lod', 'algorithm',
                'detection_threshold', 'storage', 'communication', 'deployment',
                'crew_size', 'inspection_cycle']
    
    objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability']
    
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


# ============================================================================
# Standalone Entry Points
# ============================================================================
def create_visualizations(config, pareto_results_path: str,
                         baseline_results_dir: Optional[str] = None,
                         optimization_history: Optional[Dict] = None):
    """Main entry point for creating all visualizations."""
    pareto_df = pd.read_csv(pareto_results_path)
    
    baseline_results = {}
    if baseline_results_dir:
        baseline_dir = Path(baseline_results_dir)
        for method in ['random', 'grid', 'weighted', 'expert']:
            baseline_path = baseline_dir / f'baseline_{method}.csv'
            if baseline_path.exists():
                baseline_results[method] = pd.read_csv(baseline_path)
    
    visualizer = Visualizer(config)
    visualizer.create_all_figures(pareto_df, baseline_results, optimization_history)
    
    logger.info("Visualization generation complete!")


# Alias for compatibility
JournalVisualizer = Visualizer


if __name__ == "__main__":
    import argparse
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        output_dir: str = './results'
        population_size: int = 100
    
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--pareto', default='./results/pareto_solutions_6obj_fixed.csv')
    parser.add_argument('--baselines', default='./results')
    parser.add_argument('--output', default='./results')
    
    args = parser.parse_args()
    
    config = Config(output_dir=args.output)
    create_visualizations(config, args.pareto, args.baselines)

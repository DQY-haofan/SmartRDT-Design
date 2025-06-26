#!/usr/bin/env python3
"""
Advanced Visualizations for Enhanced RMTwin Multi-Objective Optimization
Updated for 6 objectives including sustainability metrics
Compatible with Expert-Enhanced Ontology-Driven Multi-Objective Optimization Framework.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality (matching main framework)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ============================================================================
# MAIN VISUALIZATION FUNCTIONS (Updated for 6 objectives)
# ============================================================================

def create_enhanced_3d_pareto_visualization(df, output_dir='./results'):
    """Create multiple 3D visualizations showcasing different objective combinations"""
    
    # Ensure output directories exist
    os.makedirs(f'{output_dir}/png', exist_ok=True)
    os.makedirs(f'{output_dir}/pdf', exist_ok=True)
    
    # Define interesting 3D combinations
    combinations = [
        # (x, y, z, color_by, title)
        ('f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds', 'f5_environmental_impact_kWh_year',
         'Performance-Cost-Efficiency Trade-off\n(Color: Environmental Impact)'),
        ('f1_total_cost_USD', 'f5_environmental_impact_kWh_year', 'system_MTBF_hours', 'detection_recall',
         'Sustainability-Reliability-Cost Trade-off\n(Color: Detection Performance)'),
        ('detection_recall', 'f3_latency_seconds', 'f4_traffic_disruption_hours', 'f1_total_cost_USD',
         'Operational Performance Trade-offs\n(Color: Total Cost)'),
        ('f5_environmental_impact_kWh_year', 'system_MTBF_hours', 'detection_recall', 'f1_total_cost_USD',
         'Sustainability-Reliability-Performance\n(Color: Total Cost)')
    ]
    
    for idx, (x_col, y_col, z_col, color_col, title) in enumerate(combinations):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scale data for better visualization
        x_data = df[x_col] / 1000 if 'cost' in x_col.lower() else df[x_col]
        y_data = df[y_col] / 1000 if 'environmental' in y_col.lower() else df[y_col]
        z_data = df[z_col] / 1000 if 'MTBF' in z_col else df[z_col]
        color_data = df[color_col] / 1000 if 'cost' in color_col.lower() or 'environmental' in color_col.lower() else df[color_col]
        
        # Create scatter plot
        scatter = ax.scatter(x_data, y_data, z_data,
                           c=color_data, cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Labels
        x_label = x_col.replace('_', ' ').replace('f1 ', '').replace('USD', '(k$)')
        y_label = y_col.replace('_', ' ').replace('f5 ', '').replace('kWh year', '(MWh/year)')
        z_label = z_col.replace('_', ' ').replace('f3 ', '').replace('hours', '(k hours)')
        color_label = color_col.replace('_', ' ').replace('f1 ', '').replace('f5 ', '')
        
        ax.set_xlabel(x_label, fontsize=14, labelpad=10)
        ax.set_ylabel(y_label, fontsize=14, labelpad=10)
        ax.set_zlabel(z_label, fontsize=14, labelpad=10)
        ax.set_title(title, fontsize=16, pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(color_label, fontsize=12, rotation=270, labelpad=20)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Save
        plt.tight_layout()
        fig.savefig(f'{output_dir}/png/3d_pareto_{idx+1}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_dir}/pdf/3d_pareto_{idx+1}.pdf', bbox_inches='tight')
        plt.close(fig)
    
    print(f"Created {len(combinations)} 3D Pareto visualizations")

def create_decision_variable_impact_analysis(df, output_dir='./results'):
    """Analyze the impact of each decision variable on all 6 objectives"""
    
    # Create figure with subplots for each objective
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Decision variables to analyze
    decision_vars = [
        ('sensor', 'Sensor Type', True),
        ('algorithm', 'Algorithm', True),
        ('storage', 'Storage System', True),
        ('communication', 'Communication', True),
        ('deployment', 'Deployment', True),
        ('geometric_LOD', 'Geometric LOD', True),
        ('condition_LOD', 'Condition LOD', True),
        ('crew_size', 'Crew Size', False),
        ('inspection_cycle_days', 'Inspection Cycle (days)', False),
        ('data_rate_Hz', 'Data Rate (Hz)', False),
        ('detection_threshold', 'Detection Threshold', False)
    ]
    
    # Objectives to analyze
    objectives = [
        ('f1_total_cost_USD', 'Total Cost ($)', 1000),
        ('detection_recall', 'Detection Recall', 1),
        ('f3_latency_seconds', 'Latency (s)', 1),
        ('f4_traffic_disruption_hours', 'Traffic Disruption (h)', 1),
        ('f5_environmental_impact_kWh_year', 'Environmental Impact (kWh/y)', 1000),
        ('system_MTBF_hours', 'System MTBF (hours)', 1000)
    ]
    
    for var_idx, (var, var_label, is_categorical) in enumerate(decision_vars):
        if var_idx >= 12:  # Only show first 12
            break
            
        ax = fig.add_subplot(gs[var_idx // 3, var_idx % 3])
        
        if var in df.columns:
            if is_categorical:
                # Create grouped data
                grouped_data = []
                labels = []
                
                for obj, obj_label, scale in objectives[:4]:  # Show first 4 objectives
                    data_by_category = []
                    for category in df[var].unique():
                        values = df[df[var] == category][obj] / scale
                        data_by_category.append(values)
                    grouped_data.append(data_by_category)
                    labels.append(obj_label.split('(')[0].strip())
                
                # Create positions for grouped bars
                categories = [str(cat).split('/')[-1][:15] for cat in df[var].unique()]
                x = np.arange(len(categories))
                width = 0.2
                
                # Plot grouped bars
                for i, (data, label) in enumerate(zip(grouped_data, labels)):
                    means = [d.mean() for d in data]
                    ax.bar(x + i*width - 1.5*width, means, width, label=label, alpha=0.8)
                
                ax.set_xlabel(var_label, fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha='right')
                ax.legend(fontsize=10, loc='upper right')
                ax.grid(True, alpha=0.3)
                
            else:
                # Continuous variable - create 2x2 scatter plots
                for i, (obj, obj_label, scale) in enumerate(objectives[:4]):
                    color = plt.cm.tab10(i)
                    ax.scatter(df[var], df[obj]/scale, alpha=0.5, s=30, 
                             label=obj_label.split('(')[0].strip(), color=color)
                
                ax.set_xlabel(var_label, fontsize=12)
                ax.set_ylabel('Normalized Objective Values', fontsize=12)
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
        
        ax.set_title(f'Impact of {var_label}', fontsize=14)
    
    plt.suptitle('Decision Variable Impact Analysis on Multiple Objectives', fontsize=20)
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{output_dir}/png/decision_variable_impact_6obj.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/decision_variable_impact_6obj.pdf', bbox_inches='tight')
    plt.close(fig)

def create_technology_comparison_dashboard(df, output_dir='./results'):
    """Create comprehensive technology comparison for all 6 objectives"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Group by sensor technology
    sensor_groups = df.groupby('sensor').agg({
        'f1_total_cost_USD': ['mean', 'std', 'count'],
        'detection_recall': ['mean', 'std'],
        'f3_latency_seconds': ['mean', 'std'],
        'f4_traffic_disruption_hours': ['mean', 'std'],
        'f5_environmental_impact_kWh_year': ['mean', 'std'],
        'system_MTBF_hours': ['mean', 'std']
    })
    
    # Sort by cost
    sensor_groups = sensor_groups.sort_values(('f1_total_cost_USD', 'mean'))
    
    # 1. Cost comparison
    ax1 = fig.add_subplot(gs[0, 0])
    sensor_names = [s.split('/')[-1] for s in sensor_groups.index]
    y_pos = np.arange(len(sensor_names))
    
    ax1.barh(y_pos, sensor_groups[('f1_total_cost_USD', 'mean')]/1000,
            xerr=sensor_groups[('f1_total_cost_USD', 'std')]/1000,
            color='skyblue', capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sensor_names)
    ax1.set_xlabel('Average Total Cost (k$)', fontsize=12)
    ax1.set_title('Cost by Sensor Technology', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add count annotations
    for i, count in enumerate(sensor_groups[('f1_total_cost_USD', 'count')]):
        ax1.text(5, i, f'n={count}', fontsize=10, va='center')
    
    # 2. Performance comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(y_pos, sensor_groups[('detection_recall', 'mean')],
            xerr=sensor_groups[('detection_recall', 'std')],
            color='lightgreen', capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sensor_names)
    ax2.set_xlabel('Average Detection Recall', fontsize=12)
    ax2.set_title('Performance by Sensor Technology', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 1)
    
    # 3. Environmental impact
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.barh(y_pos, sensor_groups[('f5_environmental_impact_kWh_year', 'mean')]/1000,
            xerr=sensor_groups[('f5_environmental_impact_kWh_year', 'std')]/1000,
            color='lightcoral', capsize=5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sensor_names)
    ax3.set_xlabel('Average Annual Energy (MWh)', fontsize=12)
    ax3.set_title('Environmental Impact by Sensor', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Reliability comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.barh(y_pos, sensor_groups[('system_MTBF_hours', 'mean')]/8760,
            xerr=sensor_groups[('system_MTBF_hours', 'std')]/8760,
            color='gold', capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(sensor_names)
    ax4.set_xlabel('Average System MTBF (years)', fontsize=12)
    ax4.set_title('Reliability by Sensor Technology', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Algorithm comparison
    ax5 = fig.add_subplot(gs[1, 1:])
    algo_groups = df.groupby('algorithm').agg({
        'detection_recall': 'mean',
        'f3_latency_seconds': 'mean',
        'f5_environmental_impact_kWh_year': 'mean'
    }).sort_values('detection_recall', ascending=False)
    
    algo_names = [a.split('/')[-1][:20] for a in algo_groups.index[:10]]  # Top 10
    x_pos = np.arange(len(algo_names))
    
    ax5_twin = ax5.twinx()
    
    # Bar plot for recall
    bars = ax5.bar(x_pos, algo_groups['detection_recall'].head(10), 
                   alpha=0.6, color='blue', label='Detection Recall')
    
    # Line plot for latency
    line = ax5_twin.plot(x_pos, algo_groups['f3_latency_seconds'].head(10), 
                        'ro-', linewidth=2, markersize=8, label='Latency (s)')
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(algo_names, rotation=45, ha='right')
    ax5.set_ylabel('Detection Recall', fontsize=12)
    ax5_twin.set_ylabel('Latency (seconds)', fontsize=12)
    ax5.set_title('Algorithm Performance Comparison', fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 6. Multi-objective radar chart for top solutions
    ax6 = fig.add_subplot(gs[2, :], projection='polar')
    
    # Get top 5 solutions by different criteria
    top_cheap = df.nsmallest(1, 'f1_total_cost_USD').iloc[0]
    top_perf = df.nlargest(1, 'detection_recall').iloc[0]
    top_green = df.nsmallest(1, 'f5_environmental_impact_kWh_year').iloc[0]
    top_reliable = df.nlargest(1, 'system_MTBF_hours').iloc[0]
    
    solutions = [
        ('Lowest Cost', top_cheap),
        ('Best Performance', top_perf),
        ('Most Sustainable', top_green),
        ('Most Reliable', top_reliable)
    ]
    
    # Radar chart setup
    objectives = ['Cost\n(inverse)', 'Recall', 'Speed\n(inverse)', 
                 'Low\nDisruption', 'Energy\nEfficiency', 'Reliability']
    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot each solution
    for label, sol in solutions:
        values = []
        # Normalize values (1 = best, 0 = worst)
        values.append(1 - (sol['f1_total_cost_USD'] - df['f1_total_cost_USD'].min()) /
                     (df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min()))
        values.append((sol['detection_recall'] - df['detection_recall'].min()) /
                     (df['detection_recall'].max() - df['detection_recall'].min()))
        values.append(1 - (sol['f3_latency_seconds'] - df['f3_latency_seconds'].min()) /
                     (df['f3_latency_seconds'].max() - df['f3_latency_seconds'].min()))
        values.append(1 - (sol['f4_traffic_disruption_hours'] - df['f4_traffic_disruption_hours'].min()) /
                     (df['f4_traffic_disruption_hours'].max() - df['f4_traffic_disruption_hours'].min()))
        values.append(1 - (sol['f5_environmental_impact_kWh_year'] - df['f5_environmental_impact_kWh_year'].min()) /
                     (df['f5_environmental_impact_kWh_year'].max() - df['f5_environmental_impact_kWh_year'].min()))
        values.append((sol['system_MTBF_hours'] - df['system_MTBF_hours'].min()) /
                     (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min()))
        
        values += values[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, label=label)
        ax6.fill(angles, values, alpha=0.15)
    
    ax6.set_theta_offset(np.pi / 2)
    ax6.set_theta_direction(-1)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(objectives, fontsize=11)
    ax6.set_ylim(0, 1)
    ax6.set_title('Multi-Objective Comparison of Extreme Solutions', fontsize=14, pad=30)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Technology Comparison Dashboard - 6 Objectives', fontsize=20, y=0.98)
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{output_dir}/png/technology_comparison_6obj.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/technology_comparison_6obj.pdf', bbox_inches='tight')
    plt.close(fig)


def create_objective_correlation_heatmap(df, output_dir='./results'):
    """Create correlation heatmap for all 6 objectives"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Select objective columns
    objective_cols = [
        'f1_total_cost_USD',
        'detection_recall',
        'f3_latency_seconds',
        'f4_traffic_disruption_hours',
        'f5_environmental_impact_kWh_year',
        'system_MTBF_hours'
    ]
    
    # Create correlation matrix
    corr_matrix = df[objective_cols].corr()
    
    # Rename for display
    display_names = [
        'Total Cost',
        'Detection Recall',
        'Latency',
        'Traffic Disruption',
        'Environmental Impact',
        'System MTBF'
    ]
    
    corr_matrix.index = display_names
    corr_matrix.columns = display_names
    
    # 1. Standard correlation heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title('Objective Correlation Matrix', fontsize=16)
    
    # 2. Clustered heatmap with dendrogram
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Calculate distance matrix and perform hierarchical clustering
    # Convert correlation to distance
    distance_matrix = 1 - np.abs(corr_matrix.values)
    
    # COMPLETE FIX: Ensure perfect symmetry and valid distance matrix
    n = distance_matrix.shape[0]
    
    # Method 1: Create a perfectly symmetric matrix from upper triangle
    distance_matrix_clean = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_matrix_clean[i, j] = 0.0
            else:
                # Use average of both values to ensure symmetry
                avg_dist = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
                distance_matrix_clean[i, j] = avg_dist
                distance_matrix_clean[j, i] = avg_dist
    
    # Ensure all values are valid (between 0 and 1 for correlation-based distances)
    distance_matrix_clean = np.clip(distance_matrix_clean, 0, 1)
    
    # Round to avoid floating point precision issues
    distance_matrix_clean = np.round(distance_matrix_clean, decimals=10)
    
    # Verify symmetry
    if not np.allclose(distance_matrix_clean, distance_matrix_clean.T):
        print("Warning: Matrix still not perfectly symmetric, forcing symmetry...")
        distance_matrix_clean = (distance_matrix_clean + distance_matrix_clean.T) / 2
    
    try:
        # Method 1: Try with checks
        condensed_distances = squareform(distance_matrix_clean, checks=True)
    except ValueError:
        # Method 2: If that fails, manually create condensed form
        print("Using manual condensed form creation...")
        condensed_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed_distances.append(distance_matrix_clean[i, j])
        condensed_distances = np.array(condensed_distances)
    
    # Perform clustering
    try:
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Create dendrogram
        dendro = dendrogram(linkage_matrix, labels=display_names, ax=ax2, 
                           orientation='top', color_threshold=0)
        ax2.set_title('Objective Clustering Dendrogram', fontsize=16)
        ax2.set_ylabel('Distance (1 - |correlation|)', fontsize=12)
        
        # Reorder correlation matrix based on clustering
        order = dendro['leaves']
        clustered_corr = corr_matrix.iloc[order, order]
        
        # Create inset for clustered heatmap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax2, width="40%", height="40%", loc='lower right',
                          bbox_to_anchor=(0.05, 0.05, 0.9, 0.4),
                          bbox_transform=ax2.transAxes)
        
        sns.heatmap(clustered_corr, annot=False, cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True,
                    cbar=False, ax=axins)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.set_title('Clustered', fontsize=10)
        
    except Exception as e:
        print(f"Clustering failed: {e}")
        # Fallback: just show a text message
        ax2.text(0.5, 0.5, 'Clustering visualization\nunavailable due to\nnumerical issues', 
                ha='center', va='center', fontsize=14,
                transform=ax2.transAxes)
        ax2.set_title('Objective Clustering (Unavailable)', fontsize=16)
        ax2.axis('off')
    
    plt.suptitle('Objective Relationships Analysis', fontsize=18)
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{output_dir}/png/objective_correlation_6d.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/objective_correlation_6d.pdf', bbox_inches='tight')
    plt.close(fig)

    # Also save the correlation matrix as CSV for further analysis
    corr_matrix.to_csv(f'{output_dir}/objective_correlations.csv')
    print("Saved correlation matrix to objective_correlations.csv")

def create_pareto_front_2d_projections(df, output_dir='./results'):
    """Create 2D projections of 6D Pareto front"""
    
    # Create figure with subplots for key objective pairs
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define key objective pairs
    pairs = [
        ('f1_total_cost_USD', 'detection_recall', 'Cost vs Performance'),
        ('f1_total_cost_USD', 'f5_environmental_impact_kWh_year', 'Cost vs Sustainability'),
        ('f1_total_cost_USD', 'system_MTBF_hours', 'Cost vs Reliability'),
        ('detection_recall', 'f3_latency_seconds', 'Performance vs Speed'),
        ('detection_recall', 'f5_environmental_impact_kWh_year', 'Performance vs Sustainability'),
        ('f5_environmental_impact_kWh_year', 'system_MTBF_hours', 'Sustainability vs Reliability'),
        ('f3_latency_seconds', 'f4_traffic_disruption_hours', 'Operational Efficiency'),
        ('detection_recall', 'system_MTBF_hours', 'Performance vs Reliability'),
        ('f1_total_cost_USD', 'f3_latency_seconds', 'Cost vs Speed')
    ]
    
    for idx, (obj1, obj2, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Scale data if needed
        x_data = df[obj1] / 1000 if 'cost' in obj1 or 'environmental' in obj1 else df[obj1]
        y_data = df[obj2] / 1000 if 'cost' in obj2 or 'environmental' in obj2 or 'MTBF' in obj2 else df[obj2]
        
        # Color by third objective (rotating through remaining objectives)
        color_options = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds', 
                        'f4_traffic_disruption_hours', 'f5_environmental_impact_kWh_year', 
                        'system_MTBF_hours']
        color_obj = [c for c in color_options if c not in [obj1, obj2]][0]
        color_data = df[color_obj] / 1000 if 'cost' in color_obj or 'environmental' in color_obj else df[color_obj]
        
        # Create scatter plot
        scatter = ax.scatter(x_data, y_data, c=color_data, 
                           cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Labels
        x_label = obj1.replace('_', ' ').replace('f1 ', '').replace('USD', '(k$)').replace('kWh year', '(MWh/y)')
        y_label = obj2.replace('_', ' ').replace('hours', '(k hours)').replace('USD', '(k$)')
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        color_label = color_obj.replace('_', ' ').replace('f1 ', '').replace('f5 ', '')
        cbar.set_label(color_label, fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        
        # Add Pareto front line for main trade-offs
        if idx < 3:  # For first three plots, show Pareto front
            # Simple 2D Pareto front identification
            pareto_mask = np.ones(len(df), dtype=bool)
            for i in range(len(df)):
                for j in range(len(df)):
                    if i != j:
                        # For objectives to minimize (all except recall and MTBF)
                        if 'recall' in obj1 or 'MTBF' in obj1:
                            cond1 = df.iloc[j][obj1] >= df.iloc[i][obj1]
                        else:
                            cond1 = df.iloc[j][obj1] <= df.iloc[i][obj1]
                            
                        if 'recall' in obj2 or 'MTBF' in obj2:
                            cond2 = df.iloc[j][obj2] >= df.iloc[i][obj2]
                        else:
                            cond2 = df.iloc[j][obj2] <= df.iloc[i][obj2]
                            
                        if cond1 and cond2:
                            if (('recall' not in obj1 and 'MTBF' not in obj1 and df.iloc[j][obj1] < df.iloc[i][obj1]) or
                                ('recall' in obj1 or 'MTBF' in obj1) and df.iloc[j][obj1] > df.iloc[i][obj1] or
                                ('recall' not in obj2 and 'MTBF' not in obj2 and df.iloc[j][obj2] < df.iloc[i][obj2]) or
                                ('recall' in obj2 or 'MTBF' in obj2) and df.iloc[j][obj2] > df.iloc[i][obj2]):
                                pareto_mask[i] = False
                                break
            
            # Sort and plot Pareto front
            pareto_points = df[pareto_mask].sort_values(obj1)
            ax.plot(pareto_points[obj1] / (1000 if 'cost' in obj1 or 'environmental' in obj1 else 1),
                   pareto_points[obj2] / (1000 if 'cost' in obj2 or 'environmental' in obj2 or 'MTBF' in obj2 else 1),
                   'r--', linewidth=2, alpha=0.7, label='2D Pareto Front')
            ax.legend()
    
    plt.suptitle('2D Projections of 6D Pareto Front', fontsize=20)
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{output_dir}/png/pareto_2d_projections_6obj.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/pareto_2d_projections_6obj.pdf', bbox_inches='tight')
    plt.close(fig)

def create_configuration_performance_profiles(df, output_dir='./results'):
    """Create performance profiles for different configuration types"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Sensor-Algorithm combination heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Create pivot table for sensor-algorithm combinations
    pivot_recall = df.pivot_table(values='detection_recall', 
                                  index='sensor', 
                                  columns='algorithm', 
                                  aggfunc='mean')
    
    # Shorten names
    pivot_recall.index = [s.split('/')[-1][:20] for s in pivot_recall.index]
    pivot_recall.columns = [a.split('/')[-1][:15] for a in pivot_recall.columns]
    
    # Create heatmap
    sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Average Detection Recall'}, ax=ax1)
    ax1.set_title('Sensor-Algorithm Performance Matrix', fontsize=16)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Sensor', fontsize=12)
    
    # 2. LOD impact analysis
    ax2 = fig.add_subplot(gs[0, 2])
    
    lod_impact = df.groupby(['geometric_LOD', 'condition_LOD']).agg({
        'detection_recall': 'mean',
        'f1_total_cost_USD': 'mean'
    })
    
    # Create grouped bar plot
    lod_combinations = lod_impact.index
    x = np.arange(len(lod_combinations))
    width = 0.35
    
    ax2.bar(x - width/2, lod_impact['detection_recall'], width, 
           label='Avg Recall', color='skyblue')
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, lod_impact['f1_total_cost_USD']/1000000, width,
                label='Avg Cost (M$)', color='lightcoral')
    
    ax2.set_xlabel('LOD Combination\n(Geometric, Condition)', fontsize=12)
    ax2.set_ylabel('Detection Recall', fontsize=12)
    ax2_twin.set_ylabel('Cost (Million $)', fontsize=12)
    ax2.set_title('Impact of LOD Selection', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{g}\n{c}' for g, c in lod_combinations], fontsize=10)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 3. Infrastructure choices distribution
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Count configurations by infrastructure type
    infra_counts = pd.DataFrame({
        'Storage': df['storage'].value_counts(),
        'Communication': df['communication'].value_counts(),
        'Deployment': df['deployment'].value_counts()
    }).fillna(0)
    
    # Shorten names
    infra_counts.index = [i.split('/')[-1][:20] for i in infra_counts.index]
    
    # Create stacked bar plot
    infra_counts.plot(kind='bar', stacked=True, ax=ax3, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_xlabel('Infrastructure Component', fontsize=12)
    ax3.set_ylabel('Number of Configurations', fontsize=12)
    ax3.set_title('Infrastructure Choice Distribution', fontsize=14)
    ax3.legend(title='Component Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Operational parameters impact
    ax4 = fig.add_subplot(gs[1, 1:])
    
    # Create bins for crew size and inspection cycle
    df['crew_size_bin'] = pd.cut(df['crew_size'], bins=[0, 3, 6, 10], 
                                 labels=['Small (1-3)', 'Medium (4-6)', 'Large (7-10)'])
    df['inspection_cycle_bin'] = pd.cut(df['inspection_cycle_days'], 
                                       bins=[0, 30, 90, 365],
                                       labels=['Frequent (<30d)', 'Regular (30-90d)', 'Infrequent (>90d)'])
    
    # Create grouped analysis
    operational_impact = df.groupby(['crew_size_bin', 'inspection_cycle_bin']).agg({
        'f1_total_cost_USD': 'mean',
        'f4_traffic_disruption_hours': 'mean',
        'f5_environmental_impact_kWh_year': 'mean'
    })
    
    # Plot
    operational_impact.plot(kind='bar', ax=ax4)
    ax4.set_xlabel('Crew Size / Inspection Frequency', fontsize=12)
    ax4.set_ylabel('Average Values', fontsize=12)
    ax4.set_title('Impact of Operational Parameters', fontsize=14)
    ax4.legend(['Total Cost ($)', 'Traffic Disruption (h)', 'Environmental Impact (kWh/y)'],
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Configuration Performance Profiles', fontsize=20)
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{output_dir}/png/configuration_profiles_6obj.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/configuration_profiles_6obj.pdf', bbox_inches='tight')
    plt.close(fig)

def create_sustainability_focused_analysis(df, output_dir='./results'):
    """Create detailed sustainability analysis visualizations"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Calculate carbon footprint
    carbon_intensity = 0.4  # kg CO2 per kWh
    df['carbon_footprint_kg'] = df['f5_environmental_impact_kWh_year'] * carbon_intensity
    
    # 1. Environmental impact distribution by technology
    ax1 = fig.add_subplot(gs[0, :2])
    
    tech_categories = ['Sensor', 'Algorithm', 'Deployment']
    positions = np.arange(len(tech_categories))
    
    for i, (category, col) in enumerate([('Sensor', 'sensor'), 
                                         ('Algorithm', 'algorithm'), 
                                         ('Deployment', 'deployment')]):
        data_by_type = []
        labels = []
        
        for tech_type in df[col].unique()[:5]:  # Top 5 each
            carbon_values = df[df[col] == tech_type]['carbon_footprint_kg']
            if len(carbon_values) > 0:
                data_by_type.append(carbon_values)
                labels.append(tech_type.split('/')[-1][:15])
        
        # Create violin plot
        parts = ax1.violinplot(data_by_type, positions=positions[i]*6 + np.arange(len(data_by_type)),
                              widths=0.7, showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor(plt.cm.Set3(i))
            pc.set_alpha(0.7)
    
    ax1.set_xlabel('Technology Type', fontsize=12)
    ax1.set_ylabel('Annual Carbon Footprint (kg CO₂)', fontsize=12)
    ax1.set_title('Carbon Footprint Distribution by Technology Choice', fontsize=16)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Energy efficiency vs Performance trade-off
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Create 2D density plot
    x = df['f5_environmental_impact_kWh_year'] / 1000  # Convert to MWh
    y = df['detection_recall']
    
    # Calculate density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Sort by density for better visualization
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
    
    scatter = ax2.scatter(x, y, c=z, s=50, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Annual Energy (MWh)', fontsize=12)
    ax2.set_ylabel('Detection Recall', fontsize=12)
    ax2.set_title('Energy-Performance Trade-off Density', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Solution Density', fontsize=10)
    
    # 3. Sustainability metrics correlation
    ax3 = fig.add_subplot(gs[1, 0])
    
    sustainability_metrics = df[['f5_environmental_impact_kWh_year', 
                               'system_MTBF_hours', 
                               'f1_total_cost_USD',
                               'carbon_footprint_kg']].copy()
    
    # Normalize for comparison
    for col in sustainability_metrics.columns:
        sustainability_metrics[col] = (sustainability_metrics[col] - sustainability_metrics[col].min()) / \
                                    (sustainability_metrics[col].max() - sustainability_metrics[col].min())
    
    # Create correlation matrix
    corr = sustainability_metrics.corr()
    
    # Plot
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax3)
    ax3.set_title('Sustainability Metrics Correlation', fontsize=14)
    
    # 4. Best sustainable configurations
    ax4 = fig.add_subplot(gs[1, 1:])
    
    # Identify Pareto front for sustainability objectives
    sustainability_pareto = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i != j:
                if (df.iloc[j]['f5_environmental_impact_kWh_year'] <= df.iloc[i]['f5_environmental_impact_kWh_year'] and
                    df.iloc[j]['system_MTBF_hours'] >= df.iloc[i]['system_MTBF_hours'] and
                    (df.iloc[j]['f5_environmental_impact_kWh_year'] < df.iloc[i]['f5_environmental_impact_kWh_year'] or
                     df.iloc[j]['system_MTBF_hours'] > df.iloc[i]['system_MTBF_hours'])):
                    dominated = True
                    break
        if not dominated:
            sustainability_pareto.append(i)
    
    # Plot sustainability Pareto front
    pareto_df = df.iloc[sustainability_pareto]
    
    ax4.scatter(df['f5_environmental_impact_kWh_year']/1000, 
               df['system_MTBF_hours']/8760,
               c='lightgray', s=30, alpha=0.5, label='All solutions')
    
    ax4.scatter(pareto_df['f5_environmental_impact_kWh_year']/1000,
               pareto_df['system_MTBF_hours']/8760,
               c='green', s=100, alpha=0.8, edgecolors='black',
               label='Sustainability Pareto optimal')
    
    ax4.set_xlabel('Annual Energy Consumption (MWh)', fontsize=12)
    ax4.set_ylabel('System MTBF (years)', fontsize=12)
    ax4.set_title('Sustainability-Reliability Pareto Front', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Technology sustainability ranking
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate sustainability score (lower is better)
    df['sustainability_score'] = (
        0.5 * (df['f5_environmental_impact_kWh_year'] - df['f5_environmental_impact_kWh_year'].min()) / 
              (df['f5_environmental_impact_kWh_year'].max() - df['f5_environmental_impact_kWh_year'].min()) +
        0.5 * (1 - (df['system_MTBF_hours'] - df['system_MTBF_hours'].min()) / 
              (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min()))
    )
    
    # Rank by sensor type
    sensor_sustainability = df.groupby('sensor').agg({
        'sustainability_score': 'mean',
        'f5_environmental_impact_kWh_year': 'mean',
        'system_MTBF_hours': 'mean',
        'carbon_footprint_kg': 'mean'
    }).sort_values('sustainability_score')
    
    # Create table
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    for idx, (sensor, row) in enumerate(sensor_sustainability.head(10).iterrows()):
        table_data.append([
            f"{idx+1}",
            sensor.split('/')[-1][:30],
            f"{row['sustainability_score']:.3f}",
            f"{row['f5_environmental_impact_kWh_year']:.0f}",
            f"{row['carbon_footprint_kg']:.1f}",
            f"{row['system_MTBF_hours']/8760:.1f}"
        ])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Rank', 'Sensor Technology', 'Sustainability\nScore', 
                               'Avg Energy\n(kWh/year)', 'Avg CO₂\n(kg/year)', 'Avg MTBF\n(years)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.08, 0.35, 0.15, 0.15, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color cells by rank
    colors = plt.cm.Greens_r(np.linspace(0.2, 0.8, 10))
    for i in range(1, 11):
        if i < len(table_data) + 1:
            for j in range(6):
                table[(i, j)].set_facecolor(colors[i-1])
    
    ax5.set_title('Top 10 Most Sustainable Sensor Technologies', fontsize=16, pad=20)
    
    plt.suptitle('Comprehensive Sustainability Analysis', fontsize=20, y=0.98)
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{output_dir}/png/sustainability_analysis_detailed.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/sustainability_analysis_detailed.pdf', bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def create_all_visualizations(csv_file='./results/pareto_solutions_6d.csv', 
                            output_dir='./results'):
    """
    Main function to create all visualizations for 6-objective optimization results
    
    Args:
        csv_file: Path to the CSV file containing Pareto solutions
        output_dir: Directory to save visualization outputs
    """
    
    print("="*60)
    print("Advanced Visualizations for 6-Objective RMTwin Optimization")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} Pareto-optimal solutions")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run the optimization first to generate results")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Enhanced 3D Pareto visualizations
    print("1. Creating 3D Pareto visualizations...")
    create_enhanced_3d_pareto_visualization(df, output_dir)
    
    # 2. Decision variable impact analysis
    print("2. Creating decision variable impact analysis...")
    create_decision_variable_impact_analysis(df, output_dir)
    
    # 3. Technology comparison dashboard
    print("3. Creating technology comparison dashboard...")
    create_technology_comparison_dashboard(df, output_dir)
    
    # 4. Objective correlation analysis
    print("4. Creating objective correlation heatmap...")
    create_objective_correlation_heatmap(df, output_dir)
    
    # 5. 2D Pareto projections
    print("5. Creating 2D Pareto projections...")
    create_pareto_front_2d_projections(df, output_dir)
    
    # 6. Configuration performance profiles
    print("6. Creating configuration performance profiles...")
    create_configuration_performance_profiles(df, output_dir)
    
    # 7. Sustainability analysis
    print("7. Creating sustainability analysis...")
    create_sustainability_focused_analysis(df, output_dir)
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

# Run if called directly
if __name__ == "__main__":
    create_all_visualizations()
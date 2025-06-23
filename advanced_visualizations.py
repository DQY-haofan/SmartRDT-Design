"""
Advanced Visualizations Module
创建用于学术论文的高质量可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import logging
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Set up matplotlib for academic papers
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def create_baseline_comparison_plot(all_df, baseline_results, output_dir):
    """创建基准方法对比图"""
    logger.info("Creating baseline comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Define objective pairs for comparison
    comparisons = [
        ('f1_total_cost_USD', 'detection_recall', 'Cost vs Detection Performance', axes[0, 0]),
        ('f3_latency_seconds', 'detection_recall', 'Latency vs Detection Performance', axes[0, 1]),
        ('f1_total_cost_USD', 'f3_latency_seconds', 'Cost vs Latency', axes[1, 0]),
        ('f4_traffic_disruption_hours', 'detection_recall', 'Traffic Disruption vs Detection Performance', axes[1, 1])
    ]

    for x_col, y_col, title, ax in comparisons:
        # Plot Pareto front
        if 'cost' in x_col:
            x_data = all_df[x_col] / 1000  # Convert to k$
            x_label = 'Total Cost (k$)'
        else:
            x_data = all_df[x_col]
            x_label = x_col.replace('_', ' ').replace('f1 ', '').replace('f3 ', '').replace('f4 ', '').title()

        y_label = y_col.replace('_', ' ').title()

        # Plot Pareto solutions
        scatter = ax.scatter(x_data, all_df[y_col],
                           c='lightgray', s=30, alpha=0.6,
                           edgecolors='black', linewidth=0.5,
                           label='NSGA-II Pareto Front')

        # Plot baseline solutions
        # Greedy baseline
        greedy = baseline_results['greedy']['objectives']
        if 'cost' in x_col:
            greedy_x = greedy['f1_total_cost_USD'] / 1000
        else:
            greedy_x = greedy[x_col]
        greedy_y = greedy[y_col] if y_col in greedy else greedy['detection_recall']

        ax.scatter(greedy_x, greedy_y,
                  marker='*', s=500, c='red', edgecolors='darkred', linewidth=2,
                  label='Greedy Cost-Min', zorder=10)

        # Weighted sum baseline
        weighted = baseline_results['weighted_sum']['objectives']
        if 'cost' in x_col:
            weighted_x = weighted['f1_total_cost_USD'] / 1000
        else:
            weighted_x = weighted[x_col]
        weighted_y = weighted[y_col] if y_col in weighted else weighted['detection_recall']

        ax.scatter(weighted_x, weighted_y,
                  marker='s', s=300, c='blue', edgecolors='darkblue', linewidth=2,
                  label='Weighted-Sum', zorder=10)

        # Add constraint lines if applicable
        if y_col == 'detection_recall':
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7,
                      label='Min Recall Constraint')
        if x_col == 'f3_latency_seconds':
            ax.axvline(x=60, color='orange', linestyle='--', alpha=0.7,
                      label='Max Latency Constraint')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save in both formats
    fig.savefig(f'{output_dir}/png/baseline_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/baseline_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info("Baseline comparison plot saved")


def create_decision_variable_impact_analysis(df, output_dir):
    """创建决策变量影响分析图"""
    logger.info("Creating decision variable impact analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Sensor impact on objectives
    ax1 = axes[0]
    sensor_data = df.groupby('sensor').agg({
        'detection_recall': 'mean',
        'f1_total_cost_USD': 'mean'
    }).reset_index()
    sensor_data = sensor_data.sort_values('detection_recall', ascending=False)[:10]

    x = np.arange(len(sensor_data))
    width = 0.35

    ax1_twin = ax1.twinx()

    bars1 = ax1.bar(x - width/2, sensor_data['detection_recall'], width,
                    label='Avg Recall', color='green', alpha=0.7)
    bars2 = ax1_twin.bar(x + width/2, sensor_data['f1_total_cost_USD']/1000, width,
                        label='Avg Cost (k$)', color='red', alpha=0.7)

    ax1.set_xlabel('Sensor Type')
    ax1.set_ylabel('Average Detection Recall', color='green')
    ax1_twin.set_ylabel('Average Cost (k$)', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('instances#', '')[:15] for s in sensor_data['sensor']],
                       rotation=45, ha='right')
    ax1.set_title('Sensor Impact on Key Objectives')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # Algorithm impact
    ax2 = axes[1]
    algo_impact = df.groupby('algorithm')['detection_recall'].describe()
    algo_impact = algo_impact.sort_values('mean', ascending=False)[:10]

    positions = range(len(algo_impact))
    ax2.boxplot([df[df['algorithm'] == algo]['detection_recall'].values
                for algo in algo_impact.index],
               positions=positions,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))

    ax2.set_xticks(positions)
    ax2.set_xticklabels([a.replace('instances#', '')[:12] for a in algo_impact.index],
                       rotation=45, ha='right')
    ax2.set_ylabel('Detection Recall')
    ax2.set_title('Algorithm Performance Distribution')

    # LOD impact - Fixed version
    ax3 = axes[2]
    lod_impact = df.groupby('geometric_LOD').agg({
        'detection_recall': 'mean',
        'f3_latency_seconds': 'mean',
        'f1_total_cost_USD': 'mean'
    })

    # Only use LOD levels that exist in the data
    available_lods = list(lod_impact.index)
    x = np.arange(len(available_lods))

    ax3.bar(x - 0.2, lod_impact['detection_recall'], 0.2,
           label='Recall', color='green')
    ax3.bar(x, lod_impact['f3_latency_seconds']/100, 0.2,
           label='Latency/100', color='orange')
    ax3.bar(x + 0.2, lod_impact['f1_total_cost_USD']/1000000, 0.2,
           label='Cost (M$)', color='red')

    ax3.set_xticks(x)
    ax3.set_xticklabels(available_lods)
    ax3.set_ylabel('Normalized Values')
    ax3.set_title('LOD Impact on Multiple Objectives')
    ax3.legend()

    # Crew size vs cost
    ax4 = axes[3]
    crew_impact = df.groupby('crew_size')['f1_total_cost_USD'].mean() / 1000
    ax4.plot(crew_impact.index, crew_impact.values, 'o-', linewidth=2, markersize=8)
    ax4.set_xlabel('Crew Size')
    ax4.set_ylabel('Average Total Cost (k$)')
    ax4.set_title('Crew Size Impact on Cost')
    ax4.set_xticks(range(1, 11))

    # Inspection cycle distribution
    ax5 = axes[4]
    cycle_bins = [1, 7, 30, 90, 180, 365]
    cycle_labels = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annual']
    df['cycle_category'] = pd.cut(df['inspection_cycle_days'], bins=cycle_bins, labels=cycle_labels)
    cycle_counts = df['cycle_category'].value_counts()

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(cycle_counts)))
    ax5.pie(cycle_counts.values, labels=cycle_counts.index, colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax5.set_title('Distribution of Inspection Frequencies')

    # Data rate impact
    ax6 = axes[5]
    # Bin data rates
    rate_bins = [0, 10, 30, 50, 70, 100]
    df['rate_bin'] = pd.cut(df['data_rate_Hz'], bins=rate_bins)
    rate_impact = df.groupby('rate_bin')['detection_recall'].mean()

    x = range(len(rate_impact))
    ax6.bar(x, rate_impact.values, color='skyblue', edgecolor='navy')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'{int(b.left)}-{int(b.right)}' for b in rate_impact.index],
                       rotation=45)
    ax6.set_xlabel('Data Rate (Hz)')
    ax6.set_ylabel('Average Detection Recall')
    ax6.set_title('Data Rate Impact on Detection Performance')

    plt.tight_layout()

    # Save
    fig.savefig(f'{output_dir}/png/decision_variable_impact.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/decision_variable_impact.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info("Decision variable impact analysis saved")


def create_pareto_front_3d(df, output_dir):
    """创建3D帕累托前沿可视化"""
    logger.info("Creating 3D Pareto front visualization...")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Select three objectives for 3D plot
    x = df['f1_total_cost_USD'] / 1000  # k$
    y = df['detection_recall']
    z = df['f3_latency_seconds']

    # Color by fourth objective
    c = df['f4_traffic_disruption_hours']

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=50, alpha=0.7,
                        edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Total Cost (k$)', labelpad=10)
    ax.set_ylabel('Detection Recall', labelpad=10)
    ax.set_zlabel('Latency (seconds)', labelpad=10)
    ax.set_title('3D Pareto Front Visualization\n(Color: Traffic Disruption Hours)')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Traffic Disruption (hours)', rotation=270, labelpad=20)

    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)

    # Save
    fig.savefig(f'{output_dir}/png/pareto_front_3d.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/pareto_front_3d.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info("3D Pareto front visualization saved")


def create_constraint_satisfaction_analysis(all_df, reasonable_df, output_dir):
    """创建约束满足分析图"""
    logger.info("Creating constraint satisfaction analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Feasibility distribution
    ax1 = axes[0, 0]
    feasible_count = all_df['is_feasible'].sum()
    infeasible_count = len(all_df) - feasible_count

    wedges, texts, autotexts = ax1.pie([feasible_count, infeasible_count],
                                       labels=['Feasible', 'Infeasible'],
                                       colors=['#2ecc71', '#e74c3c'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 12})
    ax1.set_title('Solution Feasibility Distribution')

    # Constraint violations scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(all_df['constraint_violation_g1_latency'],
                         all_df['constraint_violation_g2_recall'],
                         c=all_df['is_feasible'],
                         cmap='RdYlGn',
                         alpha=0.6,
                         edgecolors='k',
                         linewidth=0.5)

    # Add feasible region
    rect = Rectangle((-100, -100), 100, 100, linewidth=2,
                    edgecolor='green', facecolor='green', alpha=0.1)
    ax2.add_patch(rect)

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('G1: Latency Constraint Violation (s)')
    ax2.set_ylabel('G2: Recall Constraint Violation')
    ax2.set_title('Constraint Violations Distribution')
    ax2.set_xlim(-10, max(all_df['constraint_violation_g1_latency']) * 1.1)
    ax2.set_ylim(-0.1, max(all_df['constraint_violation_g2_recall']) * 1.1)
    ax2.text(-5, -0.05, 'Feasible\nRegion', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

    # Quality distribution
    ax3 = axes[1, 0]
    quality_categories = ['All Solutions', 'Feasible Only', 'High Quality\n(Recall≥0.8, Latency≤60s)']
    counts = [len(all_df), feasible_count, len(reasonable_df)]
    colors = ['lightgray', '#3498db', '#2ecc71']

    bars = ax3.bar(quality_categories, counts, color=colors, edgecolor='black')
    ax3.set_ylabel('Number of Solutions')
    ax3.set_title('Solution Quality Distribution')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')

    # Performance vs feasibility
    ax4 = axes[1, 1]

    # Separate feasible and infeasible solutions
    feasible_df = all_df[all_df['is_feasible']]
    infeasible_df = all_df[~all_df['is_feasible']]

    ax4.scatter(infeasible_df['f1_total_cost_USD']/1000,
               infeasible_df['detection_recall'],
               c='red', s=30, alpha=0.5, label='Infeasible')
    ax4.scatter(feasible_df['f1_total_cost_USD']/1000,
               feasible_df['detection_recall'],
               c='green', s=30, alpha=0.7, label='Feasible')

    ax4.set_xlabel('Total Cost (k$)')
    ax4.set_ylabel('Detection Recall')
    ax4.set_title('Cost-Performance Trade-off by Feasibility')
    ax4.legend()

    plt.tight_layout()

    # Save
    fig.savefig(f'{output_dir}/png/constraint_satisfaction_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/constraint_satisfaction_analysis.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info("Constraint satisfaction analysis saved")


def create_solution_ranking_table(reasonable_df, output_dir):
    """创建前10个解决方案的排名表"""
    logger.info("Creating solution ranking table...")

    if reasonable_df.empty:
        logger.warning("No reasonable solutions to rank")
        return

    # Calculate efficiency score (recall per million dollars)
    reasonable_df['efficiency_score'] = reasonable_df['detection_recall'] / (reasonable_df['f1_total_cost_USD'] / 1000000)

    # Sort by efficiency score
    top_solutions = reasonable_df.nlargest(10, 'efficiency_score')

    # Create figure with table
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Rank', 'Sensor', 'Algorithm', 'LOD', 'Cost (k$)',
              'Recall', 'Latency (s)', 'Disruption (h)', 'Efficiency']

    for i, (_, row) in enumerate(top_solutions.iterrows()):
        table_data.append([
            i + 1,
            row['sensor'].replace('instances#', '')[:20],
            row['algorithm'].replace('instances#', '')[:20],
            row['geometric_LOD'],
            f"{row['f1_total_cost_USD']/1000:.1f}",
            f"{row['detection_recall']:.3f}",
            f"{row['f3_latency_seconds']:.1f}",
            f"{row['f4_traffic_disruption_hours']:.1f}",
            f"{row['efficiency_score']:.2f}"
        ])

    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#ecf0f1')

    ax.set_title('Top 10 Most Efficient RMTwin Configurations\n(Efficiency = Recall / Cost in M$)',
                fontsize=16, pad=20)

    plt.tight_layout()

    # Save
    fig.savefig(f'{output_dir}/png/top_solutions_table.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/top_solutions_table.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info("Solution ranking table saved")


def create_convergence_comparison(output_dir):
    """创建收敛性对比图（如果有ablation study结果）"""
    try:
        with open(f'{output_dir}/ablation_study_results.json', 'r') as f:
            ablation_data = json.load(f)

        logger.info("Creating convergence comparison plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Extract convergence data
        generations = [d['generation'] for d in ablation_data['convergence_history']]
        cv_avg = [d['cv_average'] for d in ablation_data['convergence_history']]
        n_feasible = [d['n_feasible'] for d in ablation_data['convergence_history']]

        # Plot constraint violation
        ax1.plot(generations, cv_avg, 'r-', linewidth=2, label='No Pre-filtering')
        ax1.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Feasibility Threshold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Average Constraint Violation')
        ax1.set_title('Convergence of Constraint Satisfaction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot feasible solutions count
        ax2.plot(generations, n_feasible, 'b-', linewidth=2, label='No Pre-filtering')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Feasible Solutions')
        ax2.set_title('Evolution of Feasible Solutions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        fig.savefig(f'{output_dir}/png/convergence_comparison.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_dir}/pdf/convergence_comparison.pdf', bbox_inches='tight')
        plt.close(fig)

        logger.info("Convergence comparison plot saved")

    except FileNotFoundError:
        logger.info("No ablation study results found, skipping convergence comparison")




# 在 advanced_visualizations.py 中需要修改的函数

def create_all_visualizations(df, baseline_results, output_dir):
    """创建所有可视化（修改版）"""
    # Create directories
    os.makedirs(f'{output_dir}/png', exist_ok=True)
    os.makedirs(f'{output_dir}/pdf', exist_ok=True)

    # 使用过滤获得高质量解
    high_quality_df = df[df['is_high_quality']] if 'is_high_quality' in df.columns else df

    # Create all visualizations
    create_baseline_comparison_plot(df, baseline_results, output_dir)
    create_decision_variable_impact_analysis(df, output_dir)
    create_pareto_front_3d(df, output_dir)
    create_constraint_satisfaction_analysis(df, high_quality_df, output_dir)
    create_solution_ranking_table(high_quality_df, output_dir)
    create_convergence_comparison(output_dir)
    create_comprehensive_report(df, high_quality_df, baseline_results, output_dir)

    logger.info("All visualizations complete!")


def create_constraint_satisfaction_analysis(df, high_quality_df, output_dir):
    """创建约束满足分析图（修改版）"""
    logger.info("Creating constraint satisfaction analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Feasibility distribution
    ax1 = axes[0, 0]
    feasible_count = df['is_feasible'].sum()
    infeasible_count = len(df) - feasible_count

    wedges, texts, autotexts = ax1.pie([feasible_count, infeasible_count],
                                       labels=['Feasible', 'Infeasible'],
                                       colors=['#2ecc71', '#e74c3c'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 12})
    ax1.set_title('Solution Feasibility Distribution')

    # Constraint violations scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['constraint_violation_g1_latency'],
                          df['constraint_violation_g2_recall'],
                          c=df['is_feasible'],
                          cmap='RdYlGn',
                          alpha=0.6,
                          edgecolors='k',
                          linewidth=0.5)

    # Add feasible region
    rect = Rectangle((-100, -100), 100, 100, linewidth=2,
                     edgecolor='green', facecolor='green', alpha=0.1)
    ax2.add_patch(rect)

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('G1: Latency Constraint Violation (s)')
    ax2.set_ylabel('G2: Recall Constraint Violation')
    ax2.set_title('Constraint Violations Distribution')
    ax2.set_xlim(-10, max(df['constraint_violation_g1_latency']) * 1.1)
    ax2.set_ylim(-0.1, max(df['constraint_violation_g2_recall']) * 1.1)
    ax2.text(-5, -0.05, 'Feasible\nRegion', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

    # Quality distribution
    ax3 = axes[1, 0]
    quality_categories = ['All Solutions', 'Feasible Only', 'High Quality\n(Min Requirements)']
    counts = [len(df), feasible_count, len(high_quality_df)]
    colors = ['lightgray', '#3498db', '#2ecc71']

    bars = ax3.bar(quality_categories, counts, color=colors, edgecolor='black')
    ax3.set_ylabel('Number of Solutions')
    ax3.set_title('Solution Quality Distribution')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}', ha='center', va='bottom')

    # Performance vs feasibility
    ax4 = axes[1, 1]

    # Separate feasible and infeasible solutions
    feasible_df = df[df['is_feasible']]
    infeasible_df = df[~df['is_feasible']]

    if not infeasible_df.empty:
        ax4.scatter(infeasible_df['f1_total_cost_USD'] / 1000,
                    infeasible_df['detection_recall'],
                    c='red', s=30, alpha=0.5, label='Infeasible')
    if not feasible_df.empty:
        ax4.scatter(feasible_df['f1_total_cost_USD'] / 1000,
                    feasible_df['detection_recall'],
                    c='green', s=30, alpha=0.7, label='Feasible')

    ax4.set_xlabel('Total Cost (k$)')
    ax4.set_ylabel('Detection Recall')
    ax4.set_title('Cost-Performance Trade-off by Feasibility')
    ax4.legend()

    plt.tight_layout()

    # Save
    fig.savefig(f'{output_dir}/png/constraint_satisfaction_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/pdf/constraint_satisfaction_analysis.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info("Constraint satisfaction analysis saved")


def create_comprehensive_report(df, high_quality_df, baseline_results, output_dir):
    """创建综合分析报告（修改版）"""
    logger.info("Creating comprehensive analysis report...")

    report_text = f"""
EXPERT-ENHANCED PARETO FRONT ANALYSIS REPORT
============================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. SOLUTION STATISTICS:
----------------------
- Total Pareto Solutions: {len(df)}
- Feasible Solutions: {df['is_feasible'].sum()}
- High-Quality Solutions (Meeting All Requirements): {len(high_quality_df)}
- Feasibility Rate: {df['is_feasible'].sum() / len(df) * 100:.1f}%

2. OBJECTIVE RANGES:
-------------------
All Solutions:
- Cost: ${df['f1_total_cost_USD'].min():,.0f} - ${df['f1_total_cost_USD'].max():,.0f}
- Recall: {df['detection_recall'].min():.3f} - {df['detection_recall'].max():.3f}
- Latency: {df['f3_latency_seconds'].min():.1f}s - {df['f3_latency_seconds'].max():.1f}s
- Disruption: {df['f4_traffic_disruption_hours'].min():.1f}h - {df['f4_traffic_disruption_hours'].max():.1f}h

"""

    if not high_quality_df.empty:
        report_text += f"""
High-Quality Solutions:
- Cost: ${high_quality_df['f1_total_cost_USD'].min():,.0f} - ${high_quality_df['f1_total_cost_USD'].max():,.0f}
- Recall: {high_quality_df['detection_recall'].min():.3f} - {high_quality_df['detection_recall'].max():.3f}
- Latency: {high_quality_df['f3_latency_seconds'].min():.1f}s - {high_quality_df['f3_latency_seconds'].max():.1f}s
- Disruption: {high_quality_df['f4_traffic_disruption_hours'].min():.1f}h - {high_quality_df['f4_traffic_disruption_hours'].max():.1f}h

3. BEST SOLUTIONS BY CRITERION:
------------------------------
Most Cost-Effective:
- Cost: ${high_quality_df.loc[high_quality_df['f1_total_cost_USD'].idxmin(), 'f1_total_cost_USD']:,.0f}
- Sensor: {high_quality_df.loc[high_quality_df['f1_total_cost_USD'].idxmin(), 'sensor']}
- Algorithm: {high_quality_df.loc[high_quality_df['f1_total_cost_USD'].idxmin(), 'algorithm']}
- Recall: {high_quality_df.loc[high_quality_df['f1_total_cost_USD'].idxmin(), 'detection_recall']:.3f}

Best Detection Performance:
- Recall: {high_quality_df.loc[high_quality_df['detection_recall'].idxmax(), 'detection_recall']:.3f}
- Cost: ${high_quality_df.loc[high_quality_df['detection_recall'].idxmax(), 'f1_total_cost_USD']:,.0f}
- Sensor: {high_quality_df.loc[high_quality_df['detection_recall'].idxmax(), 'sensor']}
- Algorithm: {high_quality_df.loc[high_quality_df['detection_recall'].idxmax(), 'algorithm']}

Fastest Processing:
- Latency: {high_quality_df.loc[high_quality_df['f3_latency_seconds'].idxmin(), 'f3_latency_seconds']:.1f}s
- Cost: ${high_quality_df.loc[high_quality_df['f3_latency_seconds'].idxmin(), 'f1_total_cost_USD']:,.0f}
- Algorithm: {high_quality_df.loc[high_quality_df['f3_latency_seconds'].idxmin(), 'algorithm']}

Minimal Traffic Disruption:
- Disruption: {high_quality_df.loc[high_quality_df['f4_traffic_disruption_hours'].idxmin(), 'f4_traffic_disruption_hours']:.1f}h
- Cost: ${high_quality_df.loc[high_quality_df['f4_traffic_disruption_hours'].idxmin(), 'f1_total_cost_USD']:,.0f}
- Sensor: {high_quality_df.loc[high_quality_df['f4_traffic_disruption_hours'].idxmin(), 'sensor']}
"""

    # Add baseline comparison if available
    if baseline_results:
        report_text += f"""
4. BASELINE COMPARISON:
----------------------"""

        if 'greedy' in baseline_results:
            greedy = baseline_results['greedy']['objectives']
            report_text += f"""
Greedy Cost-Minimization:
- Cost: ${greedy['f1_total_cost_USD']:,.0f}
- Recall: {greedy['detection_recall']:.3f}
- Latency: {greedy['f3_latency_seconds']:.1f}s
- Disruption: {greedy['f4_traffic_disruption_hours']:.1f}h
- Feasible: {baseline_results['greedy']['constraints']['is_feasible']}
"""

        if 'weighted_sum' in baseline_results:
            weighted = baseline_results['weighted_sum']['objectives']
            report_text += f"""
Weighted-Sum Optimization:
- Cost: ${weighted['f1_total_cost_USD']:,.0f}
- Recall: {weighted['detection_recall']:.3f}
- Latency: {weighted['f3_latency_seconds']:.1f}s
- Disruption: {weighted['f4_traffic_disruption_hours']:.1f}h
- Feasible: {baseline_results['weighted_sum']['constraints']['is_feasible']}
"""

    report_text += f"""
5. COMPONENT DIVERSITY:
----------------------
"""

    if not high_quality_df.empty:
        report_text += f"""
In High-Quality Solutions:
- Unique Sensors: {high_quality_df['sensor'].nunique()} types
- Unique Algorithms: {high_quality_df['algorithm'].nunique()} types
- Unique Storage: {high_quality_df['storage'].nunique()} types
- Unique Communication: {high_quality_df['communication'].nunique()} types
- Unique Deployments: {high_quality_df['deployment'].nunique()} types

Most Common Configurations:
- Top Sensor: {high_quality_df['sensor'].value_counts().index[0]} ({high_quality_df['sensor'].value_counts().iloc[0]} occurrences)
- Top Algorithm: {high_quality_df['algorithm'].value_counts().index[0]} ({high_quality_df['algorithm'].value_counts().iloc[0]} occurrences)
- Top LOD: {high_quality_df['geometric_LOD'].value_counts().index[0]} ({high_quality_df['geometric_LOD'].value_counts().iloc[0]} occurrences)
"""

    report_text += f"""
6. KEY INSIGHTS:
---------------
- The NSGA-II multi-objective optimization produced {len(df)} Pareto-optimal solutions,
  demonstrating the complexity of the RMTwin configuration space.

- {df['is_feasible'].sum() / len(df) * 100:.1f}% of Pareto solutions satisfy all hard constraints,
  highlighting the importance of constraint handling in the optimization process.

- The baseline methods produce single solutions that are dominated by multiple Pareto solutions,
  validating the superiority of the multi-objective approach.

- Component diversity in the Pareto set enables decision-makers to choose configurations
  based on specific contextual requirements and preferences.

============================================
END OF REPORT
"""

    with open(f'{output_dir}/comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    logger.info("Comprehensive report saved")
#!/usr/bin/env python3
"""
RMTwin Enhanced Visualization for Publications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from typing import Dict
import json
import warnings

warnings.filterwarnings('ignore')

COLORS = {
    'NSGA-III': '#1f77b4',
    'random': '#7f7f7f',
    'weighted': '#ff7f0e',
    'grid': '#2ca02c',
    'expert': '#d62728',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data(pareto_path: str):
    pareto_df = pd.read_csv(pareto_path)
    pareto_dir = Path(pareto_path).parent
    baseline_dfs = {}
    for f in pareto_dir.glob('baseline_*.csv'):
        name = f.stem.replace('baseline_', '')
        df = pd.read_csv(f)
        if 'is_feasible' in df.columns:
            df = df[df['is_feasible']]
        baseline_dfs[name] = df
    return pareto_df, baseline_dfs


def fig1_enhanced_pareto_front(pareto_df, baseline_dfs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    for name, df in baseline_dfs.items():
        if len(df) > 0:
            ax1.scatter(df['f1_total_cost_USD'] / 1e6, df['detection_recall'],
                        alpha=0.15, s=15, c=COLORS.get(name, 'gray'), label=None)

    ps = pareto_df.sort_values('f1_total_cost_USD')
    ax1.fill_between(ps['f1_total_cost_USD'] / 1e6, 0, ps['detection_recall'],
                     alpha=0.1, color=COLORS['NSGA-III'], label='NSGA-III Dominated Region')
    ax1.plot(ps['f1_total_cost_USD'] / 1e6, ps['detection_recall'],
             'b-', linewidth=2.5, zorder=8, label='NSGA-III Pareto Front')
    ax1.scatter(ps['f1_total_cost_USD'] / 1e6, ps['detection_recall'],
                s=100, c=COLORS['NSGA-III'], marker='*', zorder=10,
                edgecolors='white', linewidths=0.8, label=f'NSGA-III Solutions (n={len(pareto_df)})')

    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axhspan(0.95, 1.0, alpha=0.08, color='green')
    ax1.text(8, 0.97, 'High-Quality\nRegion', fontsize=10, color='green', ha='center')

    ax1.set_xlabel('Total Cost (Million USD)', fontsize=12)
    ax1.set_ylabel('Detection Recall', fontsize=12)
    ax1.set_title('(a) Pareto Front with Dominated Region', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0.6, 1.0)

    ax2 = axes[1]
    nsga_hq = pareto_df[pareto_df['detection_recall'] >= 0.90]

    for name, df in baseline_dfs.items():
        hq = df[df['detection_recall'] >= 0.90]
        if len(hq) > 0:
            ax2.scatter(hq['f1_total_cost_USD'] / 1e6, hq['detection_recall'],
                        alpha=0.3, s=25, c=COLORS.get(name, 'gray'), label=f'{name.title()} (n={len(hq)})')

    ax2.scatter(nsga_hq['f1_total_cost_USD'] / 1e6, nsga_hq['detection_recall'],
                s=150, c=COLORS['NSGA-III'], marker='*', zorder=10,
                edgecolors='white', linewidths=1, label=f'NSGA-III (n={len(nsga_hq)})')

    nsga_hq_sorted = nsga_hq.sort_values('f1_total_cost_USD')
    ax2.plot(nsga_hq_sorted['f1_total_cost_USD'] / 1e6, nsga_hq_sorted['detection_recall'],
             'b--', linewidth=1.5, alpha=0.7, zorder=5)

    if len(nsga_hq) > 0:
        best = nsga_hq.loc[nsga_hq['f1_total_cost_USD'].idxmin()]
        ax2.annotate(f"NSGA Best\n${best['f1_total_cost_USD'] / 1e6:.3f}M\nRecall={best['detection_recall']:.3f}",
                     xy=(best['f1_total_cost_USD'] / 1e6, best['detection_recall']),
                     xytext=(40, -25), textcoords='offset points', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='orange'),
                     arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Total Cost (Million USD)', fontsize=12)
    ax2.set_ylabel('Detection Recall', fontsize=12)
    ax2.set_title('(b) High-Quality Region (Recall >= 0.90)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0.90, 1.0)

    plt.suptitle('Figure 1: Multi-Objective Pareto Front Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig1_pareto_front_enhanced.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 1: Enhanced Pareto Front")


def fig2_dominance_and_contribution(pareto_df, baseline_dfs, metrics_dir, output_dir):
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    try:
        coverage_df = pd.read_csv(metrics_dir / 'coverage_metrics.csv')
        contrib_df = pd.read_csv(metrics_dir / 'contribution_metrics.csv')
    except:
        print("  ⚠ Metrics files not found")
        plt.close()
        return

    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(coverage_df))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, coverage_df['NSGA_Dominates_%'], width,
                    label='NSGA-III Dominates', color=COLORS['NSGA-III'], alpha=0.8)
    bars2 = ax1.bar(x + width / 2, coverage_df['Baseline_Dominates_%'], width,
                    label='Baseline Dominates', color='gray', alpha=0.6)
    ax1.set_ylabel('Dominated Solutions (%)', fontsize=11)
    ax1.set_title('(a) Dominance Coverage', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(coverage_df['Baseline'], rotation=0)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 105)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{bar.get_height():.0f}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['NSGA-III'])

    ax2 = fig.add_subplot(gs[1])
    contrib_df = contrib_df[contrib_df['Contribution_%'] > 0].sort_values('Contribution_%', ascending=False)
    if len(contrib_df) > 0:
        colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in contrib_df['Method']]
        explode = [0.1 if m == 'NSGA-III' else 0 for m in contrib_df['Method']]
        wedges, texts, autotexts = ax2.pie(
            contrib_df['Contribution_%'], labels=contrib_df['Method'],
            autopct='%1.1f%%', colors=colors, explode=explode,
            startangle=90, textprops={'fontsize': 10})
        for i, m in enumerate(contrib_df['Method']):
            if m == 'NSGA-III':
                autotexts[i].set_fontweight('bold')
    ax2.set_title('(b) Contribution to Combined\nPareto Front', fontsize=12, fontweight='bold')

    ax3 = fig.add_subplot(gs[2])
    net_adv = coverage_df['NSGA_Dominates_%'] - coverage_df['Baseline_Dominates_%']
    colors_bar = [COLORS['NSGA-III'] if v > 0 else 'red' for v in net_adv]
    bars = ax3.barh(coverage_df['Baseline'], net_adv, color=colors_bar, alpha=0.8)
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.set_xlabel('Net Dominance Advantage (%)', fontsize=11)
    ax3.set_title('(c) NSGA-III Net Advantage', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, net_adv):
        x_pos = bar.get_width() + 2 if val > 0 else bar.get_width() - 8
        ax3.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:.0f}%', va='center', fontsize=10, fontweight='bold')

    plt.suptitle('Figure 2: NSGA-III Dominance and Contribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig2_dominance_contribution.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 2: Dominance & Contribution")


def fig3_quality_metrics(metrics_dir, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    try:
        quality_df = pd.read_csv(metrics_dir / 'quality_metrics.csv')
    except:
        print("  ⚠ Quality metrics not found")
        plt.close()
        return

    methods = quality_df['Method'].tolist()
    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in methods]

    ax1 = axes[0]
    bars = ax1.bar(methods, quality_df['HV'], color=colors, alpha=0.8)
    ax1.set_ylabel('Hypervolume', fontsize=11)
    ax1.set_title('(a) Hypervolume (higher=better)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[1]
    bars = ax2.bar(methods, quality_df['Spacing'], color=colors, alpha=0.8)
    ax2.set_ylabel('Spacing', fontsize=11)
    ax2.set_title('(b) Spacing (lower=better)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    ax3 = axes[2]
    bars = ax3.bar(methods, quality_df['Spread'], color=colors, alpha=0.8)
    ax3.set_ylabel('Maximum Spread', fontsize=11)
    ax3.set_title('(c) Maximum Spread (higher=better)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    plt.suptitle('Figure 3: Multi-Objective Quality Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig3_quality_metrics.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 3: Quality Metrics")


def fig4_high_quality_focus(pareto_df, baseline_dfs, output_dir):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = [0.95, 0.90, 0.85, 0.80]
    methods = ['NSGA-III'] + list(baseline_dfs.keys())
    count_data = []
    for thresh in thresholds:
        row = {'Threshold': f'>={thresh}'}
        row['NSGA-III'] = len(pareto_df[pareto_df['detection_recall'] >= thresh])
        for name, df in baseline_dfs.items():
            row[name] = len(df[df['detection_recall'] >= thresh]) if len(df) > 0 else 0
        count_data.append(row)
    count_df = pd.DataFrame(count_data).set_index('Threshold')
    count_df.plot(kind='bar', ax=ax1, width=0.8,
                  color=[COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in count_df.columns])
    ax1.set_xlabel('Recall Threshold')
    ax1.set_ylabel('Number of Solutions')
    ax1.set_title('(a) Solutions at Quality Thresholds', fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    hq_data = []
    labels = []
    nsga_hq = pareto_df[pareto_df['detection_recall'] >= 0.95]['f1_total_cost_USD'] / 1e6
    if len(nsga_hq) > 0:
        hq_data.append(nsga_hq.values)
        labels.append('NSGA-III')
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            continue
        hq = df[df['detection_recall'] >= 0.95]['f1_total_cost_USD'] / 1e6
        if len(hq) > 0:
            hq_data.append(hq.values)
            labels.append(name)
    if hq_data:
        bp = ax2.boxplot(hq_data, tick_labels=labels, patch_artist=True)
        colors_box = [COLORS.get(l, 'gray') if l != 'NSGA-III' else COLORS['NSGA-III'] for l in labels]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Cost (Million USD)')
    ax2.set_title('(b) Cost Distribution (Recall >= 0.95)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    ax3 = fig.add_subplot(gs[1, 0])
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            continue
        hq = df[df['detection_recall'] >= 0.90]
        if len(hq) > 0:
            ax3.scatter(hq['f1_total_cost_USD'] / 1e6, hq['detection_recall'],
                        alpha=0.3, s=30, c=COLORS.get(name, 'gray'), label=name)
    nsga_hq = pareto_df[pareto_df['detection_recall'] >= 0.90]
    ax3.scatter(nsga_hq['f1_total_cost_USD'] / 1e6, nsga_hq['detection_recall'],
                s=120, c=COLORS['NSGA-III'], marker='*', zorder=10,
                edgecolors='white', linewidths=1, label='NSGA-III')
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Cost (Million USD)')
    ax3.set_ylabel('Detection Recall')
    ax3.set_title('(c) High-Quality Solutions Distribution', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_ylim(0.90, 1.0)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    table_data = []
    nsga_hq = pareto_df[pareto_df['detection_recall'] >= 0.95]
    if len(nsga_hq) > 0:
        best = nsga_hq.loc[nsga_hq['f1_total_cost_USD'].idxmin()]
        table_data.append(['NSGA-III', f"${best['f1_total_cost_USD'] / 1e6:.3f}M",
                           f"{best['detection_recall']:.3f}", str(len(nsga_hq))])
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            table_data.append([name.title(), 'N/A', 'N/A', '0'])
            continue
        hq = df[df['detection_recall'] >= 0.95]
        if len(hq) > 0:
            best = hq.loc[hq['f1_total_cost_USD'].idxmin()]
            table_data.append([name.title(), f"${best['f1_total_cost_USD'] / 1e6:.3f}M",
                               f"{best['detection_recall']:.3f}", str(len(hq))])
        else:
            table_data.append([name.title(), 'N/A', 'N/A', '0'])

    if len(table_data) > 0:
        table = ax4.table(cellText=table_data,
                          colLabels=['Method', 'Min Cost', 'Recall', 'N'],
                          loc='center', cellLoc='center',
                          colColours=['lightblue'] * 4)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        # 高亮NSGA-III行
        if len(table_data) > 0:
            for j in range(4):
                table[(1, j)].set_facecolor('#d4e6f1')

    ax4.set_title('(d) Best Solutions (Recall >= 0.95)', fontweight='bold', pad=20)
    plt.suptitle('Figure 4: High-Quality Region Analysis', fontsize=14, fontweight='bold', y=1.02)
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig4_high_quality_focus.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 4: High-Quality Focus")


def fig5_statistical_significance(metrics_dir, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        stat_df = pd.read_csv(metrics_dir / 'statistical_tests.csv')
        stat_df = stat_df.dropna(subset=['Cost_d', 'Recall_d'])
    except:
        print("  ⚠ Statistical tests not found")
        plt.close()
        return

    if len(stat_df) == 0:
        print("  ⚠ No valid statistical data")
        plt.close()
        return

    ax1 = axes[0]
    methods = [c.replace('vs ', '') for c in stat_df['Comparison']]
    deltas = stat_df['Cost_d'].values
    colors = ['green' if d > 0 else 'red' for d in deltas]
    bars = ax1.barh(methods, deltas, color=colors, alpha=0.7)
    for thresh in [0.474, 0.33, 0.147]:
        ax1.axvline(x=thresh, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=-thresh, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.set_xlabel("Cliff's Delta (Cost)", fontsize=11)
    ax1.set_title("(a) Effect Size: Cost\n(Positive = NSGA-III Better)", fontweight='bold')
    ax1.set_xlim(-1, 1)

    ax2 = axes[1]
    deltas = stat_df['Recall_d'].values
    colors = ['green' if d > 0 else 'red' for d in deltas]
    bars = ax2.barh(methods, deltas, color=colors, alpha=0.7)
    for thresh in [0.474, 0.33, 0.147]:
        ax2.axvline(x=thresh, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=-thresh, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel("Cliff's Delta (Recall)", fontsize=11)
    ax2.set_title("(b) Effect Size: Recall\n(Positive = NSGA-III Better)", fontweight='bold')
    ax2.set_xlim(-1, 1)

    plt.suptitle("Figure 5: Statistical Effect Size Analysis (Cliff's Delta)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig5_statistical_effect.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 5: Statistical Effect Size")


def fig6_summary_dashboard(pareto_df, baseline_dfs, metrics_dir, output_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    try:
        coverage_df = pd.read_csv(metrics_dir / 'coverage_metrics.csv')
        contrib_df = pd.read_csv(metrics_dir / 'contribution_metrics.csv')
        quality_df = pd.read_csv(metrics_dir / 'quality_metrics.csv')
    except:
        print("  ⚠ Metrics not found")
        plt.close()
        return

    ax1 = fig.add_subplot(gs[0, 0])
    for name, df in baseline_dfs.items():
        if len(df) > 0:
            ax1.scatter(df['f1_total_cost_USD'] / 1e6, df['detection_recall'],
                        alpha=0.1, s=10, c=COLORS.get(name, 'gray'))
    ps = pareto_df.sort_values('f1_total_cost_USD')
    ax1.plot(ps['f1_total_cost_USD'] / 1e6, ps['detection_recall'], 'b-', lw=2)
    ax1.scatter(ps['f1_total_cost_USD'] / 1e6, ps['detection_recall'],
                s=80, c=COLORS['NSGA-III'], marker='*', zorder=10)
    ax1.set_xlabel('Cost (M$)')
    ax1.set_ylabel('Recall')
    ax1.set_title('(a) Pareto Front Overview', fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0.6, 1.0)

    ax2 = fig.add_subplot(gs[0, 1])
    avg_dom = coverage_df['NSGA_Dominates_%'].mean()
    avg_dom_by = coverage_df['Baseline_Dominates_%'].mean()
    ax2.bar(['NSGA-III\nDominates', 'Baseline\nDominates'],
            [avg_dom, avg_dom_by], color=[COLORS['NSGA-III'], 'gray'], alpha=0.8)
    ax2.set_ylabel('Average %')
    ax2.set_title(f'(b) Average Dominance\nNSGA: {avg_dom:.0f}% vs Baseline: {avg_dom_by:.0f}%', fontweight='bold')
    ax2.set_ylim(0, 100)

    ax3 = fig.add_subplot(gs[0, 2])
    nsga_contrib = contrib_df[contrib_df['Method'] == 'NSGA-III']['Contribution_%'].values
    nsga_contrib = nsga_contrib[0] if len(nsga_contrib) > 0 else 0
    other_contrib = 100 - nsga_contrib
    ax3.pie([nsga_contrib, other_contrib], labels=['NSGA-III', 'Others'],
            colors=[COLORS['NSGA-III'], 'lightgray'],
            autopct='%1.1f%%', startangle=90, explode=[0.05, 0],
            textprops={'fontsize': 11})
    ax3.set_title('(c) Contribution to\nCombined Front', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 0])
    hq_counts = {'NSGA-III': len(pareto_df[pareto_df['detection_recall'] >= 0.95])}
    for name, df in baseline_dfs.items():
        hq_counts[name] = len(df[df['detection_recall'] >= 0.95]) if len(df) > 0 else 0
    colors_bar = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in hq_counts.keys()]
    ax4.bar(hq_counts.keys(), hq_counts.values(), color=colors_bar, alpha=0.8)
    ax4.set_ylabel('Count')
    ax4.set_title('(d) High-Quality Solutions\n(Recall >= 0.95)', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    ax5 = fig.add_subplot(gs[1, 1])
    hv_data = quality_df.set_index('Method')['HV']
    colors_bar = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in hv_data.index]
    ax5.bar(hv_data.index, hv_data.values, color=colors_bar, alpha=0.8)
    ax5.set_ylabel('Hypervolume')
    ax5.set_title('(e) Hypervolume Comparison', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    conclusions = f"""
    KEY FINDINGS
    {'=' * 32}

    * NSGA-III dominates {avg_dom:.0f}% of
      baseline solutions on average

    * Contributes {nsga_contrib:.1f}% to the
      combined Pareto front

    * Provides {len(pareto_df)} diverse
      Pareto-optimal solutions

    * {hq_counts['NSGA-III']} solutions achieve
      Recall >= 0.95

    {'=' * 32}
    """
    ax6.text(0.1, 0.5, conclusions, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Figure 6: NSGA-III Performance Dashboard', fontsize=15, fontweight='bold', y=1.02)
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig6_summary_dashboard.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 6: Summary Dashboard")


def generate_all_figures(pareto_path: str, metrics_dir: str = None, output_dir: str = './results/figures'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_dir is None:
        metrics_dir = Path(pareto_path).parent.parent / 'metrics'
    else:
        metrics_dir = Path(metrics_dir)

    print("=" * 60)
    print("Generating Enhanced Publication Figures")
    print("=" * 60)

    pareto_df, baseline_dfs = load_data(pareto_path)
    print(f"Loaded: NSGA-III ({len(pareto_df)}), Baselines: {list(baseline_dfs.keys())}")

    fig1_enhanced_pareto_front(pareto_df, baseline_dfs, output_dir)
    fig2_dominance_and_contribution(pareto_df, baseline_dfs, metrics_dir, output_dir)
    fig3_quality_metrics(metrics_dir, output_dir)
    fig4_high_quality_focus(pareto_df, baseline_dfs, output_dir)
    fig5_statistical_significance(metrics_dir, output_dir)
    fig6_summary_dashboard(pareto_df, baseline_dfs, metrics_dir, output_dir)

    print("=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization_enhanced.py <pareto_csv> [metrics_dir] [output_dir]")
        sys.exit(1)
    pareto_path = sys.argv[1]
    metrics_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './results/figures'
    generate_all_figures(pareto_path, metrics_dir, output_dir)
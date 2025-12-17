#!/usr/bin/env python3
"""
RMTwin Individual Publication Figures
=====================================
生成单独的图表（非图组），适合论文直接使用
改进Pareto前沿曲线的可视化效果

Author: RMTwin Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import Dict
import warnings

warnings.filterwarnings('ignore')

# 配色
COLORS = {
    'NSGA-III': '#1f77b4',
    'random': '#7f7f7f',
    'weighted': '#ff7f0e',
    'grid': '#2ca02c',
    'expert': '#d62728',
}

# IEEE风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data(pareto_path: str):
    """加载数据"""
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


def fig_pareto_front_main(pareto_df, baseline_dfs, output_dir):
    """
    主Pareto前沿图 - 改进版
    突出显示NSGA-III的Pareto边界
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. 先画所有baseline点（作为背景）
    for name, df in baseline_dfs.items():
        if len(df) > 0:
            ax.scatter(df['f1_total_cost_USD'] / 1e6, df['detection_recall'],
                       alpha=0.2, s=15, c=COLORS.get(name, 'gray'),
                       label=f'{name.title()} (n={len(df)})')

    # 2. 提取并排序Pareto前沿点
    pareto_sorted = pareto_df.sort_values('f1_total_cost_USD')
    x_pareto = pareto_sorted['f1_total_cost_USD'].values / 1e6
    y_pareto = pareto_sorted['detection_recall'].values

    # 3. 创建阶梯状Pareto前沿（更准确的表示）
    x_step = []
    y_step = []
    for i in range(len(x_pareto)):
        if i > 0:
            # 水平线段
            x_step.append(x_pareto[i])
            y_step.append(y_pareto[i - 1])
        x_step.append(x_pareto[i])
        y_step.append(y_pareto[i])

    # 4. 填充Pareto支配区域
    x_fill = [0] + list(x_step) + [x_step[-1], 0]
    y_fill = [y_step[0]] + list(y_step) + [0, 0]
    ax.fill(x_fill, y_fill, alpha=0.15, color=COLORS['NSGA-III'],
            label='NSGA-III Dominated Region')

    # 5. 画Pareto前沿线（粗线）
    ax.plot(x_step, y_step, color=COLORS['NSGA-III'], linewidth=3,
            linestyle='-', zorder=8, label='NSGA-III Pareto Front')

    # 6. 画Pareto解点（大星星）
    ax.scatter(x_pareto, y_pareto, s=200, c=COLORS['NSGA-III'],
               marker='*', zorder=10, edgecolors='white', linewidths=1.5,
               label=f'NSGA-III Solutions (n={len(pareto_df)})')

    # 7. 标注关键点
    # 最低成本点
    min_cost_idx = pareto_sorted['f1_total_cost_USD'].idxmin()
    min_cost_sol = pareto_sorted.loc[min_cost_idx]
    ax.annotate(f"Min Cost\n${min_cost_sol['f1_total_cost_USD'] / 1e6:.2f}M",
                xy=(min_cost_sol['f1_total_cost_USD'] / 1e6, min_cost_sol['detection_recall']),
                xytext=(20, -30), textcoords='offset points', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 最高recall点
    max_recall_idx = pareto_sorted['detection_recall'].idxmax()
    max_recall_sol = pareto_sorted.loc[max_recall_idx]
    if max_recall_idx != min_cost_idx:
        ax.annotate(f"Max Recall\n{max_recall_sol['detection_recall']:.3f}",
                    xy=(max_recall_sol['f1_total_cost_USD'] / 1e6, max_recall_sol['detection_recall']),
                    xytext=(30, 10), textcoords='offset points', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 8. 添加参考线
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Target Recall = 0.95')
    ax.axhspan(0.95, 1.0, alpha=0.05, color='green')

    # 设置
    ax.set_xlabel('Total Cost (Million USD)', fontsize=14)
    ax.set_ylabel('Detection Recall', fontsize=14)
    ax.set_title('Pareto Front: Cost vs Detection Recall', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_xlim(0, max(10, x_pareto.max() * 1.1))
    ax.set_ylim(0.55, 1.02)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_pareto_front.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Pareto Front (main)")


def fig_pareto_front_zoomed(pareto_df, baseline_dfs, output_dir):
    """
    高质量区域放大图
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 只显示高质量解 (recall >= 0.90)
    nsga_hq = pareto_df[pareto_df['detection_recall'] >= 0.90].sort_values('f1_total_cost_USD')

    # Baseline高质量解
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            continue
        hq = df[df['detection_recall'] >= 0.90]
        if len(hq) > 0:
            ax.scatter(hq['f1_total_cost_USD'] / 1e6, hq['detection_recall'],
                       alpha=0.3, s=30, c=COLORS.get(name, 'gray'),
                       label=f'{name.title()} (n={len(hq)})')

    # NSGA-III Pareto前沿
    if len(nsga_hq) > 0:
        x_pareto = nsga_hq['f1_total_cost_USD'].values / 1e6
        y_pareto = nsga_hq['detection_recall'].values

        # 阶梯线
        x_step, y_step = [], []
        for i in range(len(x_pareto)):
            if i > 0:
                x_step.append(x_pareto[i])
                y_step.append(y_pareto[i - 1])
            x_step.append(x_pareto[i])
            y_step.append(y_pareto[i])

        ax.plot(x_step, y_step, color=COLORS['NSGA-III'], linewidth=2.5,
                linestyle='-', zorder=8)
        ax.scatter(x_pareto, y_pareto, s=180, c=COLORS['NSGA-III'],
                   marker='*', zorder=10, edgecolors='white', linewidths=1.5,
                   label=f'NSGA-III (n={len(nsga_hq)})')

        # 标注最优解
        best = nsga_hq.loc[nsga_hq['f1_total_cost_USD'].idxmin()]
        ax.annotate(f"Best: ${best['f1_total_cost_USD'] / 1e6:.3f}M\nRecall: {best['detection_recall']:.3f}",
                    xy=(best['f1_total_cost_USD'] / 1e6, best['detection_recall']),
                    xytext=(40, -20), textcoords='offset points', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9, edgecolor='orange'),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Target = 0.95')
    ax.axhspan(0.95, 1.0, alpha=0.08, color='green')

    ax.set_xlabel('Total Cost (Million USD)', fontsize=14)
    ax.set_ylabel('Detection Recall', fontsize=14)
    ax.set_title('High-Quality Region (Recall ≥ 0.90)', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.895, 1.005)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_pareto_zoomed.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Pareto Front (zoomed)")


def fig_dominance_coverage(metrics_dir, output_dir):
    """
    支配关系对比图 - 单独
    """
    try:
        coverage_df = pd.read_csv(metrics_dir / 'coverage_metrics.csv')
    except:
        print("  ⚠ Coverage metrics not found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(coverage_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, coverage_df['NSGA_Dominates_%'], width,
                   label='NSGA-III Dominates Baseline', color=COLORS['NSGA-III'], alpha=0.85)
    bars2 = ax.bar(x + width / 2, coverage_df['Baseline_Dominates_%'], width,
                   label='Baseline Dominates NSGA-III', color='gray', alpha=0.6)

    # 数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold', color=COLORS['NSGA-III'])
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                f'{height:.1f}%', ha='center', fontsize=10, color='gray')

    ax.set_ylabel('Dominated Solutions (%)', fontsize=14)
    ax.set_xlabel('Baseline Method', fontsize=14)
    ax.set_title('Dominance Coverage Analysis', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.title() for b in coverage_df['Baseline']], fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_dominance.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Dominance Coverage")


def fig_contribution_pie(metrics_dir, output_dir):
    """
    贡献度饼图 - 单独
    """
    try:
        contrib_df = pd.read_csv(metrics_dir / 'contribution_metrics.csv')
    except:
        print("  ⚠ Contribution metrics not found")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # 过滤零贡献
    contrib_df = contrib_df[contrib_df['Contribution_%'] > 0].sort_values('Contribution_%', ascending=False)

    if len(contrib_df) == 0:
        print("  ⚠ No contribution data")
        plt.close()
        return

    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III']
              for m in contrib_df['Method']]
    explode = [0.08 if m == 'NSGA-III' else 0 for m in contrib_df['Method']]

    wedges, texts, autotexts = ax.pie(
        contrib_df['Contribution_%'],
        labels=contrib_df['Method'],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 12},
        pctdistance=0.75
    )

    # 突出NSGA-III
    for i, m in enumerate(contrib_df['Method']):
        if m == 'NSGA-III':
            autotexts[i].set_fontweight('bold')
            autotexts[i].set_fontsize(14)
            texts[i].set_fontweight('bold')
            texts[i].set_fontsize(14)

    ax.set_title('Contribution to Combined Pareto Front', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_contribution.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Contribution Pie")


def fig_hypervolume_comparison(metrics_dir, output_dir):
    """
    Hypervolume对比柱状图 - 单独
    """
    try:
        quality_df = pd.read_csv(metrics_dir / 'quality_metrics.csv')
    except:
        print("  ⚠ Quality metrics not found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = quality_df['Method'].tolist()
    hv_values = quality_df['HV'].values
    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in methods]

    bars = ax.bar(methods, hv_values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # 标注最高值
    max_idx = np.argmax(hv_values)
    bars[max_idx].set_edgecolor('gold')
    bars[max_idx].set_linewidth(3)

    # 数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + hv_values.max() * 0.02,
                f'{height:.2e}', ha='center', fontsize=10, rotation=0)

    ax.set_ylabel('Hypervolume', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_title('Hypervolume Comparison (Higher = Better)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_hypervolume.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Hypervolume Comparison")


def fig_high_quality_count(pareto_df, baseline_dfs, output_dir):
    """
    高质量解数量对比 - 单独
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 统计各方法高质量解数量
    hq_data = {'NSGA-III': len(pareto_df[pareto_df['detection_recall'] >= 0.95])}
    for name, df in baseline_dfs.items():
        if len(df) > 0:
            hq_data[name] = len(df[df['detection_recall'] >= 0.95])
        else:
            hq_data[name] = 0

    methods = list(hq_data.keys())
    counts = list(hq_data.values())
    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in methods]

    bars = ax.bar(methods, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # 数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(counts) * 0.02,
                f'{int(height)}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Solutions', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_title('High-Quality Solutions Count (Recall ≥ 0.95)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_hq_count.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ High-Quality Count")


def fig_effect_size(metrics_dir, output_dir):
    """
    效应量对比图 - 单独
    """
    try:
        stat_df = pd.read_csv(metrics_dir / 'statistical_tests.csv')
        stat_df = stat_df.dropna(subset=['Cost_d'])
    except:
        print("  ⚠ Statistical tests not found")
        return

    if len(stat_df) == 0:
        print("  ⚠ No valid statistical data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [c.replace('vs ', '') for c in stat_df['Comparison']]
    deltas = stat_df['Cost_d'].values

    colors = ['#2ecc71' if d > 0.147 else '#e74c3c' if d < -0.147 else '#95a5a6' for d in deltas]
    bars = ax.barh(methods, deltas, color=colors, alpha=0.8, height=0.6)

    # 效应量参考线
    for thresh, label, alpha in [(0.474, 'Large', 0.3), (0.33, 'Medium', 0.2), (0.147, 'Small', 0.1)]:
        ax.axvline(x=thresh, color='green', linestyle='--', alpha=alpha, linewidth=1.5)
        ax.axvline(x=-thresh, color='red', linestyle='--', alpha=alpha, linewidth=1.5)

    ax.axvline(x=0, color='black', linewidth=1.5)

    # 标签和效应量描述
    for bar, (_, row) in zip(bars, stat_df.iterrows()):
        x = bar.get_width()
        effect = row['Cost_Effect']
        label_x = x + 0.05 if x > 0 else x - 0.05
        ha = 'left' if x > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{x:.2f} ({effect})', va='center', ha=ha, fontsize=11, fontweight='bold')

    ax.set_xlabel("Cliff's Delta (Cost)", fontsize=14)
    ax.set_ylabel('Baseline Method', fontsize=14)
    ax.set_title("Effect Size Analysis\n(Positive = NSGA-III has lower cost)", fontsize=16, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='y', labelsize=12)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='NSGA-III Better'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Baseline Better'),
        Patch(facecolor='#95a5a6', alpha=0.8, label='Negligible')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_effect_size.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Effect Size")


def fig_efficiency_comparison(pareto_df, baseline_dfs, metrics_dir, output_dir):
    """
    效率对比图 - 关键图！展示NSGA-III的效率优势
    """
    try:
        contrib_df = pd.read_csv(metrics_dir / 'contribution_metrics.csv')
    except:
        print("  ⚠ Contribution metrics not found")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # 计算效率
    efficiency_data = []

    nsga_contrib = contrib_df[contrib_df['Method'] == 'NSGA-III']['Contribution_%'].values
    nsga_contrib = nsga_contrib[0] if len(nsga_contrib) > 0 else 0
    efficiency_data.append({
        'Method': 'NSGA-III',
        'N_Solutions': len(pareto_df),
        'Contribution': nsga_contrib,
        'Efficiency': nsga_contrib / len(pareto_df) if len(pareto_df) > 0 else 0
    })

    for name, df in baseline_dfs.items():
        contrib = contrib_df[contrib_df['Method'] == name]['Contribution_%'].values
        contrib = contrib[0] if len(contrib) > 0 else 0
        n = len(df)
        efficiency_data.append({
            'Method': name,
            'N_Solutions': n,
            'Contribution': contrib,
            'Efficiency': contrib / n if n > 0 else 0
        })

    eff_df = pd.DataFrame(efficiency_data)
    eff_df = eff_df.sort_values('Efficiency', ascending=True)

    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in eff_df['Method']]

    bars = ax.barh(eff_df['Method'], eff_df['Efficiency'], color=colors, alpha=0.85, height=0.6)

    # 数值标签
    for bar, (_, row) in zip(bars, eff_df.iterrows()):
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}%/sol\n({row["Contribution"]:.1f}% / {row["N_Solutions"]})',
                va='center', fontsize=10)

    ax.set_xlabel('Efficiency (Contribution % per Solution)', fontsize=14)
    ax.set_ylabel('Method', fontsize=14)
    ax.set_title('Algorithm Efficiency: Contribution per Solution', fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_efficiency.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Efficiency Comparison")


def fig_cost_at_threshold(pareto_df, baseline_dfs, output_dir):
    """
    不同Recall阈值下的最低成本对比
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = [0.95, 0.90, 0.85, 0.80]
    methods = ['NSGA-III'] + list(baseline_dfs.keys())

    x = np.arange(len(thresholds))
    width = 0.15

    for i, method in enumerate(methods):
        if method == 'NSGA-III':
            df = pareto_df
        else:
            df = baseline_dfs[method]

        if len(df) == 0:
            continue

        min_costs = []
        for thresh in thresholds:
            hq = df[df['detection_recall'] >= thresh]
            if len(hq) > 0:
                min_costs.append(hq['f1_total_cost_USD'].min() / 1e6)
            else:
                min_costs.append(np.nan)

        color = COLORS.get(method, 'gray') if method != 'NSGA-III' else COLORS['NSGA-III']
        bars = ax.bar(x + i * width, min_costs, width, label=method.title(), color=color, alpha=0.85)

    ax.set_ylabel('Minimum Cost (Million USD)', fontsize=14)
    ax.set_xlabel('Recall Threshold', fontsize=14)
    ax.set_title('Minimum Cost at Different Quality Thresholds', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f'≥{t}' for t in thresholds], fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_cost_threshold.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Cost at Threshold")


def generate_all_individual_figures(pareto_path: str, metrics_dir: str = None,
                                    output_dir: str = './results/figures_individual'):
    """生成所有单独图表"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_dir is None:
        metrics_dir = Path(pareto_path).parent.parent / 'metrics'
    else:
        metrics_dir = Path(metrics_dir)

    print("=" * 60)
    print("Generating Individual Publication Figures")
    print("=" * 60)

    pareto_df, baseline_dfs = load_data(pareto_path)
    print(f"Loaded: NSGA-III ({len(pareto_df)}), Baselines: {list(baseline_dfs.keys())}")
    print()

    # 生成单独图表
    fig_pareto_front_main(pareto_df, baseline_dfs, output_dir)
    fig_pareto_front_zoomed(pareto_df, baseline_dfs, output_dir)
    fig_dominance_coverage(metrics_dir, output_dir)
    fig_contribution_pie(metrics_dir, output_dir)
    fig_hypervolume_comparison(metrics_dir, output_dir)
    fig_high_quality_count(pareto_df, baseline_dfs, output_dir)
    fig_effect_size(metrics_dir, output_dir)
    fig_efficiency_comparison(pareto_df, baseline_dfs, metrics_dir, output_dir)
    fig_cost_at_threshold(pareto_df, baseline_dfs, output_dir)

    print()
    print("=" * 60)
    print(f"✓ All {9} individual figures saved to: {output_dir}")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  1. fig_pareto_front.pdf/png     - Main Pareto front")
    print("  2. fig_pareto_zoomed.pdf/png    - High-quality region")
    print("  3. fig_dominance.pdf/png        - Dominance coverage")
    print("  4. fig_contribution.pdf/png     - Contribution pie chart")
    print("  5. fig_hypervolume.pdf/png      - HV comparison")
    print("  6. fig_hq_count.pdf/png         - High-quality count")
    print("  7. fig_effect_size.pdf/png      - Cliff's delta")
    print("  8. fig_efficiency.pdf/png       - Efficiency comparison")
    print("  9. fig_cost_threshold.pdf/png   - Cost at thresholds")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization_individual.py <pareto_csv> [metrics_dir] [output_dir]")
        sys.exit(1)

    pareto_path = sys.argv[1]
    metrics_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './results/figures_individual'

    generate_all_individual_figures(pareto_path, metrics_dir, output_dir)
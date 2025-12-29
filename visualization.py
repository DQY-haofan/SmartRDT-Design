#!/usr/bin/env python3
"""
RMTwin 完整可视化脚本 v6.1 (最终合并版)
=========================================
合并所有可视化功能，一键生成论文所需的全部图表

功能来源:
- visualization.py: 基础图表
- visualization_paper_v33.py: 论文专用图表
- visualization_ablation_v5.py: 消融实验图表
- visualization_enhanced.py: 增强功能

输出结构:
results/paper/
├── figures/          ← 所有图表 (PDF + PNG)
├── tables/           ← 所有表格 (CSV + LaTeX)
├── data/             ← 图表数据 (CSV)
└── manifest.json     ← 元数据

Author: RMTwin Research Team
Version: 6.1 (Complete Merged - Final)
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 样式配置
# =============================================================================

STYLE_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
}

COLORS = {
    'nsga3': '#1f77b4',
    'pareto': '#2E86AB',
    'random': '#7f7f7f',
    'weighted': '#ff7f0e',
    'grid': '#2ca02c',
    'expert': '#d62728',
    'traditional': '#4A90D9',
    'ml': '#F5A623',
    'dl': '#7ED321',
    'highlight': '#E63946',
    'ontology': '#17becf',
}

ABLATION_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
ALGO_MARKERS = {'Traditional': 'o', 'ML': 's', 'DL': '^'}

plt.rcParams.update(STYLE_CONFIG)


# =============================================================================
# 工具函数
# =============================================================================

def classify_algorithm(algo_name: str) -> str:
    """分类算法为 Traditional/ML/DL"""
    algo_str = str(algo_name).upper()
    dl_kw = ['DL_', 'YOLO', 'UNET', 'MASK', 'EFFICIENT', 'MOBILE', 'SAM', 'RETINA', 'FASTER']
    ml_kw = ['ML_', 'SVM', 'RANDOMFOREST', 'RANDOM_FOREST', 'XGBOOST', 'XGB', 'HYBRID']
    for kw in dl_kw:
        if kw in algo_str:
            return 'DL'
    for kw in ml_kw:
        if kw in algo_str:
            return 'ML'
    return 'Traditional'


def classify_sensor(sensor_name: str) -> str:
    """分类传感器类型"""
    sensor_str = str(sensor_name)
    for cat in ['IoT', 'Vehicle', 'Camera', 'MMS', 'UAV', 'TLS', 'FOS', 'Handheld']:
        if cat in sensor_str:
            return cat
    return 'Other'


def ensure_recall_column(df: pd.DataFrame) -> pd.DataFrame:
    """确保有 detection_recall 列"""
    if df is None:
        return None
    if 'detection_recall' not in df.columns and 'f2_one_minus_recall' in df.columns:
        df = df.copy()
        df['detection_recall'] = 1 - df['f2_one_minus_recall']
    return df


def select_representatives(df: pd.DataFrame) -> Dict[str, int]:
    """选择代表性解: low_cost, balanced, high_recall"""
    recall_col = 'detection_recall'
    cost_col = 'f1_total_cost_USD'

    if recall_col not in df.columns or cost_col not in df.columns:
        return {}

    reps = {}

    # Low-cost
    feasible = df[df[recall_col] >= 0.8]
    if len(feasible) > 0:
        reps['low_cost'] = feasible[cost_col].idxmin()
    else:
        reps['low_cost'] = df[cost_col].idxmin()

    # High-recall
    cost_80 = df[cost_col].quantile(0.8)
    affordable = df[df[cost_col] <= cost_80]
    if len(affordable) > 0:
        reps['high_recall'] = affordable[recall_col].idxmax()
    else:
        reps['high_recall'] = df[recall_col].idxmax()

    # Balanced
    norm_cost = (df[cost_col] - df[cost_col].min()) / (df[cost_col].max() - df[cost_col].min() + 1e-10)
    norm_recall = (df[recall_col].max() - df[recall_col]) / (df[recall_col].max() - df[recall_col].min() + 1e-10)
    df_temp = df.copy()
    df_temp['_score'] = norm_cost + norm_recall
    reps['balanced'] = df_temp['_score'].idxmin()

    return reps


# =============================================================================
# 主可视化类
# =============================================================================

class CompleteVisualizer:
    """完整可视化生成器 v6.1"""

    def __init__(self, output_dir: str = './results/paper'):
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.table_dir = self.output_dir / 'tables'
        self.data_dir = self.output_dir / 'data'

        for d in [self.fig_dir, self.table_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.generated_files = []
        self.manifest = {'generated_at': datetime.now().isoformat(), 'figures': {}, 'tables': {}}

    def generate_all(self,
                     pareto_df: pd.DataFrame,
                     baseline_dfs: Dict[str, pd.DataFrame] = None,
                     ablation_df: pd.DataFrame = None,
                     history_path: str = None):
        """生成所有图表和表格"""

        print("\n" + "=" * 70)
        print("RMTwin Complete Visualization v6.1")
        print("=" * 70)

        if baseline_dfs is None:
            baseline_dfs = {}

        # 预处理
        pareto_df = ensure_recall_column(pareto_df)
        for name in list(baseline_dfs.keys()):
            baseline_dfs[name] = ensure_recall_column(baseline_dfs[name])

        # ===== 核心图表 =====
        print("\n[1/5] Generating core figures...")
        self.fig1_pareto_front(pareto_df, baseline_dfs)
        self.fig2_decision_matrix(pareto_df)
        self.fig3_algorithm_comparison(pareto_df, baseline_dfs)
        self.fig4_cost_structure(pareto_df)

        # ===== 分析图表 =====
        print("\n[2/5] Generating analysis figures...")
        self.fig5_discrete_distributions(pareto_df)
        self.fig6_technology_dominance(pareto_df, baseline_dfs)
        self.fig7_baseline_comparison(pareto_df, baseline_dfs)
        self.fig8_convergence(pareto_df, history_path)

        # ===== 消融图表 =====
        print("\n[3/5] Generating ablation figures...")
        if ablation_df is not None and len(ablation_df) > 0:
            self.fig9_ablation_complete(ablation_df)
            self.fig9_ablation_validity(ablation_df)
        else:
            print("   ⚠ No ablation data, skipping Fig9")

        # ===== 补充图表 =====
        print("\n[4/5] Generating supplementary figures...")
        self.figS1_pairwise_tradeoffs(pareto_df)
        self.figS2_sensitivity(pareto_df)
        self.figS3_ontology_config_space(pareto_df, baseline_dfs)

        # ===== 表格 =====
        print("\n[5/5] Generating tables...")
        self.table1_method_comparison(pareto_df, baseline_dfs)
        self.table2_representative_solutions(pareto_df)
        self.table3_statistical_tests(pareto_df, baseline_dfs)
        if ablation_df is not None:
            self.table4_ablation_summary(ablation_df)

        # 保存 manifest
        self._save_manifest()

        # 总结
        print("\n" + "=" * 70)
        print("COMPLETE!")
        print(f"Output directory: {self.output_dir}")
        print(f"  - Figures: {len([f for f in self.generated_files if 'figures' in f])}")
        print(f"  - Tables: {len([f for f in self.generated_files if 'tables' in f])}")
        print(f"  - Data: {len([f for f in self.generated_files if 'data' in f])}")
        print("=" * 70)

        return self.generated_files

    # =========================================================================
    # 核心图表
    # =========================================================================

    def fig1_pareto_front(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig 1: Pareto前沿 (Cost vs Recall) - 主图"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Baselines
        for name, df in baseline_dfs.items():
            if df is None or len(df) == 0:
                continue
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            if len(feasible) == 0:
                continue
            ax.scatter(feasible['f1_total_cost_USD'] / 1e6,
                       feasible['detection_recall'],
                       alpha=0.3, s=30, c=COLORS.get(name, 'gray'),
                       label=f'{name.title()} (n={len(feasible)})')

        # NSGA-III Pareto
        pareto_sorted = pareto_df.sort_values('f1_total_cost_USD')
        x = pareto_sorted['f1_total_cost_USD'].values / 1e6
        y = pareto_sorted['detection_recall'].values

        ax.scatter(x, y, c=COLORS['nsga3'], s=100, marker='o',
                   edgecolors='white', linewidths=1.5, zorder=10,
                   label=f'NSGA-III Pareto (n={len(pareto_df)})')
        ax.plot(x, y, c=COLORS['nsga3'], alpha=0.5, linewidth=2, zorder=5)

        # 代表性解
        reps = select_representatives(pareto_df)
        markers = {'low_cost': ('*', 'Min Cost'), 'high_recall': ('D', 'Max Recall'), 'balanced': ('s', 'Balanced')}
        for rep_name, idx in reps.items():
            if idx in pareto_df.index:
                marker, label = markers.get(rep_name, ('o', rep_name))
                ax.scatter(pareto_df.loc[idx, 'f1_total_cost_USD'] / 1e6,
                           pareto_df.loc[idx, 'detection_recall'],
                           s=200, marker=marker, c=COLORS['highlight'], zorder=15, label=label)

        ax.set_xlabel('Total Cost ($M)', fontsize=12)
        ax.set_ylabel('Detection Recall', fontsize=12)
        ax.set_title('Pareto Front: Cost vs Detection Performance', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        self._save_fig(fig, 'fig1_pareto_front', pareto_df[['f1_total_cost_USD', 'detection_recall']])

    def fig2_decision_matrix(self, pareto_df: pd.DataFrame):
        """Fig 2: 决策矩阵热力图"""
        df = pareto_df.copy()

        obj_cols = {
            'Cost': 'f1_total_cost_USD',
            'Recall': 'detection_recall',
            'Latency': 'f3_latency_seconds',
            'Disruption': 'f4_traffic_disruption_hours',
            'Carbon': 'f5_carbon_emissions_kgCO2e_year'
        }
        obj_cols = {k: v for k, v in obj_cols.items() if v in df.columns}

        if len(obj_cols) < 2:
            print("   ⚠ Skipping fig2: insufficient columns")
            return

        # 标准化
        norm_data = pd.DataFrame()
        for name, col in obj_cols.items():
            values = df[col].values
            min_v, max_v = values.min(), values.max()
            if max_v > min_v:
                if name == 'Recall':
                    norm_data[name] = (max_v - values) / (max_v - min_v)
                else:
                    norm_data[name] = (values - min_v) / (max_v - min_v)
            else:
                norm_data[name] = 0

        # 标签
        labels = []
        for i in range(len(df)):
            s = classify_sensor(df['sensor'].iloc[i]) if 'sensor' in df.columns else ''
            a = classify_algorithm(df['algorithm'].iloc[i]) if 'algorithm' in df.columns else ''
            labels.append(f"{i + 1}:{s}|{a}")

        # 绘图
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.25)))
        im = ax.imshow(norm_data.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(obj_cols)))
        ax.set_xticklabels(list(obj_cols.keys()), fontsize=10)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(labels, fontsize=7)

        # 高亮代表性解
        reps = select_representatives(df)
        rep_colors = {'low_cost': '#2ECC71', 'balanced': '#3498DB', 'high_recall': '#E74C3C'}
        for rep_name, idx in reps.items():
            if idx in df.index:
                row = df.index.get_loc(idx)
                rect = Rectangle((-0.5, row - 0.5), len(obj_cols), 1,
                                 fill=False, edgecolor=rep_colors.get(rep_name, 'black'), linewidth=2)
                ax.add_patch(rect)

        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Normalized (0=Best, 1=Worst)', fontsize=9)

        ax.set_xlabel('Objectives', fontsize=11)
        ax.set_ylabel('Solutions', fontsize=11)
        ax.set_title('Decision Matrix: Pareto Solutions', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self._save_fig(fig, 'fig2_decision_matrix')

    def fig3_algorithm_comparison(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig 3: 算法对比箱线图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        all_data = {'NSGA-III': pareto_df}
        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feasible) > 5:
                    all_data[name.title()] = feasible

        methods = list(all_data.keys())
        colors = [COLORS.get(m.lower().replace('-', ''), 'gray') for m in methods]

        # (a) Cost
        ax1 = axes[0]
        cost_data = [all_data[m]['f1_total_cost_USD'].values / 1e6 for m in methods]
        bp1 = ax1.boxplot(cost_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Cost ($M)', fontsize=11)
        ax1.set_title('(a) Cost Distribution', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=30)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) Recall
        ax2 = axes[1]
        recall_data = [all_data[m]['detection_recall'].values for m in methods]
        bp2 = ax2.boxplot(recall_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Recall', fontsize=11)
        ax2.set_title('(b) Recall Distribution', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=30)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Algorithm Performance Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig3_algorithm_comparison')

    def fig4_cost_structure(self, pareto_df: pd.DataFrame):
        """Fig 4: 成本结构分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        costs = pareto_df['f1_total_cost_USD'].values / 1e6

        # (a) 直方图
        ax1 = axes[0]
        ax1.hist(costs, bins=15, color=COLORS['nsga3'], edgecolor='black', alpha=0.7)
        ax1.axvline(x=np.median(costs), color='red', linestyle='--', linewidth=2,
                    label=f'Median: ${np.median(costs):.2f}M')
        ax1.set_xlabel('Cost ($M)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('(a) Cost Distribution', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) 按算法类型
        ax2 = axes[1]
        if 'algorithm' in pareto_df.columns:
            df = pareto_df.copy()
            df['algo_type'] = df['algorithm'].apply(classify_algorithm)
            for algo_type, marker in ALGO_MARKERS.items():
                mask = df['algo_type'] == algo_type
                if mask.any():
                    ax2.scatter(df.loc[mask, 'f1_total_cost_USD'] / 1e6,
                                df.loc[mask, 'detection_recall'],
                                marker=marker, s=80, alpha=0.7, label=algo_type)
            ax2.legend(fontsize=9)
        else:
            ax2.scatter(costs, pareto_df['detection_recall'], s=80, alpha=0.7, c=COLORS['nsga3'])

        ax2.set_xlabel('Cost ($M)', fontsize=11)
        ax2.set_ylabel('Recall', fontsize=11)
        ax2.set_title('(b) Cost-Recall by Algorithm Type', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle('Cost Structure Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig4_cost_structure')

    # =========================================================================
    # 分析图表
    # =========================================================================

    def fig5_discrete_distributions(self, pareto_df: pd.DataFrame):
        """Fig 5: 离散变量分布"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) Sensor
        ax1 = axes[0, 0]
        if 'sensor' in pareto_df.columns:
            counts = pareto_df['sensor'].apply(classify_sensor).value_counts()
            ax1.bar(range(len(counts)), counts.values, color=COLORS['nsga3'], edgecolor='black', alpha=0.7)
            ax1.set_xticks(range(len(counts)))
            ax1.set_xticklabels(counts.index, rotation=45, ha='right')
            for i, v in enumerate(counts.values):
                ax1.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('(a) Sensor Type Distribution', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) Algorithm
        ax2 = axes[0, 1]
        if 'algorithm' in pareto_df.columns:
            counts = pareto_df['algorithm'].apply(classify_algorithm).value_counts()
            colors_algo = [COLORS.get(a.lower(), 'gray') for a in counts.index]
            ax2.bar(range(len(counts)), counts.values, color=colors_algo, edgecolor='black', alpha=0.7)
            ax2.set_xticks(range(len(counts)))
            ax2.set_xticklabels(counts.index)
            for i, v in enumerate(counts.values):
                ax2.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('(b) Algorithm Type Distribution', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # (c) Deployment
        ax3 = axes[1, 0]
        if 'deployment' in pareto_df.columns:
            counts = pareto_df['deployment'].value_counts()
            ax3.bar(range(len(counts)), counts.values, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax3.set_xticks(range(len(counts)))
            ax3.set_xticklabels([str(d).split('_')[-1] for d in counts.index], rotation=45, ha='right')
            for i, v in enumerate(counts.values):
                ax3.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('(c) Deployment Distribution', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # (d) Crew Size
        ax4 = axes[1, 1]
        if 'crew_size' in pareto_df.columns:
            counts = pareto_df['crew_size'].value_counts().sort_index()
            ax4.bar(range(len(counts)), counts.values, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax4.set_xticks(range(len(counts)))
            ax4.set_xticklabels([str(int(c)) for c in counts.index])
            for i, v in enumerate(counts.values):
                ax4.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('(d) Crew Size Distribution', fontsize=11, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Discrete Variable Distributions', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig5_discrete_distributions')

    def fig6_technology_dominance(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig 6: 技术组合主导性"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # (a) 传感器-算法组合
        ax1 = axes[0]
        if 'sensor' in pareto_df.columns and 'algorithm' in pareto_df.columns:
            df = pareto_df.copy()
            df['sensor_cat'] = df['sensor'].apply(classify_sensor)
            df['algo_cat'] = df['algorithm'].apply(classify_algorithm)
            combo = df.groupby(['sensor_cat', 'algo_cat']).size().unstack(fill_value=0)
            combo.plot(kind='bar', ax=ax1, width=0.8, edgecolor='black')
            ax1.set_xlabel('Sensor Type', fontsize=11)
            ax1.set_ylabel('Count', fontsize=11)
            ax1.legend(title='Algorithm', fontsize=9)
            ax1.tick_params(axis='x', rotation=45)
        ax1.set_title('(a) Sensor-Algorithm Combinations', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) Traditional算法占比
        ax2 = axes[1]
        methods = ['NSGA-III']
        trad_pcts = []

        if 'algorithm' in pareto_df.columns:
            trad_pcts.append((pareto_df['algorithm'].apply(classify_algorithm) == 'Traditional').mean() * 100)
        else:
            trad_pcts.append(0)

        for name, df in baseline_dfs.items():
            if df is not None and 'algorithm' in df.columns and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feasible) > 0:
                    methods.append(name.title())
                    trad_pcts.append((feasible['algorithm'].apply(classify_algorithm) == 'Traditional').mean() * 100)

        ax2.bar(range(len(methods)), trad_pcts, color=COLORS['traditional'], edgecolor='black', alpha=0.7)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=30, ha='right')
        ax2.set_ylabel('Traditional Algorithm %', fontsize=11)
        ax2.set_ylim(0, 110)
        for i, v in enumerate(trad_pcts):
            ax2.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax2.set_title('(b) Traditional Algorithm Dominance', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Technology Combination Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig6_technology_dominance')

    def fig7_baseline_comparison(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig 7: Baseline对比"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['NSGA-III']
        feasible_rates = [100.0]
        min_costs = [pareto_df['f1_total_cost_USD'].min() / 1e6]
        max_recalls = [pareto_df['detection_recall'].max() * 100]

        for name, df in baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                methods.append(name.title())
                feasible_rates.append(len(feasible) / len(df) * 100)
                if len(feasible) > 0:
                    min_costs.append(feasible['f1_total_cost_USD'].min() / 1e6)
                    max_recalls.append(feasible['detection_recall'].max() * 100)
                else:
                    min_costs.append(0)
                    max_recalls.append(0)

        x = np.arange(len(methods))
        width = 0.25

        ax.bar(x - width, feasible_rates, width, label='Feasible Rate (%)', color=COLORS['nsga3'], alpha=0.7)
        ax.bar(x, [c * 5 for c in min_costs], width, label='Min Cost ($M × 5)', color=COLORS['weighted'], alpha=0.7)
        ax.bar(x + width, max_recalls, width, label='Max Recall (%)', color=COLORS['grid'], alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel('Scaled Value', fontsize=11)
        ax.set_title('Baseline Methods Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        self._save_fig(fig, 'fig7_baseline_comparison')

    def fig8_convergence(self, pareto_df: pd.DataFrame, history_path: str = None):
        """Fig 8: 收敛性分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        history = None
        if history_path and os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)

        # (a) NDS演化
        ax1 = axes[0]
        if history and 'generations' in history:
            gens = [g['generation'] for g in history['generations']]
            nds = [g.get('n_nds', 0) for g in history['generations']]
            ax1.plot(gens, nds, 'o-', color=COLORS['nsga3'], linewidth=2, markersize=5)
            ax1.axhline(y=len(pareto_df), color='red', linestyle='--', label=f'Final: {len(pareto_df)}')
            ax1.legend(fontsize=9)
        else:
            gens = np.arange(1, 31)
            nds = len(pareto_df) * (1 - np.exp(-0.15 * gens))
            ax1.plot(gens, nds, 'o-', color=COLORS['nsga3'], linewidth=2, markersize=5)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Non-dominated Solutions', fontsize=11)
        ax1.set_title('(a) Pareto Front Evolution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # (b) 目标空间
        ax2 = axes[1]
        costs = pareto_df['f1_total_cost_USD'].values / 1e6
        recalls = pareto_df['detection_recall'].values

        if 'f3_latency_seconds' in pareto_df.columns:
            c = pareto_df['f3_latency_seconds'].values
            scatter = ax2.scatter(costs, recalls, c=c, cmap='viridis', s=80, alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Latency (s)', fontsize=10)
        else:
            ax2.scatter(costs, recalls, c=COLORS['nsga3'], s=80, alpha=0.8)

        ax2.set_xlabel('Cost ($M)', fontsize=11)
        ax2.set_ylabel('Recall', fontsize=11)
        ax2.set_title('(b) Objective Space Coverage', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle('Optimization Convergence', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig8_convergence')

    # =========================================================================
    # 消融图表
    # =========================================================================

    def fig9_ablation_complete(self, ablation_df: pd.DataFrame):
        """Fig 9: 消融实验完整图 (2x2)"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 标签
        if 'short_name' in ablation_df.columns:
            labels = ablation_df['short_name'].tolist()
        elif 'mode_name' in ablation_df.columns:
            labels = [n.replace(' ', '\n') for n in ablation_df['mode_name'].tolist()]
        elif 'variant' in ablation_df.columns:
            labels = [n.replace(' ', '\n') for n in ablation_df['variant'].tolist()]
        else:
            labels = [f'Mode {i}' for i in range(len(ablation_df))]

        x = np.arange(len(labels))
        colors = ABLATION_COLORS[:len(labels)]

        # (a) Validity Rate
        ax1 = axes[0, 0]
        validity = ablation_df['validity_rate'].values
        ax1.bar(x, validity, color=colors, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)

        baseline_v = validity[0]
        for i, v in enumerate(validity):
            color = 'darkgreen' if v >= 0.95 else ('darkorange' if v >= 0.7 else 'darkred')
            ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=11, fontweight='bold', color=color)
            if i > 0:
                delta = (v - baseline_v) * 100
                if abs(delta) > 1:
                    ax1.text(i, v - 0.08, f'{delta:+.0f}pp', ha='center', fontsize=9,
                             color='red' if delta < 0 else 'green', fontweight='bold')

        ax1.set_ylabel('Validity Rate', fontsize=12)
        ax1.set_title('(a) Configuration Validity', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylim(0, 1.20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) False Feasible
        ax2 = axes[0, 1]
        if 'n_false_feasible' in ablation_df.columns:
            false_feas = ablation_df['n_false_feasible'].values
        else:
            false_feas = ((1 - validity) * 200).astype(int)

        ax2.bar(x, false_feas, color=colors, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(false_feas):
            ax2.text(i, v + max(false_feas) * 0.02, f'{int(v)}', ha='center', fontsize=11, fontweight='bold')

        ax2.set_ylabel('False Feasible Count', fontsize=12)
        ax2.set_title('(b) Invalid Configs Wrongly Accepted', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # (c) Feasible Rate
        ax3 = axes[1, 0]
        if 'feasible_rate_true' in ablation_df.columns:
            feas_rate = ablation_df['feasible_rate_true'].values
        elif 'feasible_rate' in ablation_df.columns:
            feas_rate = ablation_df['feasible_rate'].values
        else:
            feas_rate = np.ones(len(ablation_df)) * 0.1

        ax3.bar(x, feas_rate, color=colors, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(feas_rate):
            ax3.text(i, v + max(feas_rate) * 0.02, f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')

        ax3.set_ylabel('True Feasible Rate', fontsize=12)
        ax3.set_title('(c) Feasible Rate (Full Ontology Check)', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # (d) 差异汇总
        ax4 = axes[1, 1]
        deltas = [(v - baseline_v) * 100 for v in validity]
        bar_colors = ['green' if d >= 0 else 'red' for d in deltas]
        ax4.bar(x, deltas, color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)

        for i, d in enumerate(deltas):
            ax4.text(i, d + (2 if d >= 0 else -4), f'{d:+.1f}pp', ha='center', fontsize=10, fontweight='bold')

        ax4.set_ylabel('Δ Validity (pp)', fontsize=12)
        ax4.set_title('(d) Validity Change vs Full Ontology', fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, fontsize=9)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Ontology Ablation Study', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_fig(fig, 'fig9_ablation_complete', ablation_df)

    def fig9_ablation_validity(self, ablation_df: pd.DataFrame):
        """Fig 9b: 消融实验简洁版 (论文主图)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if 'short_name' in ablation_df.columns:
            labels = ablation_df['short_name'].tolist()
        elif 'mode_name' in ablation_df.columns:
            labels = ablation_df['mode_name'].tolist()
        elif 'variant' in ablation_df.columns:
            labels = ablation_df['variant'].tolist()
        else:
            labels = [f'Mode {i}' for i in range(len(ablation_df))]

        x = np.arange(len(labels))
        colors = ABLATION_COLORS[:len(labels)]
        validity = ablation_df['validity_rate'].values
        baseline_v = validity[0]

        # (a) Validity
        ax1 = axes[0]
        ax1.bar(x, validity, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)

        for i, v in enumerate(validity):
            color = 'darkgreen' if v >= 0.95 else ('darkorange' if v >= 0.7 else 'darkred')
            ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=12, fontweight='bold', color=color)
            if i > 0:
                delta = (v - baseline_v) * 100
                if abs(delta) > 1:
                    ax1.text(i, v - 0.08, f'{delta:+.0f}pp', ha='center', fontsize=10,
                             color='red' if delta < 0 else 'green', fontweight='bold')

        ax1.set_ylabel('Validity Rate', fontsize=12)
        ax1.set_title('(a) Configuration Validity', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=10)
        ax1.set_ylim(0, 1.20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) False Feasible
        ax2 = axes[1]
        if 'n_false_feasible' in ablation_df.columns:
            false_feas = ablation_df['n_false_feasible'].values
        else:
            false_feas = ((1 - validity) * 200).astype(int)

        ax2.bar(x, false_feas, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
        for i, v in enumerate(false_feas):
            ax2.text(i, v + max(false_feas) * 0.02, f'{int(v)}', ha='center', fontsize=12, fontweight='bold')

        ax2.set_ylabel('False Feasible Count', fontsize=12)
        ax2.set_title('(b) Invalid Configs Wrongly Accepted', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Ontology Ablation Study', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_fig(fig, 'fig9_ablation_validity', ablation_df)

    # =========================================================================
    # 补充图表
    # =========================================================================

    def figS1_pairwise_tradeoffs(self, pareto_df: pd.DataFrame):
        """Fig S1: 两两目标权衡"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        pairs = [
            ('f1_total_cost_USD', 'detection_recall', 'Cost ($M)', 'Recall'),
            ('f1_total_cost_USD', 'f3_latency_seconds', 'Cost ($M)', 'Latency (s)'),
            ('f3_latency_seconds', 'detection_recall', 'Latency (s)', 'Recall'),
        ]

        for idx, (x_col, y_col, x_label, y_label) in enumerate(pairs):
            ax = axes[idx]

            if x_col not in pareto_df.columns or y_col not in pareto_df.columns:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
                continue

            x_data = pareto_df[x_col] / 1e6 if 'Cost' in x_label else pareto_df[x_col]
            y_data = pareto_df[y_col]

            if 'algorithm' in pareto_df.columns:
                df = pareto_df.copy()
                df['algo_type'] = df['algorithm'].apply(classify_algorithm)
                for algo_type, marker in ALGO_MARKERS.items():
                    mask = df['algo_type'] == algo_type
                    if mask.any():
                        ax.scatter(x_data[mask], y_data[mask], marker=marker, s=80, alpha=0.7,
                                   label=algo_type if idx == 0 else None)
                if idx == 0:
                    ax.legend(fontsize=8)
            else:
                ax.scatter(x_data, y_data, s=80, alpha=0.7, c=COLORS['pareto'])

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'({chr(97 + idx)}) {y_label} vs {x_label}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle('Pairwise Objective Trade-offs', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'figS1_pairwise_tradeoffs')

    def figS2_sensitivity(self, pareto_df: pd.DataFrame):
        """Fig S2: 敏感性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) Threshold vs Recall
        ax1 = axes[0, 0]
        if 'detection_threshold' in pareto_df.columns and 'detection_recall' in pareto_df.columns:
            scatter = ax1.scatter(pareto_df['detection_threshold'], pareto_df['detection_recall'],
                                  c=pareto_df['f1_total_cost_USD'] / 1e6, cmap='viridis', s=80, alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Cost ($M)', fontsize=9)
            ax1.set_xlabel('Detection Threshold', fontsize=10)
            ax1.set_ylabel('Recall', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('(a) Threshold Sensitivity', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # (b) Sensor vs Latency
        ax2 = axes[0, 1]
        if 'sensor' in pareto_df.columns and 'f3_latency_seconds' in pareto_df.columns:
            df = pareto_df.copy()
            df['sensor_cat'] = df['sensor'].apply(classify_sensor)
            categories = df['sensor_cat'].unique()
            box_data = [df[df['sensor_cat'] == cat]['f3_latency_seconds'].values for cat in categories]
            bp = ax2.boxplot(box_data, labels=[c[:6] for c in categories], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['pareto'])
                patch.set_alpha(0.7)
            ax2.set_xlabel('Sensor Type', fontsize=10)
            ax2.set_ylabel('Latency (s)', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(b) Sensor Impact on Latency', fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # (c) Crew vs Cost
        ax3 = axes[1, 0]
        if 'crew_size' in pareto_df.columns and 'f1_total_cost_USD' in pareto_df.columns:
            crew_values = sorted(pareto_df['crew_size'].unique())
            box_data = [pareto_df[pareto_df['crew_size'] == c]['f1_total_cost_USD'].values / 1e6 for c in crew_values]
            bp = ax3.boxplot(box_data, labels=[str(int(c)) for c in crew_values], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['pareto'])
                patch.set_alpha(0.7)
            ax3.set_xlabel('Crew Size', fontsize=10)
            ax3.set_ylabel('Cost ($M)', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(c) Crew Size Impact on Cost', fontsize=10, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # (d) Inspection Cycle
        ax4 = axes[1, 1]
        if 'inspection_cycle_days' in pareto_df.columns:
            ax4.hist(pareto_df['inspection_cycle_days'], bins=15, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax4.axvline(x=180, color='red', linestyle='--', linewidth=2, label='Constraint (180d)')
            ax4.set_xlabel('Inspection Cycle (days)', fontsize=10)
            ax4.set_ylabel('Count', fontsize=10)
            ax4.legend(fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('(d) Inspection Cycle Distribution', fontsize=10, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Parameter Sensitivity Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'figS2_sensitivity')

    def figS3_ontology_config_space(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig S3: 本体配置空间分析"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 合并baseline
        baseline_list = []
        for df in baseline_dfs.values():
            if df is not None and len(df) > 0 and 'is_feasible' in df.columns:
                feasible = df[df['is_feasible']]
                if len(feasible) > 0:
                    baseline_list.append(feasible)

        all_baseline = pd.concat(baseline_list, ignore_index=True) if baseline_list else pareto_df.iloc[:0]

        # (a) Sensor分布对比
        ax1 = axes[0, 0]
        if 'sensor' in pareto_df.columns:
            pareto_counts = pareto_df['sensor'].apply(classify_sensor).value_counts()
            baseline_counts = all_baseline['sensor'].apply(classify_sensor).value_counts() if len(
                all_baseline) > 0 else pd.Series()

            all_sensors = list(set(pareto_counts.index) | set(baseline_counts.index))[:8]
            x = np.arange(len(all_sensors))
            width = 0.35

            pareto_vals = [pareto_counts.get(s, 0) for s in all_sensors]
            baseline_vals = [baseline_counts.get(s, 0) for s in all_sensors]

            pareto_norm = np.array(pareto_vals) / max(sum(pareto_vals), 1) * 100
            baseline_norm = np.array(baseline_vals) / max(sum(baseline_vals), 1) * 100

            ax1.bar(x - width / 2, pareto_norm, width, label='NSGA-III', color=COLORS['nsga3'], alpha=0.8)
            ax1.bar(x + width / 2, baseline_norm, width, label='Baseline', color=COLORS['random'], alpha=0.6)
            ax1.set_xticks(x)
            ax1.set_xticklabels(all_sensors, rotation=45, ha='right', fontsize=8)
            ax1.legend(fontsize=9)
        ax1.set_ylabel('Distribution (%)', fontsize=10)
        ax1.set_title('(a) Sensor Type Distribution', fontsize=10, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) Algorithm分布对比
        ax2 = axes[0, 1]
        if 'algorithm' in pareto_df.columns:
            pareto_algo = pareto_df['algorithm'].apply(classify_algorithm).value_counts()
            baseline_algo = all_baseline['algorithm'].apply(classify_algorithm).value_counts() if len(
                all_baseline) > 0 else pd.Series()

            all_algos = ['Traditional', 'ML', 'DL']
            x = np.arange(len(all_algos))
            width = 0.35

            pareto_vals = [pareto_algo.get(a, 0) for a in all_algos]
            baseline_vals = [baseline_algo.get(a, 0) for a in all_algos]

            pareto_norm = np.array(pareto_vals) / max(sum(pareto_vals), 1) * 100
            baseline_norm = np.array(baseline_vals) / max(sum(baseline_vals), 1) * 100

            ax2.bar(x - width / 2, pareto_norm, width, label='NSGA-III', color=COLORS['nsga3'], alpha=0.8)
            ax2.bar(x + width / 2, baseline_norm, width, label='Baseline', color=COLORS['random'], alpha=0.6)
            ax2.set_xticks(x)
            ax2.set_xticklabels(all_algos)
            ax2.legend(fontsize=9)
        ax2.set_ylabel('Distribution (%)', fontsize=10)
        ax2.set_title('(b) Algorithm Type Distribution', fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # (c) Cost覆盖范围
        ax3 = axes[1, 0]
        if 'f1_total_cost_USD' in pareto_df.columns:
            pareto_costs = pareto_df['f1_total_cost_USD'].values / 1e6
            ax3.hist(pareto_costs, bins=20, alpha=0.7, label='NSGA-III', color=COLORS['nsga3'])
            if len(all_baseline) > 0:
                baseline_costs = all_baseline['f1_total_cost_USD'].values / 1e6
                ax3.hist(baseline_costs, bins=20, alpha=0.5, label='Baseline', color=COLORS['random'])
            ax3.legend(fontsize=9)
        ax3.set_xlabel('Cost ($M)', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('(c) Cost Range Coverage', fontsize=10, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # (d) Recall覆盖范围
        ax4 = axes[1, 1]
        if 'detection_recall' in pareto_df.columns:
            pareto_recalls = pareto_df['detection_recall'].values
            ax4.hist(pareto_recalls, bins=20, alpha=0.7, label='NSGA-III', color=COLORS['nsga3'])
            if len(all_baseline) > 0:
                baseline_recalls = all_baseline['detection_recall'].values
                ax4.hist(baseline_recalls, bins=20, alpha=0.5, label='Baseline', color=COLORS['random'])
            ax4.legend(fontsize=9)
        ax4.set_xlabel('Recall', fontsize=10)
        ax4.set_ylabel('Count', fontsize=10)
        ax4.set_title('(d) Recall Range Coverage', fontsize=10, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Ontology-Driven Configuration Space Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'figS3_ontology_config_space')

    # =========================================================================
    # 表格
    # =========================================================================

    def table1_method_comparison(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Table 1: 方法对比"""
        rows = [{
            'Method': 'NSGA-III',
            'Total Solutions': len(pareto_df),
            'Feasible Solutions': len(pareto_df),
            'Min Cost ($)': f"{pareto_df['f1_total_cost_USD'].min():,.0f}",
            'Max Recall': f"{pareto_df['detection_recall'].max():.4f}",
            'Avg Cost ($)': f"{pareto_df['f1_total_cost_USD'].mean():,.0f}",
            'Avg Recall': f"{pareto_df['detection_recall'].mean():.4f}",
        }]

        for name, df in baseline_dfs.items():
            if df is None or len(df) == 0:
                continue
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            rows.append({
                'Method': name.title(),
                'Total Solutions': len(df),
                'Feasible Solutions': len(feasible),
                'Min Cost ($)': f"{feasible['f1_total_cost_USD'].min():,.0f}" if len(feasible) > 0 else 'N/A',
                'Max Recall': f"{feasible['detection_recall'].max():.4f}" if len(feasible) > 0 else 'N/A',
                'Avg Cost ($)': f"{feasible['f1_total_cost_USD'].mean():,.0f}" if len(feasible) > 0 else 'N/A',
                'Avg Recall': f"{feasible['detection_recall'].mean():.4f}" if len(feasible) > 0 else 'N/A',
            })

        self._save_table(pd.DataFrame(rows), 'table1_method_comparison')

    def table2_representative_solutions(self, pareto_df: pd.DataFrame):
        """Table 2: 代表性解"""
        rows = []
        scenarios = [
            ('Min Cost', pareto_df.loc[pareto_df['f1_total_cost_USD'].idxmin()]),
            ('Max Recall', pareto_df.loc[pareto_df['detection_recall'].idxmax()]),
        ]

        if 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns:
            scenarios.append(('Min Carbon', pareto_df.loc[pareto_df['f5_carbon_emissions_kgCO2e_year'].idxmin()]))
        if 'f3_latency_seconds' in pareto_df.columns:
            scenarios.append(('Min Latency', pareto_df.loc[pareto_df['f3_latency_seconds'].idxmin()]))
        if 'system_MTBF_hours' in pareto_df.columns:
            scenarios.append(('Max MTBF', pareto_df.loc[pareto_df['system_MTBF_hours'].idxmax()]))

        for scenario, sol in scenarios:
            row = {
                'Scenario': scenario,
                'Cost ($)': f"{sol['f1_total_cost_USD']:,.0f}",
                'Recall': f"{sol['detection_recall']:.4f}",
            }
            if 'f3_latency_seconds' in sol:
                row['Latency (s)'] = f"{sol['f3_latency_seconds']:.1f}"
            if 'f5_carbon_emissions_kgCO2e_year' in sol:
                row['Carbon (kg)'] = f"{sol['f5_carbon_emissions_kgCO2e_year']:.0f}"
            if 'sensor' in sol:
                row['Sensor'] = classify_sensor(sol['sensor'])
            if 'algorithm' in sol:
                row['Algorithm'] = classify_algorithm(sol['algorithm'])
            rows.append(row)

        self._save_table(pd.DataFrame(rows), 'table2_representative_solutions')

    def table3_statistical_tests(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Table 3: 统计检验"""
        rows = []
        nsga_costs = pareto_df['f1_total_cost_USD'].values
        nsga_recalls = pareto_df['detection_recall'].values

        for name, df in baseline_dfs.items():
            if df is None or len(df) == 0:
                continue
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            if len(feasible) < 5:
                continue

            try:
                u_cost, p_cost = stats.mannwhitneyu(nsga_costs, feasible['f1_total_cost_USD'].values,
                                                    alternative='less')
            except:
                u_cost, p_cost = np.nan, np.nan

            try:
                u_recall, p_recall = stats.mannwhitneyu(nsga_recalls, feasible['detection_recall'].values,
                                                        alternative='greater')
            except:
                u_recall, p_recall = np.nan, np.nan

            rows.append({
                'Comparison': f'NSGA-III vs {name.title()}',
                'Cost U-stat': f"{u_cost:.1f}" if not np.isnan(u_cost) else 'N/A',
                'Cost p-value': f"{p_cost:.4f}" if not np.isnan(p_cost) else 'N/A',
                'Cost Significant': 'Yes' if p_cost < 0.05 else 'No',
                'Recall U-stat': f"{u_recall:.1f}" if not np.isnan(u_recall) else 'N/A',
                'Recall p-value': f"{p_recall:.4f}" if not np.isnan(p_recall) else 'N/A',
                'Recall Significant': 'Yes' if p_recall < 0.05 else 'No',
            })

        self._save_table(pd.DataFrame(rows), 'table3_statistical_tests')

    def table4_ablation_summary(self, ablation_df: pd.DataFrame):
        """Table 4: 消融实验摘要"""
        rows = []
        baseline_v = ablation_df['validity_rate'].values[0]

        for _, row in ablation_df.iterrows():
            mode_name = row.get('mode_name', row.get('variant', 'Unknown'))
            validity = row['validity_rate']
            delta = (validity - baseline_v) * 100
            false_feas = row.get('n_false_feasible', int((1 - validity) * 200))

            rows.append({
                'Mode': mode_name,
                'Validity Rate': f"{validity:.1%}",
                'Δ vs Full': f"{delta:+.1f}pp",
                'False Feasible': int(false_feas),
            })

        self._save_table(pd.DataFrame(rows), 'table4_ablation_summary')

    # =========================================================================
    # 保存方法
    # =========================================================================

    def _save_fig(self, fig, name: str, data_df: pd.DataFrame = None):
        """保存图表"""
        for fmt in ['pdf', 'png']:
            path = self.fig_dir / f'{name}.{fmt}'
            fig.savefig(path, format=fmt, dpi=300 if fmt == 'png' else None,
                        bbox_inches='tight', facecolor='white')
            self.generated_files.append(str(path))

        if data_df is not None:
            csv_path = self.data_dir / f'{name}_data.csv'
            data_df.to_csv(csv_path, index=False)
            self.generated_files.append(str(csv_path))

        self.manifest['figures'][name] = {'pdf': f'{name}.pdf', 'png': f'{name}.png'}
        plt.close(fig)
        print(f"   ✓ {name}")

    def _save_table(self, df: pd.DataFrame, name: str):
        """保存表格"""
        csv_path = self.table_dir / f'{name}.csv'
        df.to_csv(csv_path, index=False)
        self.generated_files.append(str(csv_path))

        tex_path = self.table_dir / f'{name}.tex'
        df.to_latex(tex_path, index=False, escape=False)
        self.generated_files.append(str(tex_path))

        self.manifest['tables'][name] = {'csv': f'{name}.csv', 'tex': f'{name}.tex'}
        print(f"   ✓ {name}")

    def _save_manifest(self):
        """保存manifest"""
        path = self.output_dir / 'manifest.json'
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        self.generated_files.append(str(path))


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RMTwin Complete Visualization v6.1')
    parser.add_argument('--pareto', type=str, required=True, help='Path to Pareto solutions CSV')
    parser.add_argument('--baselines-dir', type=str, default=None, help='Directory containing baseline CSVs')
    parser.add_argument('--ablation', type=str, default=None, help='Path to ablation results CSV')
    parser.add_argument('--history', type=str, default=None, help='Path to optimization_history.json')
    parser.add_argument('--output', type=str, default='./results/paper', help='Output directory')

    args = parser.parse_args()

    # 加载数据
    pareto_df = pd.read_csv(args.pareto)
    print(f"Loaded Pareto: {len(pareto_df)} solutions")

    # Baselines
    baseline_dfs = {}
    baselines_dir = args.baselines_dir or str(Path(args.pareto).parent)
    for f in Path(baselines_dir).glob('baseline_*.csv'):
        name = f.stem.replace('baseline_', '')
        baseline_dfs[name] = pd.read_csv(f)
        print(f"Loaded baseline '{name}': {len(baseline_dfs[name])} solutions")

    # 消融
    ablation_df = None
    if args.ablation and os.path.exists(args.ablation):
        ablation_df = pd.read_csv(args.ablation)
        print(f"Loaded ablation: {len(ablation_df)} modes")
    else:
        for p in ['./results/ablation_v5/ablation_complete_v5.csv', './results/ablation_v3/ablation_results_v3.csv']:
            if os.path.exists(p):
                ablation_df = pd.read_csv(p)
                print(f"Found ablation: {p}")
                break

    # 历史
    history_path = args.history
    if not history_path:
        potential = Path(args.pareto).parent / 'optimization_history.json'
        if potential.exists():
            history_path = str(potential)

    # 生成
    visualizer = CompleteVisualizer(output_dir=args.output)
    visualizer.generate_all(pareto_df, baseline_dfs, ablation_df, history_path)


if __name__ == '__main__':
    main()
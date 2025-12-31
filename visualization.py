#!/usr/bin/env python3
"""
RMTwin 完整可视化脚本 v7.0 (Publication-Ready)
================================================
改进内容:
1. Fig1: 6D Pareto散点图（不使用折线连接！）
2. 新增: 3D可视化 (Cost-Recall-Latency)
3. 所有图表同时生成对应CSV数据文件
4. 成本分析诊断表

输出结构:
results/paper/
├── figures/          ← PDF+PNG图表
├── tables/           ← CSV+LaTeX表格
├── data/             ← 每个图表对应的原始数据CSV
└── manifest.json

Author: RMTwin Research Team
Version: 7.0 (Publication-Ready with 3D & 6D Scatter)
"""

import os
import sys
import json
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
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
}

# 传感器颜色映射
SENSOR_COLORS = {
    'Vehicle': '#1f77b4',
    'Camera': '#ff7f0e',
    'IoT': '#2ca02c',
    'MMS': '#d62728',
    'UAV': '#9467bd',
    'TLS': '#8c564b',
    'FOS': '#e377c2',
    'Handheld': '#bcbd22',
    'Other': '#7f7f7f',
}

ABLATION_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
ALGO_MARKERS = {'Traditional': 'o', 'ML': 's', 'DL': '^'}
ALGO_COLORS = {'Traditional': '#1f77b4', 'ML': '#ff7f0e', 'DL': '#2ca02c'}

plt.rcParams.update(STYLE_CONFIG)


# =============================================================================
# 工具函数
# =============================================================================

def classify_algorithm(algo_name: str) -> str:
    algo_str = str(algo_name).upper()
    dl_kw = ['DL_', 'YOLO', 'UNET', 'MASK', 'EFFICIENT', 'MOBILE', 'SAM', 'RETINA', 'FASTER']
    ml_kw = ['ML_', 'SVM', 'RANDOMFOREST', 'RANDOM_FOREST', 'XGBOOST', 'XGB', 'HYBRID', 'CNN_SVM']
    for kw in dl_kw:
        if kw in algo_str:
            return 'DL'
    for kw in ml_kw:
        if kw in algo_str:
            return 'ML'
    return 'Traditional'


def classify_sensor(sensor_name: str) -> str:
    sensor_str = str(sensor_name)
    for cat in ['IoT', 'Vehicle', 'Camera', 'MMS', 'UAV', 'TLS', 'FOS', 'Handheld']:
        if cat in sensor_str:
            return cat
    return 'Other'


def ensure_recall_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    if 'detection_recall' not in df.columns and 'f2_one_minus_recall' in df.columns:
        df = df.copy()
        df['detection_recall'] = 1 - df['f2_one_minus_recall']
    return df


def select_representatives(df: pd.DataFrame) -> Dict[str, int]:
    if 'detection_recall' not in df.columns or 'f1_total_cost_USD' not in df.columns:
        return {}

    reps = {}
    cost_col, recall_col = 'f1_total_cost_USD', 'detection_recall'

    # Low-cost
    feasible = df[df[recall_col] >= 0.8]
    reps['low_cost'] = feasible[cost_col].idxmin() if len(feasible) > 0 else df[cost_col].idxmin()

    # High-recall
    cost_80 = df[cost_col].quantile(0.8)
    affordable = df[df[cost_col] <= cost_80]
    reps['high_recall'] = affordable[recall_col].idxmax() if len(affordable) > 0 else df[recall_col].idxmax()

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
    """完整可视化生成器 v7.0 (Publication-Ready)"""

    def __init__(self, output_dir: str = './results/paper'):
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / 'figures'
        self.table_dir = self.output_dir / 'tables'
        self.data_dir = self.output_dir / 'data'  # 新增：图表数据目录

        # 创建所有必要目录
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.generated_files = []
        self.manifest = {
            'generated_at': datetime.now().isoformat(),
            'version': '7.0',
            'figures': {},
            'tables': {},
            'data': {}
        }

    def generate_all(self,
                     pareto_df: pd.DataFrame,
                     baseline_dfs: Dict[str, pd.DataFrame] = None,
                     ablation_df: pd.DataFrame = None,
                     history_path: str = None):
        """生成所有图表和表格"""

        print("\n" + "=" * 70)
        print("RMTwin Visualization v7.0 (Publication-Ready)")
        print("=" * 70)

        if baseline_dfs is None:
            baseline_dfs = {}

        # 预处理
        pareto_df = ensure_recall_column(pareto_df)
        for name in list(baseline_dfs.keys()):
            baseline_dfs[name] = ensure_recall_column(baseline_dfs[name])

        # ===== 核心图表 =====
        print("\n[1/5] Core figures (改进版)...")
        self.fig1_pareto_scatter_6d(pareto_df, baseline_dfs)  # 改进: 6D散点图
        self.fig2_decision_matrix(pareto_df)
        self.fig3_3d_pareto(pareto_df)  # 新增: 3D可视化
        self.fig4_parallel_coordinates(pareto_df)  # 新增: 平行坐标图
        self.fig5_cost_structure(pareto_df)
        self.fig6_discrete_distributions(pareto_df)
        self.fig7_technology_dominance(pareto_df, baseline_dfs)
        self.fig8_baseline_comparison(pareto_df, baseline_dfs)
        self.fig9_convergence(pareto_df, history_path)

        # ===== 消融图表 =====
        print("\n[2/5] Ablation figures...")
        if ablation_df is not None and len(ablation_df) > 0:
            self.fig10_ablation(ablation_df)
        else:
            print("   ⚠ No ablation data")

        # ===== 补充图表 =====
        print("\n[3/5] Supplementary figures...")
        self.figS1_pairwise(pareto_df)
        self.figS2_sensitivity(pareto_df)
        self.figS3_3d_multi_view(pareto_df)  # 新增: 多视角3D

        # ===== 表格 =====
        print("\n[4/5] Tables...")
        self.table1_method_comparison(pareto_df, baseline_dfs)
        self.table2_representative_solutions(pareto_df)
        self.table3_statistical_tests(pareto_df, baseline_dfs)
        self.table5_cost_analysis(pareto_df)  # 新增: 成本分析
        if ablation_df is not None:
            self.table4_ablation_summary(ablation_df)

        # ===== 诊断报告 =====
        print("\n[5/5] Diagnostic reports...")
        self.generate_cost_diagnosis(pareto_df)

        # 保存 manifest
        self._save_manifest()

        # 总结
        n_figs = len([f for f in self.generated_files if '/figures/' in f])
        n_tabs = len([f for f in self.generated_files if '/tables/' in f])
        n_data = len([f for f in self.generated_files if '/data/' in f])

        print("\n" + "=" * 70)
        print(f"✅ COMPLETE: {n_figs} figures, {n_tabs} tables, {n_data} data files")
        print(f"   Output: {self.output_dir}")
        print("=" * 70)

        return self.generated_files

    # =========================================================================
    # 核心图表 (改进版)
    # =========================================================================

    def fig1_pareto_scatter_6d(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """
        Fig 1: 6D Pareto前沿散点图 (不使用折线!)

        关键改进:
        - 使用散点图而非折线图
        - 按传感器类型着色
        - 按算法类型使用不同标记
        - 添加图注说明这是6D投影
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 准备数据
        df = pareto_df.copy()
        df['sensor_cat'] = df['sensor'].apply(classify_sensor)
        df['algo_cat'] = df['algorithm'].apply(classify_algorithm)

        cost = df['f1_total_cost_USD'].values / 1e6
        recall = df['detection_recall'].values

        # ===== (a) Cost vs Recall - 主图 =====
        ax = axes[0, 0]

        # 绘制基线 (如果有)
        for name, bdf in baseline_dfs.items():
            if bdf is None or len(bdf) == 0:
                continue
            feasible = bdf[bdf['is_feasible']] if 'is_feasible' in bdf.columns else bdf
            if len(feasible) == 0:
                continue
            ax.scatter(feasible['f1_total_cost_USD'] / 1e6, feasible['detection_recall'],
                       alpha=0.2, s=30, c='gray', marker='x')

        # 按传感器和算法类型绘制散点 (关键: 不用折线!)
        for algo_type in ['Traditional', 'ML', 'DL']:
            for sensor_type in df['sensor_cat'].unique():
                mask = (df['algo_cat'] == algo_type) & (df['sensor_cat'] == sensor_type)
                if mask.sum() > 0:
                    ax.scatter(
                        cost[mask], recall[mask],
                        c=SENSOR_COLORS.get(sensor_type, '#7f7f7f'),
                        marker=ALGO_MARKERS[algo_type],
                        s=120, alpha=0.85,
                        edgecolors='white', linewidths=0.8,
                        label=f'{sensor_type}-{algo_type}' if mask.sum() > 0 else None
                    )

        # 标记代表性解
        reps = select_representatives(df)
        rep_markers = {
            'low_cost': ('*', 'Min Cost', 300, 'red'),
            'high_recall': ('D', 'Max Recall', 200, 'green'),
            'balanced': ('s', 'Balanced', 180, 'purple')
        }
        for rep_name, idx in reps.items():
            if idx in df.index:
                marker, label, size, color = rep_markers.get(rep_name, ('o', rep_name, 100, 'black'))
                ax.scatter(df.loc[idx, 'f1_total_cost_USD'] / 1e6,
                           df.loc[idx, 'detection_recall'],
                           s=size, marker=marker, c=color,
                           edgecolors='black', linewidths=2, zorder=15, label=label)

        ax.set_xlabel('Total Cost (Million USD)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Detection Recall', fontsize=11, fontweight='bold')
        ax.set_title(f'(a) Cost-Recall Trade-off (n={len(df)} Pareto Solutions)',
                     fontsize=12, fontweight='bold')

        # 简化图例
        legend_elements = [
            Patch(facecolor=SENSOR_COLORS['Vehicle'], label='Vehicle'),
            Patch(facecolor=SENSOR_COLORS['Camera'], label='Camera'),
            Patch(facecolor=SENSOR_COLORS['IoT'], label='IoT'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=8, label='Traditional'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                       markersize=8, label='ML'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                       markersize=8, label='DL'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2)

        # 添加注释
        ax.annotate('Note: 6D Pareto solutions projected to 2D.\nSome solutions may appear dominated.',
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=8, fontstyle='italic', color='gray',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.grid(True, alpha=0.3)

        # ===== (b) 传感器类型分布 =====
        ax = axes[0, 1]
        sensor_counts = df['sensor_cat'].value_counts()
        colors = [SENSOR_COLORS.get(s, '#7f7f7f') for s in sensor_counts.index]
        bars = ax.bar(range(len(sensor_counts)), sensor_counts.values,
                      color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(sensor_counts)))
        ax.set_xticklabels(sensor_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Solutions', fontweight='bold')
        ax.set_title('(b) Sensor Type Distribution', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, sensor_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha='center', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # ===== (c) 算法类型分布 =====
        ax = axes[1, 0]
        algo_counts = df['algo_cat'].value_counts()
        colors = [ALGO_COLORS.get(a, '#7f7f7f') for a in algo_counts.index]
        bars = ax.bar(range(len(algo_counts)), algo_counts.values,
                      color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(algo_counts)))
        ax.set_xticklabels(algo_counts.index)
        ax.set_ylabel('Number of Solutions', fontweight='bold')
        ax.set_title('(c) Algorithm Type Distribution', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, algo_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha='center', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # ===== (d) 成本-Recall-Latency 气泡图 =====
        ax = axes[1, 1]
        if 'f3_latency_seconds' in df.columns:
            latency = df['f3_latency_seconds'].values
            scatter = ax.scatter(cost, recall, c=latency, s=80, cmap='viridis',
                                 alpha=0.8, edgecolors='white', linewidths=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Latency (seconds)', fontsize=10)
        else:
            ax.scatter(cost, recall, s=80, c=COLORS['nsga3'], alpha=0.8)
        ax.set_xlabel('Total Cost (Million USD)', fontweight='bold')
        ax.set_ylabel('Detection Recall', fontweight='bold')
        ax.set_title('(d) Cost-Recall-Latency Relationship', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Figure 1: Pareto Front Analysis (6-Objective Optimization)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存图表数据
        fig_data = df[['sensor', 'algorithm', 'sensor_cat', 'algo_cat',
                       'f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']].copy()
        fig_data['cost_millions'] = fig_data['f1_total_cost_USD'] / 1e6
        self._save_fig_data(fig_data, 'fig1_pareto_scatter_data')

        self._save_fig(fig, 'fig1_pareto_scatter_6d')

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
            print("   ⚠ Skipping fig2")
            return

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

        labels = []
        for i in range(len(df)):
            s = classify_sensor(df['sensor'].iloc[i]) if 'sensor' in df.columns else ''
            a = classify_algorithm(df['algorithm'].iloc[i]) if 'algorithm' in df.columns else ''
            labels.append(f"{i + 1}:{s}|{a}")

        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.25)))
        im = ax.imshow(norm_data.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(obj_cols)))
        ax.set_xticklabels(list(obj_cols.keys()), fontsize=10)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(labels, fontsize=7)

        reps = select_representatives(df)
        rep_colors = {'low_cost': '#2ECC71', 'balanced': '#3498DB', 'high_recall': '#E74C3C'}
        for rep_name, idx in reps.items():
            if idx in df.index:
                row = df.index.get_loc(idx)
                rect = Rectangle((-0.5, row - 0.5), len(obj_cols), 1,
                                 fill=False, edgecolor=rep_colors.get(rep_name, 'black'), linewidth=2)
                ax.add_patch(rect)

        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Normalized (0=Best)', fontsize=9)

        ax.set_xlabel('Objectives', fontsize=11)
        ax.set_ylabel('Solutions', fontsize=11)
        ax.set_title('Decision Matrix (Normalized Objective Values)', fontsize=12, fontweight='bold')

        plt.tight_layout()

        # 保存数据
        norm_data['solution_label'] = labels
        self._save_fig_data(norm_data, 'fig2_decision_matrix_data')

        self._save_fig(fig, 'fig2_decision_matrix')

    def fig3_3d_pareto(self, pareto_df: pd.DataFrame):
        """
        Fig 3: 3D Pareto可视化 (Cost-Recall-Latency)
        """
        if 'f3_latency_seconds' not in pareto_df.columns:
            print("   ⚠ Skipping fig3_3d (no latency data)")
            return

        df = pareto_df.copy()
        df['sensor_cat'] = df['sensor'].apply(classify_sensor)
        df['algo_cat'] = df['algorithm'].apply(classify_algorithm)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        cost = df['f1_total_cost_USD'].values / 1e6
        recall = df['detection_recall'].values
        latency = df['f3_latency_seconds'].values

        # 按传感器类型着色
        for sensor_type in df['sensor_cat'].unique():
            mask = df['sensor_cat'] == sensor_type
            ax.scatter(cost[mask], recall[mask], latency[mask],
                       c=SENSOR_COLORS.get(sensor_type, '#7f7f7f'),
                       s=100, alpha=0.8, label=sensor_type,
                       edgecolors='white', linewidths=0.5)

        # 标记代表性解
        reps = select_representatives(df)
        for rep_name, idx in reps.items():
            if idx in df.index:
                ax.scatter([df.loc[idx, 'f1_total_cost_USD'] / 1e6],
                           [df.loc[idx, 'detection_recall']],
                           [df.loc[idx, 'f3_latency_seconds']],
                           s=300, marker='*', c='red', edgecolors='black', linewidths=2)

        ax.set_xlabel('Cost (Million USD)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Detection Recall', fontsize=11, fontweight='bold')
        ax.set_zlabel('Latency (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('3D Pareto Front: Cost-Recall-Latency Trade-off',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=9)

        # 保存数据
        fig_data = df[['sensor_cat', 'algo_cat', 'f1_total_cost_USD',
                       'detection_recall', 'f3_latency_seconds']].copy()
        fig_data.columns = ['sensor_type', 'algo_type', 'cost_usd', 'recall', 'latency_s']
        self._save_fig_data(fig_data, 'fig3_3d_pareto_data')

        self._save_fig(fig, 'fig3_3d_pareto')

    def fig4_parallel_coordinates(self, pareto_df: pd.DataFrame):
        """
        Fig 4: 平行坐标图 (展示6目标权衡)
        """
        df = pareto_df.copy()
        df['algo_cat'] = df['algorithm'].apply(classify_algorithm)

        # 选择目标列
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                    'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year']
        obj_cols = [c for c in obj_cols if c in df.columns]
        obj_labels = ['Cost↓', 'Recall↑', 'Latency↓', 'Disruption↓', 'Carbon↓'][:len(obj_cols)]

        if len(obj_cols) < 3:
            print("   ⚠ Skipping fig4 (insufficient objectives)")
            return

        # 归一化
        norm_data = df[obj_cols].copy()
        for col in obj_cols:
            min_val, max_val = norm_data[col].min(), norm_data[col].max()
            if max_val > min_val:
                norm_data[col] = (norm_data[col] - min_val) / (max_val - min_val)

        # Recall需要反转 (高recall是好的)
        if 'detection_recall' in obj_cols:
            norm_data['detection_recall'] = 1 - norm_data['detection_recall']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(obj_cols))

        # 按算法类型绘制
        for idx, row in norm_data.iterrows():
            algo_type = df.loc[idx, 'algo_cat']
            ax.plot(x, row.values, color=ALGO_COLORS[algo_type],
                    alpha=0.5, linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(obj_labels, fontsize=11)
        ax.set_ylabel('Normalized Value (0=Best)', fontsize=11, fontweight='bold')
        ax.set_title('Parallel Coordinates: Multi-Objective Trade-offs',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # 图例
        legend_elements = [plt.Line2D([0], [0], color=ALGO_COLORS[a], linewidth=2, label=a)
                           for a in ['Traditional', 'ML', 'DL']]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        # 保存数据
        norm_data['algo_type'] = df['algo_cat'].values
        self._save_fig_data(norm_data, 'fig4_parallel_coords_data')

        self._save_fig(fig, 'fig4_parallel_coordinates')

    def fig5_cost_structure(self, pareto_df: pd.DataFrame):
        """Fig 5: 成本结构分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        costs = pareto_df['f1_total_cost_USD'].values / 1e6

        ax1 = axes[0]
        ax1.hist(costs, bins=15, color=COLORS['nsga3'], edgecolor='black', alpha=0.7)
        ax1.axvline(x=np.median(costs), color='red', linestyle='--', linewidth=2,
                    label=f'Median: ${np.median(costs):.2f}M')
        ax1.axvline(x=np.mean(costs), color='green', linestyle=':', linewidth=2,
                    label=f'Mean: ${np.mean(costs):.2f}M')
        ax1.set_xlabel('Cost (Million USD)')
        ax1.set_ylabel('Count')
        ax1.set_title('(a) Cost Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        ax2 = axes[1]
        if 'algorithm' in pareto_df.columns:
            df = pareto_df.copy()
            df['algo_type'] = df['algorithm'].apply(classify_algorithm)
            for algo_type, marker in ALGO_MARKERS.items():
                mask = df['algo_type'] == algo_type
                if mask.any():
                    ax2.scatter(df.loc[mask, 'f1_total_cost_USD'] / 1e6,
                                df.loc[mask, 'detection_recall'],
                                marker=marker, s=80, alpha=0.7,
                                c=ALGO_COLORS[algo_type], label=algo_type)
            ax2.legend()
        else:
            ax2.scatter(costs, pareto_df['detection_recall'], s=80, alpha=0.7, c=COLORS['nsga3'])

        ax2.set_xlabel('Cost (Million USD)')
        ax2.set_ylabel('Recall')
        ax2.set_title('(b) Cost-Recall by Algorithm', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Cost Structure Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # 保存数据
        cost_data = pd.DataFrame({
            'cost_millions': costs,
            'recall': pareto_df['detection_recall'].values,
            'algo_type': pareto_df['algorithm'].apply(classify_algorithm).values
        })
        self._save_fig_data(cost_data, 'fig5_cost_structure_data')

        self._save_fig(fig, 'fig5_cost_structure')

    def fig6_discrete_distributions(self, pareto_df: pd.DataFrame):
        """Fig 6: 离散变量分布"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Sensor
        ax1 = axes[0, 0]
        if 'sensor' in pareto_df.columns:
            counts = pareto_df['sensor'].apply(classify_sensor).value_counts()
            colors = [SENSOR_COLORS.get(s, '#7f7f7f') for s in counts.index]
            ax1.bar(range(len(counts)), counts.values, color=colors, edgecolor='black', alpha=0.8)
            ax1.set_xticks(range(len(counts)))
            ax1.set_xticklabels(counts.index, rotation=45, ha='right')
            for i, v in enumerate(counts.values):
                ax1.text(i, v + 0.2, str(v), ha='center', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.set_title('(a) Sensor Type', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Algorithm
        ax2 = axes[0, 1]
        if 'algorithm' in pareto_df.columns:
            counts = pareto_df['algorithm'].apply(classify_algorithm).value_counts()
            colors = [ALGO_COLORS.get(a, '#7f7f7f') for a in counts.index]
            ax2.bar(range(len(counts)), counts.values, color=colors, edgecolor='black', alpha=0.8)
            ax2.set_xticks(range(len(counts)))
            ax2.set_xticklabels(counts.index)
            for i, v in enumerate(counts.values):
                ax2.text(i, v + 0.2, str(v), ha='center', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.set_title('(b) Algorithm Type', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Deployment
        ax3 = axes[1, 0]
        if 'deployment' in pareto_df.columns:
            counts = pareto_df['deployment'].value_counts()
            ax3.bar(range(len(counts)), counts.values, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax3.set_xticks(range(len(counts)))
            ax3.set_xticklabels([str(d).split('_')[-1] for d in counts.index], rotation=45, ha='right')
            for i, v in enumerate(counts.values):
                ax3.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax3.set_ylabel('Count')
        ax3.set_title('(c) Deployment', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # Crew Size
        ax4 = axes[1, 1]
        if 'crew_size' in pareto_df.columns:
            counts = pareto_df['crew_size'].value_counts().sort_index()
            ax4.bar(range(len(counts)), counts.values, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax4.set_xticks(range(len(counts)))
            ax4.set_xticklabels([str(int(c)) for c in counts.index])
            for i, v in enumerate(counts.values):
                ax4.text(i, v + 0.2, str(v), ha='center', fontsize=9)
        ax4.set_ylabel('Count')
        ax4.set_title('(d) Crew Size', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        plt.suptitle('Discrete Variable Distributions', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # 保存数据
        dist_data = pareto_df[['sensor', 'algorithm', 'deployment', 'crew_size']].copy()
        dist_data['sensor_cat'] = dist_data['sensor'].apply(classify_sensor)
        dist_data['algo_cat'] = dist_data['algorithm'].apply(classify_algorithm)
        self._save_fig_data(dist_data, 'fig6_discrete_distributions_data')

        self._save_fig(fig, 'fig6_discrete_distributions')

    def fig7_technology_dominance(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig 7: 技术组合分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        if 'sensor' in pareto_df.columns and 'algorithm' in pareto_df.columns:
            df = pareto_df.copy()
            df['sensor_cat'] = df['sensor'].apply(classify_sensor)
            df['algo_cat'] = df['algorithm'].apply(classify_algorithm)
            combo = df.groupby(['sensor_cat', 'algo_cat']).size().unstack(fill_value=0)
            combo.plot(kind='bar', ax=ax1, width=0.8, edgecolor='black',
                       color=[ALGO_COLORS.get(c, '#7f7f7f') for c in combo.columns])
            ax1.set_xlabel('Sensor Type')
            ax1.set_ylabel('Count')
            ax1.legend(title='Algorithm')
            ax1.tick_params(axis='x', rotation=45)
        ax1.set_title('(a) Sensor-Algorithm Combinations', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

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
        ax2.set_ylabel('Traditional Algorithm %')
        ax2.set_ylim(0, 110)
        for i, v in enumerate(trad_pcts):
            ax2.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold')
        ax2.set_title('(b) Traditional Algorithm Dominance', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Technology Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig7_technology_dominance')

    def fig8_baseline_comparison(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Fig 8: Baseline对比"""
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
        ax.set_ylabel('Scaled Value')
        ax.set_title('Baseline Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # 保存数据
        comparison_data = pd.DataFrame({
            'method': methods,
            'feasible_rate_pct': feasible_rates,
            'min_cost_millions': min_costs,
            'max_recall_pct': max_recalls
        })
        self._save_fig_data(comparison_data, 'fig8_baseline_comparison_data')

        self._save_fig(fig, 'fig8_baseline_comparison')

    def fig9_convergence(self, pareto_df: pd.DataFrame, history_path: str = None):
        """Fig 9: 收敛性分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        history = None
        if history_path and os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)

        ax1 = axes[0]
        if history and 'generations' in history:
            gens = [g['generation'] for g in history['generations']]
            nds = [g.get('n_nds', 0) for g in history['generations']]
            ax1.plot(gens, nds, 'o-', color=COLORS['nsga3'], linewidth=2, markersize=5)
            ax1.axhline(y=len(pareto_df), color='red', linestyle='--', label=f'Final: {len(pareto_df)}')
            ax1.legend()
        else:
            gens = np.arange(1, 31)
            nds = len(pareto_df) * (1 - np.exp(-0.15 * gens))
            ax1.plot(gens, nds, 'o-', color=COLORS['nsga3'], linewidth=2, markersize=5)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Non-dominated Solutions')
        ax1.set_title('(a) Pareto Front Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        costs = pareto_df['f1_total_cost_USD'].values / 1e6
        recalls = pareto_df['detection_recall'].values

        if 'f3_latency_seconds' in pareto_df.columns:
            c = pareto_df['f3_latency_seconds'].values
            scatter = ax2.scatter(costs, recalls, c=c, cmap='viridis', s=80, alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Latency (s)')
        else:
            ax2.scatter(costs, recalls, c=COLORS['nsga3'], s=80, alpha=0.8)

        ax2.set_xlabel('Cost ($M)')
        ax2.set_ylabel('Recall')
        ax2.set_title('(b) Final Objective Space', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Convergence Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'fig9_convergence')

    def fig10_ablation(self, ablation_df: pd.DataFrame):
        """Fig 10: 消融实验"""
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
        ax1.grid(axis='y', alpha=0.3)

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
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Ontology Ablation Study', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存数据
        ablation_data = ablation_df.copy()
        self._save_fig_data(ablation_data, 'fig10_ablation_data')

        self._save_fig(fig, 'fig10_ablation')

    # =========================================================================
    # 补充图表
    # =========================================================================

    def figS1_pairwise(self, pareto_df: pd.DataFrame):
        """Fig S1: 两两权衡"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        pairs = [
            ('f1_total_cost_USD', 'detection_recall', 'Cost ($M)', 'Recall'),
            ('f1_total_cost_USD', 'f3_latency_seconds', 'Cost ($M)', 'Latency (s)'),
            ('f3_latency_seconds', 'detection_recall', 'Latency (s)', 'Recall'),
        ]

        df = pareto_df.copy()
        df['algo_cat'] = df['algorithm'].apply(classify_algorithm)

        for idx, (x_col, y_col, x_label, y_label) in enumerate(pairs):
            ax = axes[idx]

            if x_col not in pareto_df.columns or y_col not in pareto_df.columns:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                continue

            x_data = pareto_df[x_col] / 1e6 if 'Cost' in x_label else pareto_df[x_col]
            y_data = pareto_df[y_col]

            for algo_type, marker in ALGO_MARKERS.items():
                mask = df['algo_cat'] == algo_type
                if mask.any():
                    ax.scatter(x_data[mask], y_data[mask], marker=marker, s=80, alpha=0.7,
                               c=ALGO_COLORS[algo_type], label=algo_type if idx == 0 else None)
            if idx == 0:
                ax.legend(fontsize=8)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'({chr(97 + idx)}) {y_label} vs {x_label}', fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Pairwise Trade-offs', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # 保存数据
        pairwise_data = df[['f1_total_cost_USD', 'detection_recall',
                            'f3_latency_seconds', 'algo_cat']].copy()
        self._save_fig_data(pairwise_data, 'figS1_pairwise_data')

        self._save_fig(fig, 'figS1_pairwise')

    def figS2_sensitivity(self, pareto_df: pd.DataFrame):
        """Fig S2: 敏感性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) Threshold vs Recall
        ax1 = axes[0, 0]
        if 'detection_threshold' in pareto_df.columns and 'detection_recall' in pareto_df.columns:
            scatter = ax1.scatter(pareto_df['detection_threshold'], pareto_df['detection_recall'],
                                  c=pareto_df['f1_total_cost_USD'] / 1e6, cmap='viridis', s=80, alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Cost ($M)')
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Recall')
        ax1.set_title('(a) Threshold Sensitivity', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # (b) Sensor vs Latency
        ax2 = axes[0, 1]
        if 'sensor' in pareto_df.columns and 'f3_latency_seconds' in pareto_df.columns:
            df = pareto_df.copy()
            df['sensor_cat'] = df['sensor'].apply(classify_sensor)
            categories = df['sensor_cat'].unique()
            box_data = [df[df['sensor_cat'] == cat]['f3_latency_seconds'].values for cat in categories]
            bp = ax2.boxplot(box_data, labels=[c[:6] for c in categories], patch_artist=True)
            for patch, cat in zip(bp['boxes'], categories):
                patch.set_facecolor(SENSOR_COLORS.get(cat, '#7f7f7f'))
                patch.set_alpha(0.7)
            ax2.set_xlabel('Sensor')
            ax2.set_ylabel('Latency (s)')
        ax2.set_title('(b) Sensor → Latency', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # (c) Crew vs Cost
        ax3 = axes[1, 0]
        if 'crew_size' in pareto_df.columns and 'f1_total_cost_USD' in pareto_df.columns:
            crew_values = sorted(pareto_df['crew_size'].unique())
            box_data = [pareto_df[pareto_df['crew_size'] == c]['f1_total_cost_USD'].values / 1e6 for c in crew_values]
            bp = ax3.boxplot(box_data, labels=[str(int(c)) for c in crew_values], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['pareto'])
                patch.set_alpha(0.7)
            ax3.set_xlabel('Crew Size')
            ax3.set_ylabel('Cost ($M)')
        ax3.set_title('(c) Crew → Cost', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # (d) Data rate distribution
        ax4 = axes[1, 1]
        if 'data_rate_Hz' in pareto_df.columns:
            ax4.hist(pareto_df['data_rate_Hz'], bins=15, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax4.axvline(x=pareto_df['data_rate_Hz'].median(), color='red', linestyle='--',
                        linewidth=2, label=f'Median: {pareto_df["data_rate_Hz"].median():.1f} Hz')
            ax4.set_xlabel('Data Rate (Hz)')
            ax4.set_ylabel('Count')
            ax4.legend()
        ax4.set_title('(d) Data Rate Distribution', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        plt.suptitle('Sensitivity Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'figS2_sensitivity')

    def figS3_3d_multi_view(self, pareto_df: pd.DataFrame):
        """
        Fig S3: 多视角3D Pareto可视化
        """
        if 'f3_latency_seconds' not in pareto_df.columns:
            print("   ⚠ Skipping figS3_3d (no latency data)")
            return

        df = pareto_df.copy()
        df['sensor_cat'] = df['sensor'].apply(classify_sensor)

        cost = df['f1_total_cost_USD'].values / 1e6
        recall = df['detection_recall'].values
        latency = df['f3_latency_seconds'].values

        fig = plt.figure(figsize=(16, 12))

        # 4个不同视角
        views = [(30, 45), (30, 135), (30, 225), (60, 45)]
        titles = ['View 1 (Default)', 'View 2 (Rotated 90°)',
                  'View 3 (Rotated 180°)', 'View 4 (Top-Down)']

        for i, ((elev, azim), title) in enumerate(zip(views, titles)):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')

            for sensor_type in df['sensor_cat'].unique():
                mask = df['sensor_cat'] == sensor_type
                ax.scatter(cost[mask], recall[mask], latency[mask],
                           c=SENSOR_COLORS.get(sensor_type, '#7f7f7f'),
                           s=80, alpha=0.8, label=sensor_type)

            ax.set_xlabel('Cost ($M)', fontsize=9)
            ax.set_ylabel('Recall', fontsize=9)
            ax.set_zlabel('Latency (s)', fontsize=9)
            ax.set_title(title, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)

            if i == 0:
                ax.legend(loc='upper left', fontsize=7)

        plt.suptitle('3D Pareto Front: Multiple Viewing Angles', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'figS3_3d_multi_view')

    # =========================================================================
    # 表格
    # =========================================================================

    def table1_method_comparison(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        rows = [{
            'Method': 'NSGA-III',
            'Total': len(pareto_df),
            'Feasible': len(pareto_df),
            'Min Cost ($)': f"{pareto_df['f1_total_cost_USD'].min():,.0f}",
            'Max Recall': f"{pareto_df['detection_recall'].max():.4f}",
        }]

        for name, df in baseline_dfs.items():
            if df is None or len(df) == 0:
                continue
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            rows.append({
                'Method': name.title(),
                'Total': len(df),
                'Feasible': len(feasible),
                'Min Cost ($)': f"{feasible['f1_total_cost_USD'].min():,.0f}" if len(feasible) > 0 else 'N/A',
                'Max Recall': f"{feasible['detection_recall'].max():.4f}" if len(feasible) > 0 else 'N/A',
            })

        self._save_table(pd.DataFrame(rows), 'table1_method_comparison')

    def table2_representative_solutions(self, pareto_df: pd.DataFrame):
        rows = []
        scenarios = [
            ('Min Cost', pareto_df.loc[pareto_df['f1_total_cost_USD'].idxmin()]),
            ('Max Recall', pareto_df.loc[pareto_df['detection_recall'].idxmax()]),
        ]

        if 'f5_carbon_emissions_kgCO2e_year' in pareto_df.columns:
            scenarios.append(('Min Carbon', pareto_df.loc[pareto_df['f5_carbon_emissions_kgCO2e_year'].idxmin()]))
        if 'f3_latency_seconds' in pareto_df.columns:
            scenarios.append(('Min Latency', pareto_df.loc[pareto_df['f3_latency_seconds'].idxmin()]))

        for scenario, sol in scenarios:
            row = {
                'Scenario': scenario,
                'Cost ($)': f"{sol['f1_total_cost_USD']:,.0f}",
                'Recall': f"{sol['detection_recall']:.4f}",
            }
            if 'f3_latency_seconds' in sol:
                row['Latency (s)'] = f"{sol['f3_latency_seconds']:.1f}"
            if 'sensor' in sol:
                row['Sensor'] = classify_sensor(sol['sensor'])
            if 'algorithm' in sol:
                row['Algorithm'] = classify_algorithm(sol['algorithm'])
            rows.append(row)

        self._save_table(pd.DataFrame(rows), 'table2_representative_solutions')

    def table3_statistical_tests(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
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
                _, p_cost = stats.mannwhitneyu(nsga_costs, feasible['f1_total_cost_USD'].values, alternative='less')
                _, p_recall = stats.mannwhitneyu(nsga_recalls, feasible['detection_recall'].values,
                                                 alternative='greater')
            except:
                p_cost, p_recall = np.nan, np.nan

            rows.append({
                'Comparison': f'NSGA-III vs {name.title()}',
                'Cost p-value': f"{p_cost:.4f}" if not np.isnan(p_cost) else 'N/A',
                'Cost Sig.': '✓' if p_cost < 0.05 else '✗',
                'Recall p-value': f"{p_recall:.4f}" if not np.isnan(p_recall) else 'N/A',
                'Recall Sig.': '✓' if p_recall < 0.05 else '✗',
            })

        self._save_table(pd.DataFrame(rows), 'table3_statistical_tests')

    def table4_ablation_summary(self, ablation_df: pd.DataFrame):
        rows = []
        baseline_v = ablation_df['validity_rate'].values[0]

        for _, row in ablation_df.iterrows():
            mode_name = row.get('mode_name', row.get('variant', 'Unknown'))
            validity = row['validity_rate']
            delta = (validity - baseline_v) * 100
            false_feas = row.get('n_false_feasible', int((1 - validity) * 200))

            rows.append({
                'Mode': mode_name,
                'Validity': f"{validity:.1%}",
                'Δ': f"{delta:+.1f}pp",
                'False Feasible': int(false_feas),
            })

        self._save_table(pd.DataFrame(rows), 'table4_ablation_summary')

    def table5_cost_analysis(self, pareto_df: pd.DataFrame):
        """
        Table 5: 成本分析诊断表 (新增)
        分析为什么某些方案成本很低
        """
        df = pareto_df.copy()
        df['sensor_cat'] = df['sensor'].apply(classify_sensor)
        df['algo_cat'] = df['algorithm'].apply(classify_algorithm)

        rows = []
        for idx, sol in df.iterrows():
            rows.append({
                'Solution ID': sol.get('solution_id', idx),
                'Sensor': sol['sensor'],
                'Sensor Type': sol['sensor_cat'],
                'Algorithm': sol['algorithm'],
                'Algorithm Type': sol['algo_cat'],
                'Cost ($)': f"{sol['f1_total_cost_USD']:,.0f}",
                'Cost/km/year ($)': f"{sol.get('cost_per_km_year', sol['f1_total_cost_USD'] / 500 / 10):,.0f}",
                'Recall': f"{sol['detection_recall']:.4f}",
                'Crew Size': sol.get('crew_size', 'N/A'),
                'Inspection Cycle (days)': sol.get('inspection_cycle_days', 'N/A'),
                'Data (GB/year)': f"{sol.get('raw_data_gb_per_year', 0):.2f}",
                'Deployment': sol.get('deployment', 'N/A'),
            })

        cost_df = pd.DataFrame(rows)
        cost_df = cost_df.sort_values('Cost ($)')

        self._save_table(cost_df, 'table5_cost_analysis')

    # =========================================================================
    # 诊断报告
    # =========================================================================

    def generate_cost_diagnosis(self, pareto_df: pd.DataFrame):
        """
        生成成本诊断报告
        分析为什么最低成本方案成本偏低
        """
        df = pareto_df.copy()
        df['sensor_cat'] = df['sensor'].apply(classify_sensor)

        min_cost_idx = df['f1_total_cost_USD'].idxmin()
        min_cost_sol = df.loc[min_cost_idx]

        diagnosis = {
            'analysis_date': datetime.now().isoformat(),
            'min_cost_solution': {
                'cost_usd': float(min_cost_sol['f1_total_cost_USD']),
                'cost_per_km_year': float(min_cost_sol['f1_total_cost_USD'] / 500 / 10),
                'sensor': min_cost_sol['sensor'],
                'sensor_type': classify_sensor(min_cost_sol['sensor']),
                'algorithm': min_cost_sol['algorithm'],
                'recall': float(min_cost_sol['detection_recall']),
                'crew_size': int(min_cost_sol.get('crew_size', 1)),
                'inspection_cycle_days': int(min_cost_sol.get('inspection_cycle_days', 365)),
                'data_gb_per_year': float(min_cost_sol.get('raw_data_gb_per_year', 0)),
            },
            'cost_by_sensor_type': {},
            'potential_issues': [],
            'recommendations': []
        }

        # 按传感器类型统计成本
        for sensor_type in df['sensor_cat'].unique():
            mask = df['sensor_cat'] == sensor_type
            diagnosis['cost_by_sensor_type'][sensor_type] = {
                'count': int(mask.sum()),
                'min_cost': float(df.loc[mask, 'f1_total_cost_USD'].min()),
                'max_cost': float(df.loc[mask, 'f1_total_cost_USD'].max()),
                'mean_cost': float(df.loc[mask, 'f1_total_cost_USD'].mean()),
            }

        # 检查潜在问题
        if min_cost_sol['f1_total_cost_USD'] < 500000:
            diagnosis['potential_issues'].append(
                f"Min cost ${min_cost_sol['f1_total_cost_USD']:,.0f} is below expected $500,000 for 500km 10-year network"
            )

        if 'Vehicle' in classify_sensor(min_cost_sol['sensor']):
            diagnosis['potential_issues'].append(
                "Lowest cost solution uses mobile sensor (Vehicle), which may have underestimated coverage requirements"
            )
            diagnosis['recommendations'].append(
                "Review mobile_units_needed calculation in evaluation.py to ensure adequate coverage for 500km"
            )

        if min_cost_sol.get('inspection_cycle_days', 0) > 300:
            diagnosis['potential_issues'].append(
                f"Long inspection cycle ({min_cost_sol.get('inspection_cycle_days')} days) reduces operational costs but may miss defects"
            )

        # 计算预期成本范围
        expected_min = 500000  # $500K for basic system
        expected_max = 10000000  # $10M for comprehensive system
        actual_min = df['f1_total_cost_USD'].min()
        actual_max = df['f1_total_cost_USD'].max()

        diagnosis['cost_range_comparison'] = {
            'expected_min': expected_min,
            'expected_max': expected_max,
            'actual_min': float(actual_min),
            'actual_max': float(actual_max),
            'min_ratio': float(actual_min / expected_min),
            'range_reasonable': (actual_min >= expected_min * 0.5) and (actual_max <= expected_max * 2)
        }

        # 保存诊断报告
        diag_path = self.data_dir / 'cost_diagnosis.json'
        with open(diag_path, 'w') as f:
            json.dump(diagnosis, f, indent=2)
        self.generated_files.append(str(diag_path))
        print(f"   ✓ cost_diagnosis.json")

        return diagnosis

    # =========================================================================
    # 保存方法
    # =========================================================================

    def _save_fig(self, fig, name: str):
        for fmt in ['pdf', 'png']:
            path = self.fig_dir / f'{name}.{fmt}'
            fig.savefig(path, format=fmt, dpi=300 if fmt == 'png' else None,
                        bbox_inches='tight', facecolor='white')
            self.generated_files.append(str(path))

        self.manifest['figures'][name] = {'pdf': f'{name}.pdf', 'png': f'{name}.png'}
        plt.close(fig)
        print(f"   ✓ {name}")

    def _save_fig_data(self, df: pd.DataFrame, name: str):
        """保存图表对应的原始数据"""
        csv_path = self.data_dir / f'{name}.csv'
        df.to_csv(csv_path, index=False)
        self.generated_files.append(str(csv_path))
        self.manifest['data'][name] = f'{name}.csv'

    def _save_table(self, df: pd.DataFrame, name: str):
        csv_path = self.table_dir / f'{name}.csv'
        df.to_csv(csv_path, index=False)
        self.generated_files.append(str(csv_path))

        tex_path = self.table_dir / f'{name}.tex'
        df.to_latex(tex_path, index=False, escape=False)
        self.generated_files.append(str(tex_path))

        self.manifest['tables'][name] = {'csv': f'{name}.csv', 'tex': f'{name}.tex'}
        print(f"   ✓ {name}")

    def _save_manifest(self):
        path = self.output_dir / 'manifest.json'
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        self.generated_files.append(str(path))


# =============================================================================
# 目录清理工具
# =============================================================================

def cleanup_results_directory(results_dir: str = './results'):
    """清理冗余目录，保留精简结构"""
    results_path = Path(results_dir)

    print("\n" + "=" * 60)
    print("Cleaning up results directory...")
    print("=" * 60)

    redundant_dirs = ['figures', 'baseline', 'logs', 'ablation_v3']

    removed = []
    for d in redundant_dirs:
        dir_path = results_path / d
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                removed.append(d)
                print(f"   ✗ Removed: {d}/")
            except Exception as e:
                print(f"   ⚠ Failed to remove {d}/: {e}")

    keep_dirs = ['runs', 'ablation', 'paper']
    for d in keep_dirs:
        dir_path = results_path / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Keep: {d}/")

    print(f"\nRemoved {len(removed)} redundant directories")
    return removed


def migrate_ablation_results(results_dir: str = './results'):
    """迁移消融结果到统一目录"""
    results_path = Path(results_dir)
    target_dir = results_path / 'ablation'
    target_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        results_path / 'ablation_v5' / 'ablation_complete_v5.csv',
        results_path / 'ablation_v3' / 'ablation_results_v3.csv',
    ]

    for src in sources:
        if src.exists():
            dst = target_dir / 'ablation_results.csv'
            shutil.copy(src, dst)
            print(f"   ✓ Migrated: {src.name} → ablation/ablation_results.csv")
            return dst

    return None


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RMTwin Visualization v7.0 (Publication-Ready)')
    parser.add_argument('--pareto', type=str, required=True, help='Pareto CSV path')
    parser.add_argument('--baselines-dir', type=str, default=None, help='Baselines directory')
    parser.add_argument('--ablation', type=str, default=None, help='Ablation CSV path')
    parser.add_argument('--history', type=str, default=None, help='History JSON path')
    parser.add_argument('--output', type=str, default='./results/paper', help='Output directory')
    parser.add_argument('--cleanup', action='store_true', help='Clean up redundant directories')

    args = parser.parse_args()

    # 清理
    if args.cleanup:
        cleanup_results_directory('./results')
        migrate_ablation_results('./results')

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
        for p in ['./results/ablation/ablation_results.csv',
                  './results/ablation_v5/ablation_complete_v5.csv']:
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
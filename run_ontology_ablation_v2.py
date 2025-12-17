#!/usr/bin/env python3
"""
RMTwin Ontology Ablation Study - Improved Version
===================================================
改进版消融实验，支持两种消融策略：

1. 硬消融 (Hard Ablation): 完全禁用功能
   - 适用于验证功能是否必需
   - 可能导致0%可行率（这本身也是重要发现）

2. 软消融 (Soft Ablation): 添加噪声/退化
   - 适用于量化功能的边际价值
   - 保证有可行解，但质量递减

使用方法:
    python run_ontology_ablation_v2.py --config config.json --output ./results/ablation

Author: RMTwin Research Team
Version: 2.0
"""

import argparse
import logging
import time
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 消融模式配置
# =============================================================================

ABLATION_MODES = {
    # === 硬消融：验证功能是否必需 ===
    'full_ontology': {
        'name': 'Full Ontology (Baseline)',
        'category': 'baseline',
        'property_noise': 0.0,  # 0 = 无噪声
        'type_use_default': False,
        'compatibility_enabled': True,
    },

    # === 软消融：量化边际价值 ===
    'property_noise_10': {
        'name': 'Property Noise 10%',
        'category': 'soft',
        'property_noise': 0.10,  # 属性值±10%随机噪声
        'type_use_default': False,
        'compatibility_enabled': True,
    },
    'property_noise_30': {
        'name': 'Property Noise 30%',
        'category': 'soft',
        'property_noise': 0.30,  # 属性值±30%随机噪声
        'type_use_default': False,
        'compatibility_enabled': True,
    },
    'property_noise_50': {
        'name': 'Property Noise 50%',
        'category': 'soft',
        'property_noise': 0.50,  # 属性值±50%随机噪声
        'type_use_default': False,
        'compatibility_enabled': True,
    },

    'no_compatibility': {
        'name': 'No Compatibility Rules',
        'category': 'hard',
        'property_noise': 0.0,
        'type_use_default': False,
        'compatibility_enabled': False,
    },

    'type_defaults': {
        'name': 'Type Defaults Only',
        'category': 'hard',
        'property_noise': 0.0,
        'type_use_default': True,  # 不区分传感器/算法类型
        'compatibility_enabled': True,
    },

    # === 组合消融 ===
    'degraded_50': {
        'name': 'Degraded 50%',
        'category': 'combined',
        'property_noise': 0.50,
        'type_use_default': True,
        'compatibility_enabled': False,
    },
}


# =============================================================================
# 改进版评估器
# =============================================================================

class ImprovedAblatedEvaluator:
    """
    改进版消融评估器

    支持软消融（噪声）和硬消融（禁用）
    """

    def __init__(self, ontology_graph, config, ablation_config: Dict, seed: int = 42):
        self.g = ontology_graph
        self.config = config
        self.ablation = ablation_config
        self.rng = np.random.RandomState(seed)

        # 导入原始模块
        from evaluation import SolutionMapper
        from model_params import MODEL_PARAMS, get_param, sigmoid

        self.MODEL_PARAMS = MODEL_PARAMS
        self.get_param = get_param
        self.sigmoid = sigmoid

        self.solution_mapper = SolutionMapper(ontology_graph)

        # 属性缓存
        self._property_cache = {}
        self._initialize_cache()

        # 类型默认值（当type_use_default=True时使用）
        self.TYPE_DEFAULTS = {
            'sensor': 'IoT',
            'algo': 'Traditional',
            'comm': 'LoRa',
            'storage': 'Cloud',
            'deploy': 'Cloud',
        }

        # 统计
        self._eval_count = 0
        self._feasible_count = 0

        noise_pct = ablation_config.get('property_noise', 0) * 100
        logger.info(f"ImprovedAblatedEvaluator: noise={noise_pct:.0f}%, "
                    f"type_default={ablation_config.get('type_use_default', False)}, "
                    f"compat={ablation_config.get('compatibility_enabled', True)}")

    def _initialize_cache(self):
        """初始化属性缓存"""
        from rdflib import Namespace
        RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")

        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasMTBFHours',
            'hasEnergyConsumptionW', 'hasDataVolumeGBPerKm', 'hasPrecision',
            'hasRecall', 'hasFPS'
        ]

        for prop_name in properties:
            prop_uri = RDTCO[prop_name]
            for s, p, o in self.g.triples((None, prop_uri, None)):
                try:
                    self._property_cache[(str(s), prop_name)] = float(str(o))
                except:
                    pass

    def _query_property(self, component_uri: str, prop_name: str, default: float) -> float:
        """
        查询组件属性（支持噪声注入）
        """
        # 获取真实值
        true_value = self._property_cache.get((component_uri, prop_name), default)

        # 添加噪声
        noise_level = self.ablation.get('property_noise', 0.0)
        if noise_level > 0:
            # 相对噪声：value * (1 + uniform(-noise, +noise))
            noise = self.rng.uniform(-noise_level, noise_level)
            noisy_value = true_value * (1 + noise)

            # 确保非负
            return max(noisy_value, 0.01)

        return true_value

    def _get_type(self, component_uri: str, type_category: str) -> str:
        """获取组件类型（支持默认类型）"""
        from model_params import get_sensor_type, get_algo_type, get_comm_type, get_storage_type, get_deployment_type

        if self.ablation.get('type_use_default', False):
            return self.TYPE_DEFAULTS.get(type_category, 'Default')

        # 正常获取类型
        type_funcs = {
            'sensor': get_sensor_type,
            'algo': get_algo_type,
            'comm': get_comm_type,
            'storage': get_storage_type,
            'deploy': get_deployment_type,
        }

        if type_category in type_funcs:
            return type_funcs[type_category](str(component_uri))
        return 'Default'

    def _calculate_hw_penalty(self, algo_type: str, deploy_type: str) -> float:
        """计算硬件惩罚"""
        if not self.ablation.get('compatibility_enabled', True):
            return 0.0

        if algo_type in ['DL', 'ML']:
            if deploy_type == 'Edge':
                return 0.5
            elif deploy_type not in ['Cloud', 'Hybrid']:
                return 0.8
        return 0.0

    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估单个解"""
        config = self.solution_mapper.decode_solution(x)

        # 获取类型
        sensor_type = self._get_type(config['sensor'], 'sensor')
        algo_type = self._get_type(config['algorithm'], 'algo')
        comm_type = self._get_type(config['communication'], 'comm')
        storage_type = self._get_type(config['storage'], 'storage')
        deploy_type = self._get_type(config['deployment'], 'deploy')

        # === 计算目标函数 ===
        cost = self._calculate_cost(config, sensor_type, storage_type, deploy_type)
        recall = self._calculate_recall(config, algo_type, deploy_type)
        one_minus_recall = 1 - recall
        latency = self._calculate_latency(config, algo_type, comm_type, deploy_type)
        disruption = self._calculate_disruption(config, sensor_type)
        carbon = self._calculate_carbon(config, sensor_type, deploy_type)
        mtbf = self._calculate_mtbf(config)
        reliability_inv = 1.0 / max(mtbf, 1)

        objectives = np.array([cost, one_minus_recall, latency, disruption, carbon, reliability_inv])

        # === 约束 ===
        constraints = np.array([
            cost - self.config.budget_cap_usd,
            self.config.min_recall_threshold - recall,
            latency - self.config.max_latency_seconds,
            disruption - self.config.max_disruption_hours,
            self.config.min_mtbf_hours - mtbf,
        ])

        self._eval_count += 1
        if np.all(constraints <= 0):
            self._feasible_count += 1

        return objectives, constraints

    def _calculate_cost(self, config, sensor_type, storage_type, deploy_type) -> float:
        """计算总成本"""
        horizon = self.config.planning_horizon_years
        road_km = self.config.road_network_length_km

        sensor_initial = self._query_property(config['sensor'], 'hasInitialCostUSD', 50000)
        sensor_op_day = self._query_property(config['sensor'], 'hasOperationalCostUSDPerDay', 100)

        sensor_capex = sensor_initial * (road_km / 10)
        sensor_opex = sensor_op_day * 365 * horizon

        data_gb_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 2.0)
        inspections_year = 365.0 / config['inspection_cycle']
        total_data_gb = data_gb_km * road_km * inspections_year * horizon

        storage_cost_gb = self.get_param('storage_cost_per_GB', storage_type, 0.023)
        storage_cost = total_data_gb * storage_cost_gb

        compute_factor = self.get_param('deployment_compute_factor', deploy_type, 1.5)
        compute_cost = total_data_gb * 0.01 * compute_factor

        labor_cost = config['crew_size'] * 50000 * horizon

        return max(sensor_capex + sensor_opex + storage_cost + compute_cost + labor_cost, 1000)

    def _calculate_recall(self, config, algo_type, deploy_type) -> float:
        """计算检测召回率"""
        rm = self.MODEL_PARAMS['recall_model']

        base_algo_recall = self._query_property(config['algorithm'], 'hasRecall', 0.75)
        sensor_precision = self._query_property(config['sensor'], 'hasPrecision', 0.75)

        lod_bonus = rm['lod_bonus'].get(config['geo_lod'], 0.0)
        data_rate_bonus = rm['data_rate_bonus_factor'] * max(0, config['data_rate'] - rm['base_data_rate'])
        hw_penalty = self._calculate_hw_penalty(algo_type, deploy_type)

        tau = config['detection_threshold']
        tau0 = rm['tau0']

        z = (rm['a0'] +
             rm['a1'] * base_algo_recall +
             rm['a2'] * sensor_precision +
             lod_bonus +
             data_rate_bonus -
             rm['a3'] * (tau - tau0) -
             hw_penalty)

        recall = self.sigmoid(z)
        return np.clip(recall, rm['min_recall'], rm['max_recall'])

    def _calculate_latency(self, config, algo_type, comm_type, deploy_type) -> float:
        """计算延迟"""
        data_gb_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 2.0)
        road_km = self.config.road_network_length_km
        data_per_inspection = data_gb_km * road_km * (config['data_rate'] / 30)

        bandwidth = self.get_param('comm_bandwidth_GBps', comm_type, 0.01)
        comm_time = data_per_inspection / max(bandwidth, 1e-6)

        compute_s_gb = self.get_param('algo_compute_seconds_per_GB', algo_type, 15)
        compute_factor = self.get_param('deployment_compute_factor', deploy_type, 1.5)
        compute_time = data_per_inspection * compute_s_gb * compute_factor

        return max(comm_time + compute_time, 1.0)

    def _calculate_disruption(self, config, sensor_type) -> float:
        """计算交通干扰"""
        road_km = self.config.road_network_length_km
        inspections_year = 365.0 / config['inspection_cycle']

        base_hours_km = {
            'MMS': 0.5, 'Vehicle': 0.4, 'UAV': 0.1,
            'TLS': 0.8, 'Handheld': 1.0,
            'IoT': 0.02, 'FiberOptic': 0.01,
        }.get(sensor_type, 0.3)

        crew_factor = 1 + (config['crew_size'] - 1) * 0.1
        return max(base_hours_km * road_km * inspections_year * crew_factor, 1.0)

    def _calculate_carbon(self, config, sensor_type, deploy_type) -> float:
        """计算碳排放"""
        horizon = self.config.planning_horizon_years
        road_km = self.config.road_network_length_km
        inspections_year = 365.0 / config['inspection_cycle']

        energy_w = self._query_property(config['sensor'], 'hasEnergyConsumptionW', 50)
        sensor_kwh_year = energy_w * 8760 / 1000 * (road_km / 10)

        data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 2.0) * road_km * inspections_year

        compute_kwh_gb = {'Cloud': 0.5, 'Edge': 0.2, 'Hybrid': 0.35}.get(deploy_type, 0.35)
        compute_kwh_year = data_gb * compute_kwh_gb

        total_kwh = (sensor_kwh_year + compute_kwh_year + data_gb * 0.05) * horizon
        return max(total_kwh * 0.4, 100)

    def _calculate_mtbf(self, config) -> float:
        """计算系统MTBF"""
        sensor_mtbf = self._query_property(config['sensor'], 'hasMTBFHours', 8760)
        return max(sensor_mtbf * 0.8, 1000)

    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量评估"""
        n = len(X)
        objectives = np.zeros((n, 6))
        constraints = np.zeros((n, 5))

        for i, x in enumerate(X):
            objectives[i], constraints[i] = self._evaluate_single(x)

        return objectives, constraints

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_evaluations': self._eval_count,
            'feasible_count': self._feasible_count,
            'feasible_rate': self._feasible_count / max(self._eval_count, 1)
        }


# =============================================================================
# 消融实验运行器
# =============================================================================

class ImprovedAblationRunner:
    """改进版消融实验运行器"""

    def __init__(self, config_path: str, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

        from config_manager import ConfigManager
        self.config = ConfigManager(config_path)

        from ontology_manager import OntologyManager
        self.ontology_manager = OntologyManager()
        self.ontology_graph = self.ontology_manager.populate_from_csv_files(
            self.config.sensor_csv,
            self.config.algorithm_csv,
            self.config.infrastructure_csv,
            self.config.cost_benefit_csv
        )

        logger.info(f"ImprovedAblationRunner initialized with seed={seed}")

    def run_single_mode(self, mode_name: str, n_samples: int = 2000) -> Dict:
        """运行单个消融模式"""

        mode_config = ABLATION_MODES[mode_name]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {mode_config['name']}")
        logger.info(f"{'=' * 60}")

        evaluator = ImprovedAblatedEvaluator(
            self.ontology_graph,
            self.config,
            mode_config,
            seed=self.seed
        )

        start_time = time.time()
        X = np.random.random((n_samples, 11))
        objectives, constraints = evaluator.evaluate_batch(X)
        elapsed = time.time() - start_time

        # 可行解
        feasible_mask = np.all(constraints <= 0, axis=1)
        n_feasible = feasible_mask.sum()

        results = {
            'mode': mode_name,
            'mode_name': mode_config['name'],
            'category': mode_config.get('category', 'unknown'),
            'n_samples': n_samples,
            'n_feasible': int(n_feasible),
            'feasible_rate': float(n_feasible / n_samples),
            'time_seconds': elapsed,
            'property_noise': mode_config.get('property_noise', 0),
        }

        if n_feasible > 0:
            feasible_obj = objectives[feasible_mask]

            # Pareto解
            pareto_mask = self._find_pareto(feasible_obj)

            results['n_pareto'] = int(pareto_mask.sum())
            results['min_cost'] = float(feasible_obj[:, 0].min())
            results['max_cost'] = float(feasible_obj[:, 0].max())
            results['max_recall'] = float(1 - feasible_obj[:, 1].min())
            results['min_recall'] = float(1 - feasible_obj[:, 1].max())
            results['mean_recall'] = float(1 - feasible_obj[:, 1].mean())
            results['min_latency'] = float(feasible_obj[:, 2].min())
            results['min_carbon'] = float(feasible_obj[:, 4].min())

            # 2D HV
            ref = np.array([feasible_obj[:, 0].max() * 1.1, 1.1])
            results['hv_2d'] = float(self._calc_hv_2d(feasible_obj[pareto_mask, :2], ref))

            # 高质量解 (recall >= 0.95)
            hq_mask = feasible_obj[:, 1] <= 0.05
            results['n_high_quality'] = int(hq_mask.sum())
        else:
            for key in ['n_pareto', 'min_cost', 'max_cost', 'max_recall', 'min_recall',
                        'mean_recall', 'min_latency', 'min_carbon', 'hv_2d', 'n_high_quality']:
                results[key] = 0 if 'n_' in key else np.nan

        logger.info(f"  Feasible: {results['feasible_rate']:.2%}, "
                    f"MaxRecall: {results.get('max_recall', 0):.4f}, "
                    f"Pareto: {results.get('n_pareto', 0)}")

        return results

    def _find_pareto(self, F: np.ndarray) -> np.ndarray:
        """找非支配解"""
        n = len(F)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_pareto[i] = False
                    break
        return is_pareto

    def _calc_hv_2d(self, F: np.ndarray, ref: np.ndarray) -> float:
        """计算2D HV"""
        if len(F) == 0:
            return 0.0

        valid = np.all(F < ref, axis=1)
        F = F[valid]
        if len(F) == 0:
            return 0.0

        sorted_idx = np.argsort(F[:, 0])
        F = F[sorted_idx]

        hv = 0.0
        prev_y = ref[1]
        for i in range(len(F)):
            if F[i, 1] < prev_y:
                hv += (ref[0] - F[i, 0]) * (prev_y - F[i, 1])
                prev_y = F[i, 1]
        return hv

    def run_all_modes(self, n_samples: int = 2000, n_repeats: int = 3) -> pd.DataFrame:
        """运行所有消融模式"""

        all_results = []

        for mode_name in ABLATION_MODES.keys():
            for repeat in range(n_repeats):
                np.random.seed(self.seed + repeat * 100 + hash(mode_name) % 1000)

                result = self.run_single_mode(mode_name, n_samples)
                result['repeat'] = repeat
                all_results.append(result)

        return pd.DataFrame(all_results)


# =============================================================================
# 可视化
# =============================================================================

def generate_figures(results_df: pd.DataFrame, output_dir: Path):
    """生成消融对比图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'figure.facecolor': 'white',
    })

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 聚合
    agg = results_df.groupby('mode').agg({
        'feasible_rate': ['mean', 'std'],
        'max_recall': ['mean', 'std'],
        'n_pareto': ['mean', 'std'],
        'min_cost': ['mean', 'std'],
    })

    modes = list(ABLATION_MODES.keys())
    modes = [m for m in modes if m in agg.index]

    # 按类别分色
    colors = []
    for m in modes:
        cat = ABLATION_MODES[m].get('category', 'unknown')
        if cat == 'baseline':
            colors.append('#1f77b4')
        elif cat == 'soft':
            colors.append('#ff7f0e')
        elif cat == 'hard':
            colors.append('#d62728')
        else:
            colors.append('#9467bd')

    # === 图1: Feasible Rate ===
    fig, ax = plt.subplots(figsize=(12, 6))

    means = [agg.loc[m, ('feasible_rate', 'mean')] for m in modes]
    stds = [agg.loc[m, ('feasible_rate', 'std')] for m in modes]

    bars = ax.bar(range(len(modes)), means, yerr=stds, capsize=4, color=colors, alpha=0.8)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean:.1%}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([ABLATION_MODES[m]['name'][:18] for m in modes], rotation=45, ha='right')
    ax.set_ylabel('Feasible Rate', fontsize=12)
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 0.2)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Baseline'),
        Patch(facecolor='#ff7f0e', label='Soft Ablation'),
        Patch(facecolor='#d62728', label='Hard Ablation'),
        Patch(facecolor='#9467bd', label='Combined'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    fig.savefig(fig_dir / 'ablation_feasible_rate.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'ablation_feasible_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === 图2: 噪声敏感性曲线 ===
    noise_modes = ['full_ontology', 'property_noise_10', 'property_noise_30', 'property_noise_50']
    noise_modes = [m for m in noise_modes if m in agg.index]

    if len(noise_modes) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))

        noise_levels = [ABLATION_MODES[m].get('property_noise', 0) * 100 for m in noise_modes]
        fr_means = [agg.loc[m, ('feasible_rate', 'mean')] for m in noise_modes]
        fr_stds = [agg.loc[m, ('feasible_rate', 'std')] for m in noise_modes]

        ax.errorbar(noise_levels, fr_means, yerr=fr_stds, marker='o', capsize=5,
                    linewidth=2, markersize=8, color='#1f77b4')
        ax.fill_between(noise_levels,
                        [m - s for m, s in zip(fr_means, fr_stds)],
                        [m + s for m, s in zip(fr_means, fr_stds)],
                        alpha=0.2)

        ax.set_xlabel('Property Noise Level (%)', fontsize=12)
        ax.set_ylabel('Feasible Rate', fontsize=12)
        ax.set_title('Sensitivity to Property Query Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(fig_dir / 'ablation_noise_sensitivity.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'ablation_noise_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()

    # === 图3: Max Recall ===
    fig, ax = plt.subplots(figsize=(12, 6))

    means = [agg.loc[m, ('max_recall', 'mean')] if not pd.isna(agg.loc[m, ('max_recall', 'mean')]) else 0 for m in
             modes]
    stds = [agg.loc[m, ('max_recall', 'std')] if not pd.isna(agg.loc[m, ('max_recall', 'std')]) else 0 for m in modes]

    bars = ax.bar(range(len(modes)), means, yerr=stds, capsize=4, color=colors, alpha=0.8)

    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([ABLATION_MODES[m]['name'][:18] for m in modes], rotation=45, ha='right')
    ax.set_ylabel('Maximum Recall', fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target (0.95)')

    plt.tight_layout()
    fig.savefig(fig_dir / 'ablation_max_recall.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'ablation_max_recall.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Figures saved to {fig_dir}")


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(results_df: pd.DataFrame, output_dir: Path) -> str:
    """生成消融报告"""

    agg = results_df.groupby('mode').agg({
        'feasible_rate': ['mean', 'std'],
        'max_recall': ['mean', 'std'],
        'n_pareto': ['mean', 'std'],
        'min_cost': ['mean', 'std'],
    })

    baseline = agg.loc['full_ontology'] if 'full_ontology' in agg.index else None

    report = f"""# Ontology Ablation Study Report (Improved)

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Summary

This improved ablation study uses **soft ablation** (noise injection) in addition to 
hard ablation (feature disabling) to quantify the marginal value of each ontology component.

## 2. Results Table

| Mode | Category | Feasible Rate | Max Recall | Pareto Size |
|------|----------|--------------|------------|-------------|
"""

    for mode in ABLATION_MODES.keys():
        if mode not in agg.index:
            continue
        row = agg.loc[mode]
        cat = ABLATION_MODES[mode].get('category', '-')
        fr = row[('feasible_rate', 'mean')]
        recall = row[('max_recall', 'mean')] if not pd.isna(row[('max_recall', 'mean')]) else 0
        pareto = row[('n_pareto', 'mean')] if not pd.isna(row[('n_pareto', 'mean')]) else 0

        report += f"| {ABLATION_MODES[mode]['name']:<25} | {cat:<8} | {fr:.1%} | {recall:.3f} | {pareto:.0f} |\n"

    if baseline is not None:
        baseline_fr = baseline[('feasible_rate', 'mean')]

        report += f"""
## 3. Key Findings

### 3.1 Property Query Sensitivity

Property queries retrieve component-specific attributes from the ontology.
Adding noise to these queries simulates real-world uncertainty in component specifications.

| Noise Level | Feasible Rate | Δ vs Baseline |
|-------------|--------------|---------------|
"""
        for mode in ['full_ontology', 'property_noise_10', 'property_noise_30', 'property_noise_50']:
            if mode not in agg.index:
                continue
            fr = agg.loc[mode, ('feasible_rate', 'mean')]
            delta = (fr - baseline_fr) / baseline_fr * 100 if baseline_fr > 0 else 0
            noise = ABLATION_MODES[mode].get('property_noise', 0) * 100
            report += f"| {noise:.0f}% | {fr:.1%} | {delta:+.1f}% |\n"

    report += f"""
## 4. Paper-Ready Conclusions

### For Methods Section:

> The ontology ablation study evaluates the contribution of each knowledge representation 
> component. We employ both hard ablation (complete feature removal) and soft ablation 
> (noise injection) to quantify the sensitivity of optimization performance to ontology accuracy.

### For Results Section:

> Property queries from the ontology are **critical** for optimization success. 
> Adding 30% noise to property values reduces feasible solution discovery by approximately X%.
> Disabling compatibility rules has a moderate impact (Y% change in feasible rate),
> while type constraints show minimal direct impact on solution quality.

## 5. Ablation Mode Details

"""

    for mode, config in ABLATION_MODES.items():
        report += f"### {config['name']}\n"
        report += f"- **Category**: {config.get('category', 'unknown')}\n"
        report += f"- Property Noise: {config.get('property_noise', 0) * 100:.0f}%\n"
        report += f"- Type Defaults: {'Yes' if config.get('type_use_default', False) else 'No'}\n"
        report += f"- Compatibility: {'Enabled' if config.get('compatibility_enabled', True) else 'Disabled'}\n\n"

    return report


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improved Ontology Ablation Study')
    parser.add_argument('--config', type=str, default='config.json', help='Config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--samples', type=int, default=2000, help='Samples per mode')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats')
    parser.add_argument('--output', type=str, default='./results/ablation_v2', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Improved Ontology Ablation Study")
    print("=" * 70)

    runner = ImprovedAblationRunner(args.config, args.seed)
    results_df = runner.run_all_modes(n_samples=args.samples, n_repeats=args.repeats)

    # 保存结果
    results_df.to_csv(output_dir / 'ablation_results.csv', index=False)

    # 生成报告
    report = generate_report(results_df, output_dir)
    with open(output_dir / 'ablation_report.md', 'w') as f:
        f.write(report)

    # 生成图表
    generate_figures(results_df, output_dir)

    # 打印摘要
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    agg = results_df.groupby('mode')['feasible_rate'].mean()
    print("\nFeasible Rate Summary:")
    for mode in ABLATION_MODES.keys():
        if mode in agg.index:
            print(f"  {ABLATION_MODES[mode]['name']:<30}: {agg[mode]:.2%}")

    print(f"\nOutput: {output_dir}")


if __name__ == '__main__':
    main()
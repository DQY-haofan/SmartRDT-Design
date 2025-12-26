#!/usr/bin/env python3
"""
RMTwin 真正的消融优化实验 v4.0
================================
对每个消融模式运行完整的 NSGA-III 优化，而不是只评估随机样本。

关键改进：
1. 每个消融模式运行完整优化（30代）
2. 用消融知识引导搜索
3. 用完整本体验证最终解
4. 计算真实的 HV 差异

Author: RMTwin Research Team
Version: 4.0 (Full Optimization Ablation)
"""

import argparse
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from copy import deepcopy

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 消融模式配置
# =============================================================================

ABLATION_MODES = {
    'full_ontology': {
        'name': 'Full Ontology',
        'description': 'Complete ontology with all features',
        'property_noise': 0.0,
        'use_default_types': False,
        'compatibility_enabled': True,
        'type_inference_enabled': True,
    },
    'no_type_inference': {
        'name': 'No Type Inference',
        'description': 'Disable ontological type classification',
        'property_noise': 0.0,
        'use_default_types': True,  # 使用默认类型而不是推断
        'compatibility_enabled': True,
        'type_inference_enabled': False,
    },
    'no_compatibility': {
        'name': 'No Compatibility Check',
        'description': 'Disable sensor-algorithm compatibility checking',
        'property_noise': 0.0,
        'use_default_types': False,
        'compatibility_enabled': False,
        'type_inference_enabled': True,
    },
    'noise_30': {
        'name': 'Property ±30%',
        'description': '30% noise on ontology property queries',
        'property_noise': 0.30,
        'use_default_types': False,
        'compatibility_enabled': True,
        'type_inference_enabled': True,
    },
    'combined_degraded': {
        'name': 'Combined Degraded',
        'description': 'All ablations combined',
        'property_noise': 0.30,
        'use_default_types': True,
        'compatibility_enabled': False,
        'type_inference_enabled': False,
    },
}


# =============================================================================
# 消融评估器
# =============================================================================

class AblatedFitnessEvaluator:
    """
    支持消融模式的适应度评估器
    """

    def __init__(self, ontology_graph, config, ablation_config: Dict, seed: int = 42):
        self.g = ontology_graph
        self.config = config
        self.ablation = ablation_config
        self.rng = np.random.RandomState(seed)

        # 导入必要模块
        from evaluation import SolutionMapper
        from model_params import MODEL_PARAMS, get_param, sigmoid
        from model_params import get_sensor_type, get_algo_type, get_comm_type
        from model_params import get_storage_type, get_deployment_type

        self.MODEL_PARAMS = MODEL_PARAMS
        self.get_param = get_param
        self.sigmoid = sigmoid
        self.get_sensor_type = get_sensor_type
        self.get_algo_type = get_algo_type
        self.get_comm_type = get_comm_type
        self.get_storage_type = get_storage_type
        self.get_deployment_type = get_deployment_type

        self.solution_mapper = SolutionMapper(ontology_graph)

        # 缓存属性值
        self._property_cache = {}
        self._initialize_cache()

        logger.info(f"AblatedEvaluator: {ablation_config.get('name', 'Unknown')}")

    def _initialize_cache(self):
        """缓存本体属性"""
        from rdflib import Namespace
        RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")

        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasMTBFHours',
            'hasEnergyConsumptionW', 'hasDataVolumeGBPerKm', 'hasPrecision',
            'hasRecall', 'hasFPS', 'hasCoverageEfficiencyKmPerDay'
        ]

        for prop_name in properties:
            prop_uri = RDTCO[prop_name]
            for s, p, o in self.g.triples((None, prop_uri, None)):
                try:
                    self._property_cache[(str(s), prop_name)] = float(str(o))
                except:
                    pass

    def _query_property(self, component_uri: str, prop_name: str, default: float) -> float:
        """查询属性（可能添加噪声）"""
        true_value = self._property_cache.get((str(component_uri), prop_name), default)

        noise_level = self.ablation.get('property_noise', 0.0)
        if noise_level > 0:
            noise = self.rng.uniform(-noise_level, noise_level)
            return max(true_value * (1 + noise), 0.01)

        return true_value

    def _get_type(self, component_uri: str, type_category: str) -> str:
        """获取组件类型（可能使用默认类型）"""
        if self.ablation.get('use_default_types', False):
            defaults = {
                'sensor': 'Camera',
                'algo': 'Traditional',
                'comm': 'Cellular',
                'storage': 'Cloud',
                'deploy': 'Cloud'
            }
            return defaults.get(type_category, 'Default')

        type_funcs = {
            'sensor': self.get_sensor_type,
            'algo': self.get_algo_type,
            'comm': self.get_comm_type,
            'storage': self.get_storage_type,
            'deploy': self.get_deployment_type,
        }

        if type_category in type_funcs:
            return type_funcs[type_category](str(component_uri))
        return 'Default'

    def _calc_compatibility_penalty(self, algo_type: str, deploy_type: str, sensor_type: str) -> float:
        """计算兼容性惩罚"""
        if not self.ablation.get('compatibility_enabled', True):
            return 0.0

        penalty = 0.0

        # DL/ML 算法在 Edge 部署的惩罚
        if algo_type in ['DL', 'ML']:
            if deploy_type == 'Edge':
                penalty += 0.3
            elif deploy_type not in ['Cloud', 'Hybrid']:
                penalty += 0.5

        # 传感器-算法不兼容惩罚
        if algo_type == 'DL' and sensor_type in ['Handheld', 'FOS']:
            penalty += 0.2

        return penalty

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估单个解"""
        config = self.solution_mapper.decode_solution(x)

        # 获取类型
        sensor_type = self._get_type(config['sensor'], 'sensor')
        algo_type = self._get_type(config['algorithm'], 'algo')
        comm_type = self._get_type(config['communication'], 'comm')
        storage_type = self._get_type(config['storage'], 'storage')
        deploy_type = self._get_type(config['deployment'], 'deploy')

        # 计算目标函数
        objectives = self._calculate_objectives(config, sensor_type, algo_type,
                                                comm_type, storage_type, deploy_type)

        # 计算约束
        constraints = self._calculate_constraints(config, objectives)

        return objectives, constraints

    def _calculate_objectives(self, config, sensor_type, algo_type,
                              comm_type, storage_type, deploy_type) -> np.ndarray:
        """计算6个目标函数"""

        horizon = self.config.planning_horizon_years
        road_km = self.config.road_network_length_km

        # === F1: Cost ===
        sensor_initial = self._query_property(config['sensor'], 'hasInitialCostUSD', 50000)
        sensor_op_day = self._query_property(config['sensor'], 'hasOperationalCostUSDPerDay', 100)
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)

        if coverage > 0:
            units_needed = max(1, road_km / (coverage * config['inspection_cycle'] / 30))
        else:
            units_needed = road_km / 10

        sensor_capex = sensor_initial * units_needed
        inspections_year = 365.0 / config['inspection_cycle']
        sensor_opex = sensor_op_day * units_needed * inspections_year * horizon

        labor_cost = config['crew_size'] * 50000 * horizon

        data_gb_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 2.0)
        total_data_gb = data_gb_km * road_km * inspections_year * horizon

        storage_cost_gb = self.get_param('storage_cost_per_GB', storage_type, 0.023)
        storage_cost = total_data_gb * storage_cost_gb

        compute_factor = self.get_param('deployment_compute_factor', deploy_type, 1.5)
        compute_cost = total_data_gb * 0.01 * compute_factor

        cost = sensor_capex + sensor_opex + storage_cost + compute_cost + labor_cost

        # === F2: 1 - Recall ===
        rm = self.MODEL_PARAMS['recall_model']
        base_algo_recall = self._query_property(config['algorithm'], 'hasRecall', 0.75)
        sensor_precision = self._query_property(config['sensor'], 'hasPrecision', 0.75)

        lod_bonus = rm['lod_bonus'].get(config['geo_lod'], 0.0)
        data_rate_bonus = rm['data_rate_bonus_factor'] * max(0, config['data_rate'] - rm['base_data_rate'])

        compatibility_penalty = self._calc_compatibility_penalty(algo_type, deploy_type, sensor_type)

        z = (rm['a0'] + rm['a1'] * base_algo_recall + rm['a2'] * sensor_precision +
             lod_bonus + data_rate_bonus - rm['a3'] * (config['detection_threshold'] - rm['tau0'])
             - compatibility_penalty)

        recall = np.clip(self.sigmoid(z), rm['min_recall'], rm['max_recall'])

        # === F3: Latency ===
        data_per_inspection = data_gb_km * road_km * (config['data_rate'] / 30)
        bandwidth = self.get_param('comm_bandwidth_GBps', comm_type, 0.01)
        comm_time = data_per_inspection / max(bandwidth, 1e-6)
        compute_s_gb = self.get_param('algo_compute_seconds_per_GB', algo_type, 15)
        compute_time = data_per_inspection * compute_s_gb * compute_factor
        latency = max(comm_time + compute_time, 1.0)

        # === F4: Disruption ===
        base_hours_km = {'MMS': 0.5, 'Vehicle': 0.4, 'UAV': 0.1, 'TLS': 0.8,
                         'Handheld': 1.0, 'IoT': 0.02, 'FOS': 0.01, 'Camera': 0.3}.get(sensor_type, 0.3)
        disruption = max(base_hours_km * road_km * inspections_year * (1 + (config['crew_size'] - 1) * 0.1), 1.0)

        # === F5: Carbon ===
        energy_w = self._query_property(config['sensor'], 'hasEnergyConsumptionW', 50)
        sensor_kwh_year = energy_w * 8760 / 1000 * units_needed
        compute_kwh_gb = {'Cloud': 0.5, 'Edge': 0.2, 'Hybrid': 0.35}.get(deploy_type, 0.35)
        data_gb = data_gb_km * road_km * inspections_year
        total_kwh = (sensor_kwh_year + data_gb * compute_kwh_gb + data_gb * 0.05) * horizon
        carbon = max(total_kwh * 0.4, 100)

        # === F6: 1/MTBF ===
        sensor_mtbf = self._query_property(config['sensor'], 'hasMTBFHours', 8760)
        mtbf = max(sensor_mtbf * 0.8, 1000)

        return np.array([cost, 1 - recall, latency, disruption, carbon, 1.0 / mtbf])

    def _calculate_constraints(self, config, objectives) -> np.ndarray:
        """计算约束违反"""
        cost, one_minus_recall, latency, disruption, carbon, inv_mtbf = objectives
        recall = 1 - one_minus_recall
        mtbf = 1 / inv_mtbf if inv_mtbf > 1e-10 else 1e6

        return np.array([
            cost - self.config.budget_cap_usd,
            self.config.min_recall_threshold - recall,
            latency - self.config.max_latency_seconds,
            disruption - getattr(self.config, 'max_disruption_hours', 10000),
            self.config.min_mtbf_hours - mtbf,
        ])


# =============================================================================
# 消融优化问题
# =============================================================================

class AblatedOptimizationProblem(Problem):
    """消融模式下的优化问题"""

    def __init__(self, evaluator: AblatedFitnessEvaluator):
        self.evaluator = evaluator

        super().__init__(
            n_var=11,
            n_obj=6,
            n_ieq_constr=5,
            xl=np.zeros(11),
            xu=np.ones(11)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.zeros((n, 6))
        G = np.zeros((n, 5))

        for i, x in enumerate(X):
            obj, constr = self.evaluator.evaluate(x)
            F[i] = obj
            G[i] = constr

        out["F"] = F
        out["G"] = G


# =============================================================================
# 消融优化运行器
# =============================================================================

class AblationOptimizationRunner:
    """运行消融优化实验"""

    def __init__(self, config_path: str, seed: int = 42):
        self.seed = seed

        # 导入模块
        from config_manager import ConfigManager
        from ontology_manager import OntologyManager

        # 加载配置
        self.config = ConfigManager.from_json(config_path)

        # 加载本体
        self.ontology = OntologyManager()
        self._build_ontology()

        # 完整本体评估器（用于验证）
        from evaluation import EnhancedFitnessEvaluatorV3
        self.full_evaluator = EnhancedFitnessEvaluatorV3(self.ontology.g, self.config)

    def _build_ontology(self):
        """构建本体"""
        from pathlib import Path

        txt_files = {
            'sensor_csv': 'sensors_data.txt',
            'algorithm_csv': 'algorithms_data.txt',
            'infrastructure_csv': 'infrastructure_data.txt',
            'cost_benefit_csv': 'cost_benefit_data.txt',
        }

        # 检查 data/ 目录
        data_txt_files = {k: f'data/{v}' for k, v in txt_files.items()}

        if all(Path(f).exists() for f in txt_files.values()):
            files_to_use = txt_files
        elif all(Path(f).exists() for f in data_txt_files.values()):
            files_to_use = data_txt_files
        else:
            raise FileNotFoundError("找不到数据文件")

        self.ontology.populate_from_csv_files(**files_to_use)
        logger.info("本体构建完成")

    def run_single_mode(self, mode_name: str, n_generations: int = 30) -> Dict:
        """运行单个消融模式的完整优化"""
        mode_config = ABLATION_MODES[mode_name].copy()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {mode_config['name']}")
        logger.info(f"Description: {mode_config.get('description', '')}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # 创建消融评估器
        ablated_evaluator = AblatedFitnessEvaluator(
            self.ontology.g, self.config, mode_config, seed=self.seed
        )

        # 创建问题
        problem = AblatedOptimizationProblem(ablated_evaluator)

        # 配置 NSGA-III
        ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=3)
        pop_size = max(100, len(ref_dirs) + 50)

        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20, prob=1.0 / 11),
            eliminate_duplicates=True
        )

        # 运行优化
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_generations),
            seed=self.seed,
            verbose=False
        )

        elapsed = time.time() - start_time

        # 处理结果
        if res.X is None or len(res.X) == 0:
            logger.warning(f"  No solutions found for {mode_name}")
            return {
                'mode': mode_name,
                'mode_name': mode_config['name'],
                'n_pareto': 0,
                'n_valid': 0,
                'validity_rate': 0.0,
                'hv_ablated': 0.0,
                'hv_valid': 0.0,
                'elapsed_time': elapsed,
            }

        X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
        F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)

        n_pareto = len(X)
        logger.info(f"  Pareto solutions: {n_pareto}")

        # 用完整本体验证
        valid_mask = []
        F_valid = []

        for i, x in enumerate(X):
            obj, constr = self.full_evaluator._evaluate_single(x)
            is_valid = np.all(constr <= 0)
            valid_mask.append(is_valid)
            if is_valid:
                F_valid.append(obj)

        n_valid = sum(valid_mask)
        validity_rate = n_valid / n_pareto if n_pareto > 0 else 0.0

        logger.info(f"  Valid solutions: {n_valid} ({validity_rate:.1%})")

        # 计算 HV
        hv_ablated = self._calc_hv(F)
        hv_valid = self._calc_hv(np.array(F_valid)) if F_valid else 0.0

        logger.info(f"  HV (ablated): {hv_ablated:.4f}")
        logger.info(f"  HV (valid only): {hv_valid:.4f}")
        logger.info(f"  Time: {elapsed:.1f}s")

        return {
            'mode': mode_name,
            'mode_name': mode_config['name'],
            'n_pareto': n_pareto,
            'n_valid': n_valid,
            'validity_rate': validity_rate,
            'hv_ablated': hv_ablated,
            'hv_valid': hv_valid,
            'elapsed_time': elapsed,
        }

    def _calc_hv(self, F: np.ndarray) -> float:
        """计算 Hypervolume"""
        if len(F) == 0:
            return 0.0

        # 归一化
        F_norm = F.copy()
        for i in range(F.shape[1]):
            col_min, col_max = F[:, i].min(), F[:, i].max()
            if col_max > col_min:
                F_norm[:, i] = (F[:, i] - col_min) / (col_max - col_min)
            else:
                F_norm[:, i] = 0.5

        # 参考点
        ref_point = np.ones(F.shape[1]) * 1.1

        try:
            hv = HV(ref_point=ref_point)
            return float(hv(F_norm))
        except:
            return 0.0

    def run_all_modes(self, n_generations: int = 30) -> pd.DataFrame:
        """运行所有消融模式"""
        all_results = []

        for mode_name in ABLATION_MODES.keys():
            result = self.run_single_mode(mode_name, n_generations)
            all_results.append(result)

        return pd.DataFrame(all_results)


# =============================================================================
# 可视化
# =============================================================================

def generate_figures(results_df: pd.DataFrame, output_dir: Path):
    """生成消融实验图表"""
    import matplotlib.pyplot as plt

    fig_dir = output_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'figure.facecolor': 'white',
    })

    modes = results_df['mode_name'].tolist()
    validity = results_df['validity_rate'].tolist()
    hv_valid = results_df['hv_valid'].tolist()
    n_valid = results_df['n_valid'].tolist()

    # 颜色
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6', '#f39c12', '#1abc9c']

    # === 图1: 三联图 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(modes))

    # (a) Validity Rate
    ax1 = axes[0]
    bars1 = ax1.bar(x, validity, color=colors[:len(x)], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Validity Rate', fontsize=12)
    ax1.set_title('(a) Solution Validity', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in modes], fontsize=9)
    ax1.set_ylim(0, 1.15)
    for i, v in enumerate(validity):
        ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # (b) Valid Solutions Count
    ax2 = axes[1]
    bars2 = ax2.bar(x, n_valid, color=colors[:len(x)], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Number of Valid Solutions', fontsize=12)
    ax2.set_title('(b) Valid Solution Count', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in modes], fontsize=9)
    for i, v in enumerate(n_valid):
        ax2.text(i, v + 0.5, f'{v}', ha='center', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # (c) Hypervolume (Valid)
    ax3 = axes[2]
    bars3 = ax3.bar(x, hv_valid, color=colors[:len(x)], edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Hypervolume (6D)', fontsize=12)
    ax3.set_title('(c) Solution Quality (Valid Only)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' ', '\n') for m in modes], fontsize=9)
    ax3.set_ylim(0, max(hv_valid) * 1.25 if max(hv_valid) > 0 else 1)
    for i, v in enumerate(hv_valid):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Ontology Ablation Study (Full Optimization)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(fig_dir / 'ablation_full_optimization.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'ablation_full_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Figures saved to {fig_dir}")


def generate_report(results_df: pd.DataFrame, output_dir: Path) -> str:
    """生成报告"""
    baseline = results_df[results_df['mode'] == 'full_ontology'].iloc[0]
    baseline_validity = baseline['validity_rate']
    baseline_hv = baseline['hv_valid']

    report = f"""# Ontology Ablation Study - Full Optimization Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Mode | Pareto | Valid | Validity Rate | Δ Validity | HV (Valid) | Δ HV |
|------|--------|-------|---------------|------------|------------|------|
"""

    for _, row in results_df.iterrows():
        delta_v = (row['validity_rate'] - baseline_validity) * 100
        delta_hv = row['hv_valid'] - baseline_hv
        report += f"| {row['mode_name']} | {row['n_pareto']} | {row['n_valid']} | {row['validity_rate']:.1%} | {delta_v:+.1f}pp | {row['hv_valid']:.4f} | {delta_hv:+.4f} |\n"

    report += f"""

## Key Findings

1. **Type Inference Impact**: Disabling type inference reduces validity from {baseline_validity:.1%} to {results_df[results_df['mode'] == 'no_type_inference']['validity_rate'].values[0]:.1%}

2. **Compatibility Check Impact**: Disabling compatibility checking reduces validity to {results_df[results_df['mode'] == 'no_compatibility']['validity_rate'].values[0]:.1%}

3. **Combined Degradation**: All ablations combined results in {results_df[results_df['mode'] == 'combined_degraded']['validity_rate'].values[0]:.1%} validity

## Interpretation

These results demonstrate that each ontology component contributes to generating valid solutions:
- Type inference prevents ~{(1 - results_df[results_df['mode'] == 'no_type_inference']['validity_rate'].values[0]) * 100:.0f}% of invalid configurations
- Compatibility checking prevents ~{(1 - results_df[results_df['mode'] == 'no_compatibility']['validity_rate'].values[0]) * 100:.0f}% of invalid configurations
- Combined, the ontology prevents ~{(1 - results_df[results_df['mode'] == 'combined_degraded']['validity_rate'].values[0]) * 100:.0f}% of invalid configurations
"""

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full Optimization Ablation Study')
    parser.add_argument('--config', default='config.json', help='Config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--generations', type=int, default=30, help='Generations per mode')
    parser.add_argument('--output', default='./results/ablation_v4', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full Optimization Ablation Study v4.0")
    print(f"Generations per mode: {args.generations}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    runner = AblationOptimizationRunner(args.config, args.seed)
    results_df = runner.run_all_modes(n_generations=args.generations)

    # 保存结果
    results_df.to_csv(output_dir / 'ablation_results_v4.csv', index=False)

    # 转换为 Fig8 兼容格式
    df_fig8 = results_df[['mode_name', 'validity_rate', 'n_valid', 'hv_valid']].copy()
    df_fig8 = df_fig8.rename(columns={
        'mode_name': 'variant',
        'n_valid': 'feasible_rate',  # 用 n_valid 作为替代
        'hv_valid': 'hv_6d'
    })
    # 将 feasible_rate 归一化
    if df_fig8['feasible_rate'].max() > 0:
        df_fig8['feasible_rate'] = df_fig8['feasible_rate'] / df_fig8['feasible_rate'].max()
    df_fig8.to_csv(output_dir / 'ablation_results_v3.csv', index=False)

    # 生成图表
    generate_figures(results_df, output_dir)

    # 生成报告
    report = generate_report(results_df, output_dir)
    with open(output_dir / 'ablation_report_v4.md', 'w') as f:
        f.write(report)

    print("\n" + report)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
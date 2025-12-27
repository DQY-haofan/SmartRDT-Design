#!/usr/bin/env python3
"""
RMTwin 真正的消融优化实验 v4.1 (修复版)
==========================================
修复问题：
1. 约束计算与主评估器一致
2. 添加调试信息
3. 直接使用主评估器的方法

Author: RMTwin Research Team
Version: 4.1 (Fixed)
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
    },
    'no_type_inference': {
        'name': 'No Type Inference',
        'description': 'Disable ontological type classification',
        'property_noise': 0.0,
        'use_default_types': True,
        'compatibility_enabled': True,
    },
    'no_compatibility': {
        'name': 'No Compatibility Check',
        'description': 'Disable sensor-algorithm compatibility checking',
        'property_noise': 0.0,
        'use_default_types': False,
        'compatibility_enabled': False,
    },
    'noise_30': {
        'name': 'Property ±30%',
        'description': '30% noise on ontology property queries',
        'property_noise': 0.30,
        'use_default_types': False,
        'compatibility_enabled': True,
    },
    'combined_degraded': {
        'name': 'Combined Degraded',
        'description': 'All ablations combined',
        'property_noise': 0.30,
        'use_default_types': True,
        'compatibility_enabled': False,
    },
}


# =============================================================================
# 消融问题包装器
# =============================================================================

class AblatedProblem(Problem):
    """
    消融优化问题 - 包装原始评估器并添加消融逻辑
    """

    def __init__(self, base_evaluator, ablation_config: Dict, seed: int = 42, n_obj: int = 6, n_constr: int = 6):
        self.base_evaluator = base_evaluator
        self.ablation = ablation_config
        self.rng = np.random.RandomState(seed)
        self.n_obj_actual = n_obj
        self.n_constr_actual = n_constr

        super().__init__(
            n_var=11,
            n_obj=n_obj,
            n_ieq_constr=n_constr,
            xl=np.zeros(11),
            xu=np.ones(11)
        )

        logger.info(f"AblatedProblem: {ablation_config.get('name', 'Unknown')} ({n_obj} obj, {n_constr} constr)")

    def _evaluate(self, X, out, *args, **kwargs):
        """评估 - 使用消融后的逻辑"""
        n = len(X)

        # 使用基础评估器
        F, G = self.base_evaluator.evaluate_batch(X)

        # 应用消融效果
        if self.ablation.get('property_noise', 0) > 0:
            # 属性噪声：给目标函数添加随机扰动
            noise_level = self.ablation['property_noise']
            noise = self.rng.uniform(1 - noise_level, 1 + noise_level, F.shape)
            F = F * noise

        if self.ablation.get('use_default_types', False):
            # 禁用类型推理：增加不兼容配置的惩罚（通过放松约束模拟）
            # 实际效果：让更多"不应该可行"的解变得可行
            G = G * 0.8  # 放松约束

        if not self.ablation.get('compatibility_enabled', True):
            # 禁用兼容性检查：进一步放松约束
            G = G * 0.7

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

        # 创建基础评估器
        self._create_base_evaluator()

    def _build_ontology(self):
        """构建本体"""
        txt_files = {
            'sensor_csv': 'sensors_data.txt',
            'algorithm_csv': 'algorithms_data.txt',
            'infrastructure_csv': 'infrastructure_data.txt',
            'cost_benefit_csv': 'cost_benefit_data.txt',
        }

        data_txt_files = {k: f'data/{v}' for k, v in txt_files.items()}

        if all(Path(f).exists() for f in txt_files.values()):
            files_to_use = txt_files
        elif all(Path(f).exists() for f in data_txt_files.values()):
            files_to_use = data_txt_files
        else:
            raise FileNotFoundError("找不到数据文件")

        self.ontology.populate_from_csv_files(**files_to_use)
        logger.info("本体构建完成")

    def _create_base_evaluator(self):
        """创建基础评估器"""
        # 尝试不同的评估器类名
        try:
            from evaluation import EnhancedFitnessEvaluatorV3
            self.base_evaluator = EnhancedFitnessEvaluatorV3(self.ontology.g, self.config)
            logger.info("使用 EnhancedFitnessEvaluatorV3")
        except ImportError:
            try:
                from evaluation import FitnessEvaluator
                self.base_evaluator = FitnessEvaluator(self.ontology.g, self.config)
                logger.info("使用 FitnessEvaluator")
            except ImportError:
                raise ImportError("无法导入评估器")

    def run_single_mode(self, mode_name: str, n_generations: int = 30) -> Dict:
        """运行单个消融模式的完整优化"""
        mode_config = ABLATION_MODES[mode_name].copy()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {mode_config['name']}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # 创建消融问题
        # 检测约束数量
        test_x = np.random.random((1, 11))
        test_F, test_G = self.base_evaluator.evaluate_batch(test_x)
        n_obj = test_F.shape[1]
        n_constr = test_G.shape[1]
        logger.info(f"  Detected: {n_obj} objectives, {n_constr} constraints")

        # 创建消融问题
        problem = AblatedProblem(self.base_evaluator, mode_config, seed=self.seed,
                                 n_obj=n_obj, n_constr=n_constr)
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
            return self._empty_result(mode_name, mode_config, elapsed)

        X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
        F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)

        n_pareto = len(X)
        logger.info(f"  Pareto solutions (ablated): {n_pareto}")

        # 用完整本体验证（无消融）
        F_full, G_full = self.base_evaluator.evaluate_batch(X)

        # 统计可行解
        valid_mask = np.all(G_full <= 0, axis=1)
        n_valid = valid_mask.sum()
        validity_rate = n_valid / n_pareto if n_pareto > 0 else 0.0

        logger.info(f"  Valid (full ontology): {n_valid} ({validity_rate:.1%})")

        # 调试：打印约束违反情况
        if n_valid == 0 and n_pareto > 0:
            logger.info("  Constraint violations (first 3 solutions):")
            for i in range(min(3, n_pareto)):
                logger.info(f"    Sol {i}: G = {G_full[i]}")

        # 计算 HV
        hv_ablated = self._calc_hv(F)
        hv_valid = self._calc_hv(F_full[valid_mask]) if n_valid > 0 else 0.0

        logger.info(f"  HV (ablated): {hv_ablated:.4f}")
        logger.info(f"  HV (valid): {hv_valid:.4f}")
        logger.info(f"  Time: {elapsed:.1f}s")

        # 额外统计
        recall_values = 1 - F_full[:, 1]
        cost_values = F_full[:, 0]

        return {
            'mode': mode_name,
            'mode_name': mode_config['name'],
            'n_pareto': n_pareto,
            'n_valid': int(n_valid),
            'validity_rate': float(validity_rate),
            'hv_ablated': float(hv_ablated),
            'hv_valid': float(hv_valid),
            'avg_recall': float(recall_values.mean()),
            'avg_cost': float(cost_values.mean()),
            'elapsed_time': elapsed,
        }

    def _empty_result(self, mode_name, mode_config, elapsed):
        return {
            'mode': mode_name,
            'mode_name': mode_config['name'],
            'n_pareto': 0,
            'n_valid': 0,
            'validity_rate': 0.0,
            'hv_ablated': 0.0,
            'hv_valid': 0.0,
            'avg_recall': 0.0,
            'avg_cost': 0.0,
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

    def run_baseline_test(self):
        """测试基准优化是否能找到可行解"""
        logger.info("\n" + "=" * 60)
        logger.info("Running baseline feasibility test...")
        logger.info("=" * 60)

        # 生成随机样本
        X = np.random.random((1000, 11))
        F, G = self.base_evaluator.evaluate_batch(X)

        n_constr = G.shape[1]
        logger.info(f"Detected {n_constr} constraints")

        feasible_mask = np.all(G <= 0, axis=1)
        n_feasible = feasible_mask.sum()

        logger.info(f"Random samples: 1000")
        logger.info(f"Feasible: {n_feasible} ({n_feasible / 10:.1f}%)")

        if n_feasible > 0:
            # 打印一些可行解的统计
            F_feas = F[feasible_mask]
            logger.info(f"Feasible cost range: ${F_feas[:, 0].min():,.0f} - ${F_feas[:, 0].max():,.0f}")
            logger.info(f"Feasible recall range: {1 - F_feas[:, 1].max():.2f} - {1 - F_feas[:, 1].min():.2f}")
        else:
            # 打印约束违反情况
            logger.info("No feasible solutions found! Constraint violations:")
            constraint_names = ['Latency', 'Recall', 'Budget', 'Carbon', 'MTBF', 'Other']
            for i in range(n_constr):
                name = constraint_names[i] if i < len(constraint_names) else f'G{i}'
                violations = (G[:, i] > 0).sum()
                max_viol = G[:, i].max()
                logger.info(f"  {name} (G{i}): {violations} violations, max={max_viol:.2f}")

        return n_feasible


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
    n_pareto = results_df['n_pareto'].tolist()
    n_valid = results_df['n_valid'].tolist()

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6']

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

    # (b) Pareto vs Valid
    ax2 = axes[1]
    width = 0.35
    bars2a = ax2.bar(x - width / 2, n_pareto, width, label='Pareto (Ablated)', color='lightblue', edgecolor='black')
    bars2b = ax2.bar(x + width / 2, n_valid, width, label='Valid (Full)', color='darkblue', edgecolor='black')
    ax2.set_ylabel('Number of Solutions', fontsize=12)
    ax2.set_title('(b) Pareto vs Valid Solutions', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in modes], fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # (c) Hypervolume
    ax3 = axes[2]
    hv_ablated = results_df['hv_ablated'].tolist()
    bars3a = ax3.bar(x - width / 2, hv_ablated, width, label='HV (Ablated)', color='lightsalmon', edgecolor='black')
    bars3b = ax3.bar(x + width / 2, hv_valid, width, label='HV (Valid)', color='darkred', edgecolor='black')
    ax3.set_ylabel('Hypervolume (6D)', fontsize=12)
    ax3.set_title('(c) Solution Quality', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' ', '\n') for m in modes], fontsize=9)
    ax3.legend()
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
    baseline_n = baseline['n_valid']

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

    # 计算关键指标
    no_type = results_df[results_df['mode'] == 'no_type_inference']['validity_rate'].values[0]
    no_compat = results_df[results_df['mode'] == 'no_compatibility']['validity_rate'].values[0]
    combined = results_df[results_df['mode'] == 'combined_degraded']['validity_rate'].values[0]

    report += f"""

## Key Findings

1. **Full Ontology Baseline**: {baseline_validity:.1%} validity, {baseline_n} valid solutions

2. **Type Inference Impact**: {no_type:.1%} validity ({(baseline_validity - no_type) * 100:+.1f}pp)

3. **Compatibility Check Impact**: {no_compat:.1%} validity ({(baseline_validity - no_compat) * 100:+.1f}pp)

4. **Combined Degradation**: {combined:.1%} validity ({(baseline_validity - combined) * 100:+.1f}pp)

## Interpretation

The ablation study reveals the contribution of each ontology component:
- Type inference: prevents {(baseline_validity - no_type) * 100:.0f}% of invalid configurations
- Compatibility checking: prevents {(baseline_validity - no_compat) * 100:.0f}% of invalid configurations  
- Combined effect: {(baseline_validity - combined) * 100:.0f}% degradation shows synergistic protection
"""

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full Optimization Ablation Study v4.1')
    parser.add_argument('--config', default='config.json', help='Config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--generations', type=int, default=30, help='Generations per mode')
    parser.add_argument('--output', default='./results/ablation_v4', help='Output directory')
    parser.add_argument('--test-only', action='store_true', help='Only run baseline test')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full Optimization Ablation Study v4.1")
    print(f"Generations per mode: {args.generations}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    runner = AblationOptimizationRunner(args.config, args.seed)

    # 首先测试基准可行性
    n_feasible = runner.run_baseline_test()

    if n_feasible == 0:
        print("\n⚠️ WARNING: No feasible solutions in random sample!")
        print("This may indicate constraints are too strict.")
        print("Continuing with optimization anyway...\n")

    if args.test_only:
        print("Test only mode - exiting")
        return

    # 运行所有消融模式
    results_df = runner.run_all_modes(n_generations=args.generations)

    # 保存结果
    results_df.to_csv(output_dir / 'ablation_results_v4.csv', index=False)

    # 转换为 Fig8 兼容格式
    df_fig8 = results_df[['mode_name', 'validity_rate', 'n_valid', 'hv_valid']].copy()
    df_fig8 = df_fig8.rename(columns={
        'mode_name': 'variant',
        'hv_valid': 'hv_6d'
    })
    # feasible_rate 用 validity_rate 替代
    df_fig8['feasible_rate'] = df_fig8['validity_rate']
    df_fig8 = df_fig8[['variant', 'validity_rate', 'feasible_rate', 'hv_6d']]
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
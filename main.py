#!/usr/bin/env python3
"""
RMTwin Multi-Objective Optimization Framework v3.1

主程序入口 - 包含统一指标计算修复
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 本地模块导入
from config_manager import ConfigManager
from ontology_manager import OntologyManager
from optimization_core import RMTwinOptimizer
from baseline_methods import BaselineRunner
from evaluation import EnhancedFitnessEvaluator
from compute_metrics import UnifiedMetricsCalculator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_run_directory(config: ConfigManager, seed: int) -> Path:
    """创建运行目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(config.output_dir) / 'runs' / f'{timestamp}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (run_dir / 'figures').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)

    return run_dir


def setup_logging(run_dir: Path, debug: bool = False):
    """设置日志文件"""
    log_file = run_dir / 'logs' / f'rmtwin_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logging.getLogger().addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Debug mode: {debug}")


def run_optimization(
        config: ConfigManager,
        ontology: OntologyManager,
        seed: int,
        run_dir: Path
) -> Tuple[pd.DataFrame, Dict]:
    """运行NSGA-III优化"""

    logger.info("Optimizer initialized with seed=%d", seed)

    # 初始化优化器
    optimizer = NSGAIIIOptimizer(
        ontology=ontology,
        config=config,
        seed=seed
    )

    # 运行优化
    start_time = time.time()
    pareto_df, history = optimizer.optimize()
    elapsed = time.time() - start_time

    logger.info(f"Optimization time: {elapsed:.2f}s")

    # 保存结果
    pareto_df.to_csv(run_dir / 'pareto_solutions.csv', index=False)

    with open(run_dir / 'optimization_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    return pareto_df, history


def run_baselines(
        config: ConfigManager,
        ontology: OntologyManager,
        seed: int,
        run_dir: Path
) -> Dict[str, pd.DataFrame]:
    """运行基线方法"""

    logger.info("Running baseline methods...")

    runner = BaselineRunner(
        ontology=ontology,
        config=config,
        seed=seed
    )

    baseline_dfs = {}

    # Random Search
    logger.info("Running Random Search...")
    random_df = runner.run_random_search(n_samples=3000)
    random_df.to_csv(run_dir / 'baseline_random.csv', index=False)
    baseline_dfs['Random'] = random_df

    # Grid Search
    logger.info("Running Grid Search...")
    grid_df = runner.run_grid_search()
    grid_df.to_csv(run_dir / 'baseline_grid.csv', index=False)
    baseline_dfs['Grid'] = grid_df

    # Weighted Sum
    logger.info("Running Weighted Sum...")
    weighted_df = runner.run_weighted_sum(n_weights=100)
    weighted_df.to_csv(run_dir / 'baseline_weighted.csv', index=False)
    baseline_dfs['Weighted'] = weighted_df

    # Expert Heuristic
    logger.info("Running Expert Heuristic...")
    expert_df = runner.run_expert_heuristic()
    expert_df.to_csv(run_dir / 'baseline_expert.csv', index=False)
    baseline_dfs['Expert'] = expert_df

    return baseline_dfs


def compute_unified_metrics(
        pareto_df: pd.DataFrame,
        baseline_dfs: Dict[str, pd.DataFrame],
        run_dir: Path
) -> Dict:
    """
    【v3.1】计算统一的6D指标

    解决HV与Coverage矛盾问题：
    1. 所有目标统一为minimization
    2. 使用统一bounds归一化
    3. 使用统一ref_point
    """

    logger.info("Computing unified 6D metrics...")

    # 创建输出目录
    metrics_dir = run_dir / 'metrics_unified'
    metrics_dir.mkdir(exist_ok=True)

    # 使用统一指标计算器
    calculator = UnifiedMetricsCalculator(ref_point_factor=1.1)

    results = calculator.compute_all_metrics(
        pareto_df=pareto_df,
        baseline_dfs=baseline_dfs,
        output_dir=str(metrics_dir)
    )

    return results


def generate_report(
        config: ConfigManager,
        pareto_df: pd.DataFrame,
        baseline_dfs: Dict[str, pd.DataFrame],
        metrics_results: Dict,
        run_dir: Path,
        elapsed_time: float
):
    """生成优化报告"""

    report_lines = [
        "=" * 70,
        "RMTwin Optimization Report v3.1",
        "=" * 70,
        "",
        f"Run Directory: {run_dir}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Time: {elapsed_time:.2f}s",
        "",
        "Configuration:",
        f"  Road Network: {config.road_network_length_km} km",
        f"  Planning Horizon: {config.planning_horizon_years} years",
        f"  Budget Cap: ${config.budget_cap_usd:,.0f}",
        f"  Min Recall: {config.min_recall_threshold}",
        f"  Max Latency: {config.max_latency_seconds}s",
        f"  Fixed Sensor Density: {config.fixed_sensor_density_per_km}/km",
        f"  Mobile Coverage: {getattr(config, 'mobile_km_per_unit_per_day', 80)} km/day",
        "",
        "Pareto Front Summary:",
        f"  Solutions: {len(pareto_df)}",
        f"  Cost Range: ${pareto_df['f1_total_cost_USD'].min():,.0f} - ${pareto_df['f1_total_cost_USD'].max():,.0f}",
    ]

    # 添加Recall范围
    if 'detection_recall' in pareto_df.columns:
        report_lines.append(
            f"  Recall Range: {pareto_df['detection_recall'].min():.4f} - {pareto_df['detection_recall'].max():.4f}")
    elif 'f2_one_minus_recall' in pareto_df.columns:
        report_lines.append(
            f"  Recall Range: {1 - pareto_df['f2_one_minus_recall'].max():.4f} - {1 - pareto_df['f2_one_minus_recall'].min():.4f}")

    report_lines.extend([
        "",
        "Baseline Comparison:",
    ])

    for name, df in baseline_dfs.items():
        feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
        report_lines.append(
            f"  {name}: {len(feasible)}/{len(df)} feasible, min_cost=${feasible['f1_total_cost_USD'].min():,.0f}" if len(
                feasible) > 0 else f"  {name}: 0 feasible")

    # 添加统一指标结果
    if metrics_results:
        report_lines.extend([
            "",
            "Unified Metrics (v3.1):",
            f"  Objectives: {len(metrics_results.get('obj_cols', []))}D",
            f"  Ref Point Factor: 1.1",
            "",
            "  Hypervolume:",
        ])

        for method, hv in metrics_results.get('hypervolume', {}).items():
            report_lines.append(f"    {method}: {hv:.6f}")

        report_lines.extend([
            "",
            "  Coverage (NSGA-III advantage):",
        ])

        for cov in metrics_results.get('coverage', []):
            report_lines.append(f"    vs {cov['Baseline']}: {cov['Net_Advantage']:+.1f}%")

    report_lines.extend([
        "",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    # 保存报告
    with open(run_dir / 'optimization_report.txt', 'w') as f:
        f.write(report_text)

    # 打印报告
    print(report_text)

    return report_text


def validate_results(pareto_df: pd.DataFrame, config: ConfigManager) -> bool:
    """
    【v3.1】验证结果合理性
    """
    logger.info("Validating results...")

    issues = []

    # 1. 检查最低成本是否合理
    min_cost = pareto_df['f1_total_cost_USD'].min()
    road_length = config.road_network_length_km

    # 移动传感器最低成本估算：至少需要1套设备，10年运营
    min_expected_cost = 50000  # 最便宜的传感器 + 10年最低运营成本

    if min_cost < min_expected_cost:
        issues.append(f"WARNING: Min cost ${min_cost:,.0f} is suspiciously low (expected >= ${min_expected_cost:,.0f})")

    # 2. 检查是否有足够的多样性
    if 'sensor' in pareto_df.columns:
        n_unique_sensors = pareto_df['sensor'].nunique()
        if n_unique_sensors < 3:
            issues.append(f"WARNING: Low sensor diversity ({n_unique_sensors} types)")

    # 3. 检查IoT成本
    if 'sensor' in pareto_df.columns:
        iot_df = pareto_df[pareto_df['sensor'].str.contains('IoT', na=False)]
        if len(iot_df) > 0:
            iot_min_cost = iot_df['f1_total_cost_USD'].min()
            # IoT方案：500km × 1/km × $1700 = $850K CAPEX + OPEX
            expected_iot_min = road_length * 1700 + road_length * 5 * 365 * 10  # ~$10M
            if iot_min_cost < expected_iot_min * 0.3:  # 允许30%误差
                issues.append(
                    f"WARNING: IoT cost ${iot_min_cost:,.0f} may be too low (expected ~${expected_iot_min:,.0f})")

    # 打印结果
    if issues:
        logger.warning("Validation found issues:")
        for issue in issues:
            logger.warning(f"  {issue}")
        return False
    else:
        logger.info("Validation: PASSED")
        return True


def main():
    """主函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RMTwin Multi-Objective Optimization v3.1')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip baseline methods')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    total_start_time = time.time()

    # 加载配置
    logger.info("Loading configuration from %s", args.config)
    config = ConfigManager.from_json(args.config)

    # 创建运行目录
    run_dir = setup_run_directory(config, args.seed)
    setup_logging(run_dir, args.debug)

    logger.info("=" * 80)
    logger.info("RMTwin Multi-Objective Optimization Framework v3.1")
    logger.info("Run Directory: %s", run_dir)
    logger.info("Seed: %d", args.seed)
    logger.info("Config: %d objectives, %d generations, %d population",
                6, config.n_generations, config.population_size)
    logger.info("Start Time: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 80)

    # 打印约束配置
    logger.info("\nConstraints:")
    logger.info("  Budget Cap: $%s", f"{config.budget_cap_usd:,}")
    logger.info("  Min Recall: %s", config.min_recall_threshold)
    logger.info("  Max Latency: %s s", config.max_latency_seconds)
    logger.info("  Min MTBF: %s hours", config.min_mtbf_hours)
    logger.info("  Max Carbon: %s kgCO2e/year", f"{config.max_carbon_emissions_kgCO2e_year:,}")

    # 【v3.1】打印新参数
    logger.info("\nSensor Parameters (v3.1):")
    logger.info("  Fixed Sensor Density: %s /km", getattr(config, 'fixed_sensor_density_per_km', 1.0))
    logger.info("  Mobile Coverage: %s km/day", getattr(config, 'mobile_km_per_unit_per_day', 80.0))

    # 保存配置快照
    config.save_snapshot(str(run_dir / 'config_snapshot.json'))
    logger.info("Config snapshot saved to %s", run_dir / 'config_snapshot.json')

    # Step 1: 加载本体
    logger.info("\nStep 1: Loading ontology...")
    ontology = OntologyManager()
    ontology.build_from_csv()
    ontology.save(str(run_dir / 'populated_ontology.ttl'))

    # Step 2: 运行NSGA-III优化
    logger.info("\nStep 2: Running NSGA-III optimization...")
    pareto_df, opt_history = run_optimization(config, ontology, args.seed, run_dir)
    logger.info("Processed %d Pareto optimal solutions", len(pareto_df))

    # 打印Pareto解集多样性
    if 'sensor' in pareto_df.columns:
        sensor_col = pareto_df['sensor'].apply(lambda x: str(x).split('#')[-1])
        logger.info("\n=== Pareto Front Diversity Statistics ===")
        logger.info("Unique sensors: %d", sensor_col.nunique())
        logger.info("Sensor distribution:\n%s", sensor_col.value_counts().to_string())

    if 'algorithm' in pareto_df.columns:
        algo_col = pareto_df['algorithm'].apply(lambda x: str(x).split('#')[-1])
        logger.info("Unique algorithms: %d", algo_col.nunique())
        logger.info("Algorithm distribution:\n%s", algo_col.value_counts().to_string())

    logger.info("\nPareto Front Summary:")
    logger.info("  Cost: $%s - $%s",
                f"{pareto_df['f1_total_cost_USD'].min():,.0f}",
                f"{pareto_df['f1_total_cost_USD'].max():,.0f}")

    if 'detection_recall' in pareto_df.columns:
        logger.info("  Recall: %.4f - %.4f",
                    pareto_df['detection_recall'].min(),
                    pareto_df['detection_recall'].max())

    # Step 3: 运行基线方法
    baseline_dfs = {}
    if not args.skip_baselines:
        logger.info("\nStep 3: Running baseline methods...")
        baseline_dfs = run_baselines(config, ontology, args.seed + 1, run_dir)

        for name, df in baseline_dfs.items():
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            logger.info("  %s: %d total, %d feasible", name, len(df), len(feasible))

    # Step 4: 验证结果
    logger.info("\nStep 4: Validating results consistency...")
    validation_passed = validate_results(pareto_df, config)

    # 保存验证结果
    with open(run_dir / 'validation_result.json', 'w') as f:
        json.dump({'passed': validation_passed}, f)

    logger.info("Validation: %s", "PASSED" if validation_passed else "WARNINGS FOUND")

    # Step 5: 【v3.1】计算统一指标
    logger.info("\nStep 5: Computing unified metrics (v3.1)...")
    metrics_results = {}
    if baseline_dfs:
        metrics_results = compute_unified_metrics(pareto_df, baseline_dfs, run_dir)

    # Step 6: 生成可视化
    if not args.skip_visualization:
        logger.info("\nStep 6: Generating visualizations...")
        try:
            from visualization import ResultVisualizer
            visualizer = ResultVisualizer(
                pareto_df=pareto_df,
                baseline_dfs=baseline_dfs,
                config=config,
                output_dir=str(run_dir / 'figures')
            )
            visualizer.generate_all()
            logger.info("Visualizations saved to %s", run_dir / 'figures')
        except Exception as e:
            logger.warning("Visualization failed: %s", str(e))

    # Step 7: 生成报告
    logger.info("\nStep 7: Generating report...")
    total_elapsed = time.time() - total_start_time

    generate_report(
        config=config,
        pareto_df=pareto_df,
        baseline_dfs=baseline_dfs,
        metrics_results=metrics_results,
        run_dir=run_dir,
        elapsed_time=total_elapsed
    )

    # 保存摘要JSON
    summary = {
        'run_dir': str(run_dir),
        'seed': args.seed,
        'pareto_solutions': len(pareto_df),
        'baseline_feasible': sum(len(df[df['is_feasible']]) if 'is_feasible' in df.columns else len(df)
                                 for df in baseline_dfs.values()),
        'validation_passed': validation_passed,
        'total_time_seconds': total_elapsed,
        'min_cost': float(pareto_df['f1_total_cost_USD'].min()),
        'max_cost': float(pareto_df['f1_total_cost_USD'].max()),
    }

    with open(run_dir / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # 最终输出
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("Run Directory: %s", run_dir)
    logger.info("Pareto Solutions: %d", len(pareto_df))
    logger.info("Baseline Feasible: %d", summary['baseline_feasible'])
    logger.info("Validation: %s", "PASSED" if validation_passed else "WARNINGS")
    logger.info("Total Time: %.2fs", total_elapsed)
    logger.info("=" * 80)

    # 打印简洁摘要（用于脚本解析）
    print(f"\n[SUMMARY] run_dir={run_dir}, pareto={len(pareto_df)}, "
          f"baseline_feasible={summary['baseline_feasible']}, "
          f"validation={'PASSED' if validation_passed else 'WARNINGS'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""
RMTwin多目标优化框架
主入口点 - 修复版 v2

修复内容:
1. run_id输出目录化 - 每次运行独立目录，避免文件覆盖
2. 统一Pareto导出来源 - 确保同一评估器
3. 报告逻辑纠错 - 消除"有解却显示无解"的矛盾
4. Schema/口径断言 - 验证结果一致性
"""

import argparse
import logging
import time
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 导入核心模块
from config_manager import ConfigManager
from ontology_manager import OntologyManager
from optimization_core import RMTwinOptimizer
from baseline_methods import BaselineRunner
from visualization import Visualizer
from utils import setup_logging, save_results_summary, NumpyEncoder

logger = logging.getLogger(__name__)


def create_run_directory(base_dir: Path, seed: int) -> Path:
    """
    创建唯一的run输出目录
    格式: runs/YYYYMMDD_HHMMSS_seedXX/
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{timestamp}_seed{seed}"
    run_dir = base_dir / 'runs' / run_id
    
    # 如果目录已存在，添加序号
    if run_dir.exists():
        counter = 1
        while (base_dir / 'runs' / f"{run_id}_{counter}").exists():
            counter += 1
        run_dir = base_dir / 'runs' / f"{run_id}_{counter}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    
    return run_dir


def save_config_snapshot(config, run_dir: Path, seed: int):
    """保存配置快照到run目录"""
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'population_size': config.population_size,
        'n_generations': config.n_generations,
        'n_objectives': config.n_objectives,
        'budget_cap_usd': config.budget_cap_usd,
        'min_recall_threshold': config.min_recall_threshold,
        'max_latency_seconds': config.max_latency_seconds,
        'max_disruption_hours': config.max_disruption_hours,
        'max_carbon_emissions_kgCO2e_year': getattr(config, 'max_carbon_emissions_kgCO2e_year', 200000),
        'min_mtbf_hours': config.min_mtbf_hours,
        'road_network_length_km': config.road_network_length_km,
        'planning_horizon_years': config.planning_horizon_years,
        'n_random_samples': getattr(config, 'n_random_samples', 3000),
        'grid_resolution': getattr(config, 'grid_resolution', 5),
        'weight_combinations': getattr(config, 'weight_combinations', 100),
    }
    
    with open(run_dir / 'config_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    logger.info(f"Config snapshot saved to {run_dir / 'config_snapshot.json'}")


def validate_results_consistency(pareto_df: pd.DataFrame, 
                                  baseline_dfs: dict,
                                  config) -> dict:
    """
    验证结果一致性 (Schema/口径断言)
    
    Returns:
        dict: {'valid': bool, 'errors': list, 'warnings': list}
    """
    result = {'valid': True, 'errors': [], 'warnings': []}
    
    # 断言1: Pareto必需列检查
    required_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                    'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year',
                    'is_feasible']
    
    if len(pareto_df) > 0:
        missing_pareto = [c for c in required_cols if c not in pareto_df.columns]
        if missing_pareto:
            result['errors'].append(f"Pareto missing columns: {missing_pareto}")
            result['valid'] = False
    
    # 断言2: Baseline必需列检查
    for name, bdf in (baseline_dfs or {}).items():
        if bdf is not None and len(bdf) > 0:
            missing_baseline = [c for c in required_cols if c not in bdf.columns]
            if missing_baseline:
                result['errors'].append(f"Baseline {name} missing columns: {missing_baseline}")
                result['valid'] = False
    
    # 断言3: 口径一致性检查 - Pareto vs Baselines
    if len(pareto_df) > 0 and baseline_dfs:
        pareto_carbon_min = pareto_df['f5_carbon_emissions_kgCO2e_year'].min()
        pareto_disruption_unique = pareto_df['f4_traffic_disruption_hours'].nunique()
        
        for name, bdf in baseline_dfs.items():
            if bdf is not None and len(bdf) > 0:
                # 检查碳排放量纲是否一致 (差异不超过100倍)
                if 'f5_carbon_emissions_kgCO2e_year' in bdf.columns:
                    baseline_carbon_min = bdf['f5_carbon_emissions_kgCO2e_year'].min()
                    
                    if pareto_carbon_min > 0 and baseline_carbon_min > 0:
                        ratio = max(pareto_carbon_min, baseline_carbon_min) / min(pareto_carbon_min, baseline_carbon_min)
                        if ratio > 100:
                            result['warnings'].append(
                                f"Carbon scale mismatch: Pareto min={pareto_carbon_min:.0f}, "
                                f"{name} min={baseline_carbon_min:.0f} (ratio={ratio:.1f}x)"
                            )
                
                # 检查disruption分布一致性
                if 'f4_traffic_disruption_hours' in bdf.columns:
                    baseline_disruption_unique = bdf['f4_traffic_disruption_hours'].nunique()
                    
                    # 检查是否一个是离散(<=5档)另一个是连续(>20档)
                    if (pareto_disruption_unique <= 5 and baseline_disruption_unique > 20) or \
                       (pareto_disruption_unique > 20 and baseline_disruption_unique <= 5):
                        result['warnings'].append(
                            f"Disruption distribution mismatch: Pareto has {pareto_disruption_unique} unique values, "
                            f"{name} has {baseline_disruption_unique}"
                        )
    
    # 断言4: 目标值范围合理性
    if len(pareto_df) > 0:
        if (pareto_df['detection_recall'] < 0).any() or (pareto_df['detection_recall'] > 1).any():
            result['errors'].append("Recall values outside [0,1] range")
            result['valid'] = False
        
        if (pareto_df['f1_total_cost_USD'] < 0).any():
            result['errors'].append("Negative cost values found")
            result['valid'] = False
        
        if (pareto_df['f5_carbon_emissions_kgCO2e_year'] < 0).any():
            result['errors'].append("Negative carbon values found")
            result['valid'] = False
    
    return result


def main():
    """主执行函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description='RMTwin多目标优化')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='跳过NSGA-III优化')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='跳过基线方法')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='跳过可视化生成')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试日志')
    parser.add_argument('--legacy-output', action='store_true',
                       help='使用旧版输出目录结构 (不创建runs子目录)')
    args = parser.parse_args()
    
    # 设置
    start_time = time.time()
    config = ConfigManager(args.config)
    
    # 创建输出目录
    if args.legacy_output:
        # 兼容旧版: 直接使用config.output_dir
        run_dir = Path(config.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'figures').mkdir(exist_ok=True)
        log_dir = config.log_dir
    else:
        # 新版: 创建run_id子目录
        run_dir = create_run_directory(Path(config.output_dir), args.seed)
        log_dir = run_dir / 'logs'
    
    # 设置日志
    main_logger = setup_logging(debug=args.debug, log_dir=log_dir)
    
    main_logger.info("=" * 80)
    main_logger.info("RMTwin Multi-Objective Optimization Framework v2")
    main_logger.info(f"Run Directory: {run_dir}")
    main_logger.info(f"Seed: {args.seed}")
    main_logger.info(f"Config: {config.n_objectives} objectives, "
                    f"{config.n_generations} generations, "
                    f"{config.population_size} population")
    main_logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info("=" * 80)
    
    # 显示关键约束设置
    main_logger.info("\nConstraints:")
    main_logger.info(f"  Budget Cap: ${config.budget_cap_usd:,.0f}")
    main_logger.info(f"  Min Recall: {config.min_recall_threshold}")
    main_logger.info(f"  Max Latency: {config.max_latency_seconds} s")
    main_logger.info(f"  Min MTBF: {config.min_mtbf_hours} hours")
    main_logger.info(f"  Max Carbon: {getattr(config, 'max_carbon_emissions_kgCO2e_year', 200000)} kgCO2e/year")
    
    # 保存配置快照
    save_config_snapshot(config, run_dir, args.seed)
    
    # 初始化结果存储
    pareto_df = pd.DataFrame()
    baseline_dfs = {}
    optimization_history = None
    timings = {}
    
    try:
        # 步骤1：加载和填充本体
        main_logger.info("\nStep 1: Loading ontology...")
        ontology_manager = OntologyManager()
        ontology_graph = ontology_manager.populate_from_csv_files(
            config.sensor_csv,
            config.algorithm_csv,
            config.infrastructure_csv,
            config.cost_benefit_csv
        )
        
        # 保存填充的本体
        ontology_path = run_dir / 'populated_ontology.ttl'
        ontology_graph.serialize(destination=str(ontology_path), format='turtle')
        main_logger.info(f"Ontology saved to {ontology_path}")
        
        # 步骤2：运行NSGA-III优化
        if not args.skip_optimization:
            main_logger.info("\nStep 2: Running NSGA-III optimization...")
            optimizer = RMTwinOptimizer(ontology_graph, config, seed=args.seed)
            
            t0 = time.time()
            pareto_df, optimization_history = optimizer.optimize()
            timings['nsga3'] = time.time() - t0
            
            # 保存Pareto结果 (关键: 直接从本次优化结果保存)
            pareto_path = run_dir / 'pareto_solutions.csv'
            pareto_df.to_csv(pareto_path, index=False)
            main_logger.info(f"Saved {len(pareto_df)} Pareto solutions to {pareto_path}")
            
            # 同时保存到旧路径以兼容
            if not args.legacy_output:
                compat_path = run_dir / 'pareto_solutions_6obj_fixed.csv'
                pareto_df.to_csv(compat_path, index=False)
            
            if len(pareto_df) > 0:
                main_logger.info("\nPareto Front Summary:")
                main_logger.info(f"  Cost: ${pareto_df['f1_total_cost_USD'].min():,.0f} - "
                               f"${pareto_df['f1_total_cost_USD'].max():,.0f}")
                main_logger.info(f"  Recall: {pareto_df['detection_recall'].min():.4f} - "
                               f"{pareto_df['detection_recall'].max():.4f}")
                main_logger.info(f"  Carbon: {pareto_df['f5_carbon_emissions_kgCO2e_year'].min():,.0f} - "
                               f"{pareto_df['f5_carbon_emissions_kgCO2e_year'].max():,.0f} kgCO2e/year")
            else:
                main_logger.warning("No Pareto solutions found!")
            
            main_logger.info(f"Optimization time: {timings['nsga3']:.2f}s")
            
            # 保存history
            if optimization_history:
                history_path = run_dir / 'optimization_history.json'
                serializable_history = {
                    'n_evals': optimization_history.get('n_evals'),
                    'exec_time': optimization_history.get('exec_time'),
                    'n_gen': optimization_history.get('n_gen'),
                    'convergence': optimization_history.get('convergence', {})
                }
                with open(history_path, 'w') as f:
                    json.dump(serializable_history, f, indent=2, cls=NumpyEncoder)
        
        # 步骤3：运行基线方法
        if not args.skip_baselines:
            main_logger.info("\nStep 3: Running baseline methods...")
            
            # 关键: 使用同一个ontology_graph，确保评估器一致
            baseline_runner = BaselineRunner(ontology_graph, config, seed=args.seed)
            
            t0 = time.time()
            baseline_dfs = baseline_runner.run_all_methods()
            timings['baselines'] = time.time() - t0
            
            # 保存baseline结果
            for method_name, df in baseline_dfs.items():
                if df is not None and len(df) > 0:
                    baseline_path = run_dir / f'baseline_{method_name}.csv'
                    df.to_csv(baseline_path, index=False)
                    if 'is_feasible' in df.columns:
                        n_feasible = df['is_feasible'].sum()
                        main_logger.info(f"  {method_name}: {len(df)} total, {n_feasible} feasible")
                    else:
                        main_logger.info(f"  {method_name}: {len(df)} solutions")
                else:
                    main_logger.info(f"  {method_name}: No solutions")
            
            main_logger.info(f"Baseline time: {timings.get('baselines', 0):.2f}s")
        
        # 步骤4：验证结果一致性
        main_logger.info("\nStep 4: Validating results consistency...")
        validation = validate_results_consistency(pareto_df, baseline_dfs, config)
        
        if validation['errors']:
            for err in validation['errors']:
                main_logger.error(f"Validation Error: {err}")
        
        if validation['warnings']:
            for warn in validation['warnings']:
                main_logger.warning(f"Validation Warning: {warn}")
        
        if validation['valid']:
            main_logger.info("Validation: PASSED")
        else:
            main_logger.error("Validation: FAILED - Results may be inconsistent!")
        
        # 保存验证结果
        with open(run_dir / 'validation_result.json', 'w') as f:
            json.dump(validation, f, indent=2)
        
        # 步骤5：生成报告
        main_logger.info("\nStep 5: Generating report...")
        
        # 构建all_results结构（兼容save_results_summary）
        all_results = {}
        if len(pareto_df) > 0 or not args.skip_optimization:
            all_results['nsga3'] = {
                'dataframe': pareto_df,
                'time': timings.get('nsga3', 0),
                'history': optimization_history
            }
        
        if baseline_dfs:
            all_results['baselines'] = {
                'dataframes': baseline_dfs,
                'time': timings.get('baselines', 0)
            }
        
        # 临时修改config的output_dir指向run_dir
        original_output_dir = config.output_dir
        config.output_dir = run_dir
        
        save_results_summary(all_results, config)
        
        # 步骤6：生成可视化
        if not args.skip_visualization and len(pareto_df) > 0:
            main_logger.info("\nStep 6: Generating visualizations...")
            
            visualizer = Visualizer(config)
            visualizer.create_all_figures(
                pareto_results=pareto_df,
                baseline_results=baseline_dfs if baseline_dfs else None
            )
            
            if baseline_dfs:
                main_logger.info("Generating enhanced baseline comparison...")
                visualizer.create_enhanced_baseline_comparison(pareto_df, baseline_dfs)
            
            main_logger.info("Visualizations complete")
        elif len(pareto_df) == 0:
            main_logger.warning("No Pareto solutions for visualization")
        
        # 恢复config
        config.output_dir = original_output_dir
        
        # 最终摘要
        total_time = time.time() - start_time
        
        n_pareto = len(pareto_df)
        n_baseline_feasible = sum(
            df['is_feasible'].sum() if df is not None and 'is_feasible' in df.columns else 0
            for df in baseline_dfs.values()
        )
        
        main_logger.info("\n" + "=" * 80)
        main_logger.info("OPTIMIZATION COMPLETE")
        main_logger.info(f"Run Directory: {run_dir}")
        main_logger.info(f"Pareto Solutions: {n_pareto}")
        main_logger.info(f"Baseline Feasible: {n_baseline_feasible}")
        main_logger.info(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
        main_logger.info(f"Total Time: {total_time:.2f}s")
        main_logger.info("=" * 80)
        
        # 打印一行总结（便于自动化脚本解析）
        print(f"\n[SUMMARY] run_dir={run_dir}, pareto={n_pareto}, "
              f"baseline_feasible={n_baseline_feasible}, "
              f"validation={'PASSED' if validation['valid'] else 'FAILED'}")
        
        return 0 if validation['valid'] else 1
        
    except Exception as e:
        main_logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    
    finally:
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())

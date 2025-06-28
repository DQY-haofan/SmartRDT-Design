#!/usr/bin/env python3
"""
RMTwin多目标优化框架
主入口点 - 修复版
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# 导入核心模块
from config_manager import ConfigManager
from ontology_manager import OntologyManager
from optimization_core import RMTwinOptimizer
from baseline_methods import BaselineRunner
from visualization import Visualizer
from utils import setup_logging, save_results_summary


def main():
    """主执行函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description='RMTwin多目标优化')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='跳过NSGA-II优化')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='跳过基线方法')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='跳过可视化生成')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试日志')
    args = parser.parse_args()
    
    # 设置
    start_time = time.time()
    config = ConfigManager(args.config)
    logger = setup_logging(debug=args.debug, log_dir=config.log_dir)
    
    logger.info("="*80)
    logger.info("RMTwin多目标优化框架")
    logger.info(f"配置: {config.n_objectives} 个目标, "
               f"{config.n_generations} 代, "
               f"{config.population_size} 种群大小")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # 显示关键约束设置
    logger.info("\n关键约束设置:")
    logger.info(f"  预算上限: ${config.budget_cap_usd:,.0f}")
    logger.info(f"  最小召回率: {config.min_recall_threshold}")
    logger.info(f"  最大延迟: {config.max_latency_seconds} 秒")
    logger.info(f"  最小MTBF: {config.min_mtbf_hours} 小时")
    logger.info(f"  最大碳排放: {config.max_carbon_emissions_kgCO2e_year} kgCO2e/年")
    
    try:
        # 步骤1：加载和填充本体
        logger.info("\n步骤1：加载本体...")
        ontology_manager = OntologyManager()
        ontology_graph = ontology_manager.populate_from_csv_files(
            config.sensor_csv,
            config.algorithm_csv,
            config.infrastructure_csv,
            config.cost_benefit_csv
        )
        
        # 保存填充的本体
        ontology_path = config.output_dir / 'populated_ontology.ttl'
        ontology_graph.serialize(destination=str(ontology_path), format='turtle')
        logger.info(f"保存填充的本体到 {ontology_path}")
        
        # 初始化结果存储
        all_results = {}
        
        # 步骤2：运行NSGA-III优化
        if not args.skip_optimization:
            logger.info("\n步骤2：运行NSGA-III优化...")
            optimizer = RMTwinOptimizer(ontology_graph, config)
            
            optimization_start = time.time()
            pareto_results, optimization_history = optimizer.optimize()
            optimization_time = time.time() - optimization_start
            
            # 保存结果
            pareto_path = config.output_dir / 'pareto_solutions_6obj_fixed.csv'
            if len(pareto_results) > 0:
                pareto_results.to_csv(pareto_path, index=False)
                logger.info(f"找到 {len(pareto_results)} 个Pareto最优解")
                
                # 显示摘要统计
                logger.info("\nPareto前沿摘要:")
                logger.info(f"  成本范围: ${pareto_results['f1_total_cost_USD'].min():,.0f} - "
                           f"${pareto_results['f1_total_cost_USD'].max():,.0f}")
                logger.info(f"  召回率范围: {pareto_results['detection_recall'].min():.3f} - "
                           f"{pareto_results['detection_recall'].max():.3f}")
                logger.info(f"  碳排放范围: {pareto_results['f5_carbon_emissions_kgCO2e_year'].min():.0f} - "
                           f"{pareto_results['f5_carbon_emissions_kgCO2e_year'].max():.0f} kgCO2e/年")
            else:
                logger.warning("未找到Pareto最优解！")
            logger.info(f"优化时间: {optimization_time:.2f} 秒")
            
            all_results['nsga3'] = {
                'dataframe': pareto_results,
                'time': optimization_time,
                'history': optimization_history
            }
        
        # 步骤3：运行基线方法
        if not args.skip_baselines:
            logger.info("\n步骤3：运行基线方法...")
            baseline_runner = BaselineRunner(ontology_graph, config)
            
            baseline_start = time.time()
            baseline_results = baseline_runner.run_all_methods()
            baseline_time = time.time() - baseline_start
            
            # 保存基线结果
            for method_name, df in baseline_results.items():
                if df is not None and not df.empty:
                    baseline_path = config.output_dir / f'baseline_{method_name}.csv'
                    df.to_csv(baseline_path, index=False)
                    if 'is_feasible' in df.columns:
                        logger.info(f"{method_name}: {len(df)} 个解 "
                                  f"({df['is_feasible'].sum()} 个可行)")
                    else:
                        logger.info(f"{method_name}: {len(df)} 个解")
                else:
                    logger.info(f"{method_name}: 未生成解")
            
            all_results['baselines'] = {
                'dataframes': baseline_results,
                'time': baseline_time
            }
        
        # 步骤4：生成可视化
        if not args.skip_visualization:
            logger.info("\n步骤4：生成可视化...")
            visualizer = Visualizer(config)
            
            # 如果需要，加载结果
            if args.skip_optimization:
                pareto_path = config.output_dir / 'pareto_solutions_6obj_fixed.csv'
                if pareto_path.exists():
                    import pandas as pd
                    pareto_results = pd.read_csv(pareto_path)
                    all_results['nsga3'] = {'dataframe': pareto_results}
            
            if 'nsga3' in all_results and len(all_results['nsga3']['dataframe']) > 0:
                visualizer.create_all_figures(
                    pareto_results=all_results['nsga3']['dataframe'],
                    baseline_results=all_results.get('baselines', {}).get('dataframes')
                )
                logger.info("可视化完成")
                # 新增：增强的基线对比可视化
                if 'baselines' in all_results:
                    logger.info("生成增强的基线对比图表...")
                    visualizer.create_enhanced_baseline_comparison(
                        all_results['nsga3']['dataframe'],
                        all_results['baselines']['dataframes']
                    )
                
                logger.info("可视化完成")
            else:
                logger.warning("没有可用于可视化的解")
        
        # 步骤5：生成综合报告
        logger.info("\n步骤5：生成最终报告...")
        save_results_summary(all_results, config)
        
        # 最终摘要
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("优化完成")
        logger.info(f"总执行时间: {total_time:.2f} 秒")
        logger.info(f"结果保存到: {config.output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"致命错误: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理
        logging.shutdown()


if __name__ == "__main__":
    main()
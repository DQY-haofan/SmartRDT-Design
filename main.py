#!/usr/bin/env python3
"""
RMTwin Multi-Objective Optimization Framework
Main Entry Point - Fixed version
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Import core modules
from config_manager import ConfigManager
from ontology_manager import OntologyManager
from optimization_core import RMTwinOptimizer
from baseline_methods import BaselineRunner
from visualization import Visualizer
from utils import setup_logging, save_results_summary


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='RMTwin Multi-Objective Optimization')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Skip NSGA-II optimization')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='Skip baseline methods')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    args = parser.parse_args()
    
    # Setup
    start_time = time.time()
    config = ConfigManager(args.config)
    logger = setup_logging(debug=args.debug, log_dir=config.log_dir)
    
    logger.info("="*80)
    logger.info("RMTwin Multi-Objective Optimization Framework")
    logger.info(f"Configuration: {config.n_objectives} objectives, "
               f"{config.n_generations} generations, "
               f"{config.population_size} population")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        # Step 1: Load and populate ontology
        logger.info("\nStep 1: Loading ontology...")
        ontology_manager = OntologyManager()
        ontology_graph = ontology_manager.populate_from_csv_files(
            config.sensor_csv,
            config.algorithm_csv,
            config.infrastructure_csv,
            config.cost_benefit_csv
        )
        
        # Save populated ontology
        ontology_path = config.output_dir / 'populated_ontology.ttl'
        ontology_graph.serialize(destination=str(ontology_path), format='turtle')
        logger.info(f"Saved populated ontology to {ontology_path}")
        
        # Initialize results storage
        all_results = {}
        
        # Step 2: Run NSGA-II optimization
        if not args.skip_optimization:
            logger.info("\nStep 2: Running NSGA-II optimization...")
            optimizer = RMTwinOptimizer(ontology_graph, config)
            
            optimization_start = time.time()
            pareto_results, optimization_history = optimizer.optimize()
            optimization_time = time.time() - optimization_start
            
            # Save results
            pareto_path = config.output_dir / 'pareto_solutions_6d_enhanced.csv'
            if len(pareto_results) > 0:
                pareto_results.to_csv(pareto_path, index=False)
                logger.info(f"Found {len(pareto_results)} Pareto-optimal solutions")
            else:
                logger.warning("No Pareto-optimal solutions found!")
            logger.info(f"Optimization time: {optimization_time:.2f} seconds")
            
            all_results['nsga2'] = {
                'dataframe': pareto_results,
                'time': optimization_time,
                'history': optimization_history
            }
        
        # Step 3: Run baseline methods
        if not args.skip_baselines:
            logger.info("\nStep 3: Running baseline methods...")
            baseline_runner = BaselineRunner(ontology_graph, config)
            
            baseline_start = time.time()
            baseline_results = baseline_runner.run_all_methods()
            baseline_time = time.time() - baseline_start
            
            # Save baseline results
            for method_name, df in baseline_results.items():
                baseline_path = config.output_dir / f'baseline_{method_name}.csv'
                if len(df) > 0:
                    df.to_csv(baseline_path, index=False)
                    if 'is_feasible' in df.columns:
                        logger.info(f"{method_name}: {len(df)} solutions "
                                  f"({df['is_feasible'].sum()} feasible)")
                    else:
                        logger.info(f"{method_name}: {len(df)} solutions")
                else:
                    logger.info(f"{method_name}: No solutions generated")
            
            all_results['baselines'] = {
                'dataframes': baseline_results,
                'time': baseline_time
            }
        
        # Step 4: Generate visualizations
        if not args.skip_visualization:
            logger.info("\nStep 4: Generating visualizations...")
            visualizer = Visualizer(config)
            
            # Load results if needed
            if args.skip_optimization:
                pareto_path = config.output_dir / 'pareto_solutions_6d_enhanced.csv'
                if pareto_path.exists():
                    import pandas as pd
                    pareto_results = pd.read_csv(pareto_path)
                    all_results['nsga2'] = {'dataframe': pareto_results}
            
            if 'nsga2' in all_results and len(all_results['nsga2']['dataframe']) > 0:
                visualizer.create_all_figures(
                    pareto_results=all_results['nsga2']['dataframe'],
                    baseline_results=all_results.get('baselines', {}).get('dataframes')
                )
                logger.info("Visualizations completed")
            else:
                logger.warning("No solutions available for visualization")
        
        # Step 5: Generate comprehensive report
        logger.info("\nStep 5: Generating final report...")
        save_results_summary(all_results, config)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        logging.shutdown()


if __name__ == "__main__":
    main()
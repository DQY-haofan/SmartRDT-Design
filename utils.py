#!/usr/bin/env python3
"""
Utility Functions for RMTwin Optimization
Common helper functions and utilities
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def setup_logging(debug: bool = False, log_dir: Path = Path('./results/logs')) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'rmtwin_optimization_{timestamp}.log'
    
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console always INFO or higher
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Debug mode: {debug}")
    
    return logger


def save_results_summary(results: Dict, config) -> None:
    """Save comprehensive results summary"""
    logger = logging.getLogger(__name__)
    
    summary_path = config.output_dir / 'optimization_summary.json'
    report_path = config.output_dir / 'optimization_report.txt'
    
    # Prepare summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'road_network_km': config.road_network_length_km,
            'planning_years': config.planning_horizon_years,
            'budget_usd': config.budget_cap_usd,
            'objectives': config.n_objectives,
            'population_size': config.population_size,
            'generations': config.n_generations
        },
        'results': {}
    }
    
    # Process NSGA-II results if available
    if 'nsga2' in results:
        nsga2_df = results['nsga2']['dataframe']
        summary['results']['nsga2'] = {
            'total_solutions': len(nsga2_df),
            'computation_time': results['nsga2'].get('time', 0),
            'objectives': {
                'cost': {
                    'min': float(nsga2_df['f1_total_cost_USD'].min()),
                    'max': float(nsga2_df['f1_total_cost_USD'].max()),
                    'mean': float(nsga2_df['f1_total_cost_USD'].mean())
                },
                'recall': {
                    'min': float(nsga2_df['detection_recall'].min()),
                    'max': float(nsga2_df['detection_recall'].max()),
                    'mean': float(nsga2_df['detection_recall'].mean())
                },
                'carbon': {
                    'min': float(nsga2_df['f5_carbon_emissions_kgCO2e_year'].min()),
                    'max': float(nsga2_df['f5_carbon_emissions_kgCO2e_year'].max()),
                    'mean': float(nsga2_df['f5_carbon_emissions_kgCO2e_year'].mean())
                }
            }
        }
    
    # Process baseline results if available
    if 'baselines' in results:
        summary['results']['baselines'] = {}
        for method, df in results['baselines']['dataframes'].items():
            if len(df) > 0:
                feasible = df[df['is_feasible']]
                summary['results']['baselines'][method] = {
                    'total_solutions': len(df),
                    'feasible_solutions': len(feasible),
                    'best_cost': float(feasible['f1_total_cost_USD'].min()) if len(feasible) > 0 else None,
                    'best_recall': float(feasible['detection_recall'].max()) if len(feasible) > 0 else None
                }
    
    # Save JSON summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    # Generate text report
    report = generate_text_report(summary, config)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Saved optimization summary to {summary_path}")
    logger.info(f"Saved optimization report to {report_path}")


def generate_text_report(summary: Dict, config) -> str:
    """Generate human-readable text report"""
    report = []
    report.append("="*80)
    report.append("RMTWIN MULTI-OBJECTIVE OPTIMIZATION REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Generated: {summary['timestamp']}")
    report.append("")
    
    # Configuration section
    report.append("CONFIGURATION")
    report.append("-"*40)
    conf = summary['configuration']
    report.append(f"Road Network Length: {conf['road_network_km']} km")
    report.append(f"Planning Horizon: {conf['planning_years']} years")
    report.append(f"Budget Cap: ${conf['budget_usd']:,.0f}")
    report.append(f"Number of Objectives: {conf['objectives']}")
    report.append(f"Population Size: {conf['population_size']}")
    report.append(f"Generations: {conf['generations']}")
    report.append("")
    
    # Results section
    if summary['results']:
        report.append("OPTIMIZATION RESULTS")
        report.append("-"*40)
        
        # NSGA-II results
        if 'nsga2' in summary['results']:
            nsga2 = summary['results']['nsga2']
            report.append("NSGA-II Pareto Front:")
            report.append(f"  Total Solutions: {nsga2['total_solutions']}")
            report.append(f"  Computation Time: {nsga2['computation_time']:.2f} seconds")
            report.append("  Objective Ranges:")
            
            obj = nsga2['objectives']
            report.append(f"    Cost: ${obj['cost']['min']:,.0f} - ${obj['cost']['max']:,.0f} "
                         f"(avg: ${obj['cost']['mean']:,.0f})")
            report.append(f"    Recall: {obj['recall']['min']:.3f} - {obj['recall']['max']:.3f} "
                         f"(avg: {obj['recall']['mean']:.3f})")
            report.append(f"    Carbon: {obj['carbon']['min']:,.0f} - {obj['carbon']['max']:,.0f} "
                         f"kgCO2e/year (avg: {obj['carbon']['mean']:,.0f})")
            report.append("")
        
        # Baseline results
        if 'baselines' in summary['results']:
            report.append("Baseline Methods Comparison:")
            for method, data in summary['results']['baselines'].items():
                report.append(f"  {method.title()}:")
                report.append(f"    Total: {data['total_solutions']}, "
                             f"Feasible: {data['feasible_solutions']}")
                if data['best_cost']:
                    report.append(f"    Best Cost: ${data['best_cost']:,.0f}")
                    report.append(f"    Best Recall: {data['best_recall']:.3f}")
            report.append("")
    
    report.append("="*80)
    report.append("END OF REPORT")
    
    return '\n'.join(report)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


def calculate_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Calculate hypervolume indicator"""
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=ref_point)
        return ind(F)
    except ImportError:
        logging.warning("pymoo not available for hypervolume calculation")
        return 0.0


def calculate_metrics(F: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics for a Pareto front"""
    metrics = {}
    
    # Basic statistics
    metrics['n_solutions'] = len(F)
    
    # Spread metrics
    for i in range(F.shape[1]):
        metrics[f'obj{i+1}_range'] = float(F[:, i].max() - F[:, i].min())
        metrics[f'obj{i+1}_std'] = float(F[:, i].std())
    
    # Coverage metrics
    metrics['hypervolume'] = calculate_hypervolume(
        F, 
        ref_point=np.array([2e7, 0.3, 200, 200, 50000, 1e-3])
    )
    
    return metrics


def create_optimization_info_file(config, output_dir: Path) -> None:
    """Create detailed optimization info file for debugging"""
    info_path = output_dir / 'optimization_info.txt'
    
    with open(info_path, 'w') as f:
        f.write("RMTWIN OPTIMIZATION DETAILED INFORMATION\n")
        f.write("="*60 + "\n\n")
        
        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration details
        f.write("CONFIGURATION PARAMETERS\n")
        f.write("-"*30 + "\n")
        f.write(f"Road Network Length: {config.road_network_length_km} km\n")
        f.write(f"Planning Horizon: {config.planning_horizon_years} years\n")
        f.write(f"Budget Cap: ${config.budget_cap_usd:,.0f}\n")
        f.write(f"Daily Wage per Person: ${config.daily_wage_per_person}\n")
        f.write(f"Carbon Intensity: {config.carbon_intensity_factor} kgCO2e/kWh\n")
        f.write(f"Scenario Type: {config.scenario_type}\n")
        f.write("\n")
        
        # Constraints
        f.write("CONSTRAINTS\n")
        f.write("-"*30 + "\n")
        f.write(f"Min Recall Threshold: {config.min_recall_threshold}\n")
        f.write(f"Max Latency: {config.max_latency_seconds} seconds\n")
        f.write(f"Max Disruption: {config.max_disruption_hours} hours\n")
        f.write(f"Max Energy: {config.max_energy_kwh_year:,.0f} kWh/year\n")
        f.write(f"Min MTBF: {config.min_mtbf_hours:,.0f} hours\n")
        f.write("\n")
        
        # Optimization settings
        f.write("OPTIMIZATION SETTINGS\n")
        f.write("-"*30 + "\n")
        f.write(f"Algorithm: NSGA-II/III\n")
        f.write(f"Population Size: {config.population_size}\n")
        f.write(f"Generations: {config.n_generations}\n")
        f.write(f"Crossover Probability: {config.crossover_prob}\n")
        f.write(f"Crossover Distribution Index: {config.crossover_eta}\n")
        f.write(f"Mutation Distribution Index: {config.mutation_eta}\n")
        f.write(f"Parallel Processing: {config.use_parallel} ({config.n_processes} cores)\n")
        f.write("\n")
        
        # Baseline settings
        f.write("BASELINE METHOD SETTINGS\n")
        f.write("-"*30 + "\n")
        f.write(f"Random Search Samples: {config.n_random_samples}\n")
        f.write(f"Grid Search Resolution: {config.grid_resolution}\n")
        f.write(f"Weight Combinations: {config.weight_combinations}\n")
        f.write("\n")
        
        # File paths
        f.write("DATA FILES\n")
        f.write("-"*30 + "\n")
        f.write(f"Sensor CSV: {config.sensor_csv}\n")
        f.write(f"Algorithm CSV: {config.algorithm_csv}\n")
        f.write(f"Infrastructure CSV: {config.infrastructure_csv}\n")
        f.write(f"Cost-Benefit CSV: {config.cost_benefit_csv}\n")
        f.write("\n")
        
        # Advanced parameters
        f.write("ADVANCED PARAMETERS\n")
        f.write("-"*30 + "\n")
        f.write("Class Imbalance Penalties:\n")
        for algo, penalty in config.class_imbalance_penalties.items():
            f.write(f"  {algo}: {penalty}\n")
        f.write("\nNetwork Quality Factors:\n")
        for scenario, factors in config.network_quality_factors.items():
            f.write(f"  {scenario}:\n")
            for tech, factor in factors.items():
                f.write(f"    {tech}: {factor}\n")
        f.write("\nRedundancy Multipliers:\n")
        for deploy, mult in config.redundancy_multipliers.items():
            f.write(f"  {deploy}: {mult}\n")
        f.write("\n")
        
        # Decision variables
        f.write("DECISION VARIABLES\n")
        f.write("-"*30 + "\n")
        f.write("1. Sensor System Selection (discrete)\n")
        f.write("2. Data Acquisition Rate (continuous, 10-100 Hz)\n")
        f.write("3. Geometric LOD (discrete: Micro/Meso/Macro)\n")
        f.write("4. Condition LOD (discrete: Micro/Meso/Macro)\n")
        f.write("5. Detection Algorithm (discrete)\n")
        f.write("6. Detection Threshold (continuous, 0.1-0.9)\n")
        f.write("7. Storage Architecture (discrete)\n")
        f.write("8. Communication Scheme (discrete)\n")
        f.write("9. Inference Deployment (discrete)\n")
        f.write("10. Crew Size (integer, 1-10)\n")
        f.write("11. Inspection Cycle (integer, 1-365 days)\n")
        f.write("\n")
        
        # Objectives
        f.write("OPTIMIZATION OBJECTIVES\n")
        f.write("-"*30 + "\n")
        f.write("1. Minimize Total Cost (USD)\n")
        f.write("2. Minimize (1 - Detection Recall)\n")
        f.write("3. Minimize Data-to-Decision Latency (seconds)\n")
        f.write("4. Minimize Traffic Disruption (hours/year)\n")
        f.write("5. Minimize Carbon Emissions (kgCO2e/year)\n")
        f.write("6. Minimize (1 / System MTBF)\n")
        
    logging.info(f"Created optimization info file: {info_path}")


def compare_solutions(sol1: pd.Series, sol2: pd.Series) -> Dict[str, float]:
    """Compare two solutions across all objectives"""
    comparison = {}
    
    objectives = [
        ('Cost', 'f1_total_cost_USD', 'minimize'),
        ('Recall', 'detection_recall', 'maximize'),
        ('Latency', 'f3_latency_seconds', 'minimize'),
        ('Disruption', 'f4_traffic_disruption_hours', 'minimize'),
        ('Carbon', 'f5_carbon_emissions_kgCO2e_year', 'minimize'),
        ('MTBF', 'system_MTBF_hours', 'maximize')
    ]
    
    for name, col, direction in objectives:
        val1 = sol1[col]
        val2 = sol2[col]
        
        if direction == 'minimize':
            improvement = (val2 - val1) / val1 * 100  # Negative is better
        else:
            improvement = (val1 - val2) / val2 * 100  # Positive is better
        
        comparison[name] = {
            'sol1': val1,
            'sol2': val2,
            'improvement_pct': improvement,
            'better': 1 if improvement < 0 else 2 if improvement > 0 else 0
        }
    
    return comparison


def format_solution_summary(solution: pd.Series) -> str:
    """Format a solution for display"""
    summary = []
    summary.append(f"Solution ID: {solution.get('solution_id', 'N/A')}")
    summary.append(f"Configuration:")
    summary.append(f"  Sensor: {solution['sensor']}")
    summary.append(f"  Algorithm: {solution['algorithm']}")
    summary.append(f"  Deployment: {solution['deployment']}")
    summary.append(f"  Crew Size: {solution['crew_size']}")
    summary.append(f"  Inspection Cycle: {solution['inspection_cycle_days']} days")
    summary.append(f"Performance:")
    summary.append(f"  Cost: ${solution['f1_total_cost_USD']:,.0f}")
    summary.append(f"  Recall: {solution['detection_recall']:.3f}")
    summary.append(f"  Latency: {solution['f3_latency_seconds']:.1f}s")
    summary.append(f"  Carbon: {solution['f5_carbon_emissions_kgCO2e_year']:,.0f} kgCO2e/year")
    summary.append(f"  MTBF: {solution['system_MTBF_hours']:,.0f} hours")
    
    return '\n'.join(summary)


def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def validate_data_files(config) -> bool:
    """Validate that all required data files exist"""
    required_files = [
        config.sensor_csv,
        config.algorithm_csv,
        config.infrastructure_csv,
        config.cost_benefit_csv
    ]
    
    all_exist = True
    for filepath in required_files:
        if not Path(filepath).exists():
            logging.error(f"Required file not found: {filepath}")
            all_exist = False
        else:
            logging.info(f"Found data file: {filepath}")
    
    return all_exist


def create_experiment_id() -> str:
    """Create unique experiment ID"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_checkpoint(data: Dict, checkpoint_dir: Path, name: str):
    """Save optimization checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_checkpoint.pkl"
    
    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load optimization checkpoint"""
    if not checkpoint_path.exists():
        return None
    
    import pickle
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    logging.info(f"Loaded checkpoint: {checkpoint_path}")
    return data
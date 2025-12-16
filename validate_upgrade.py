#!/usr/bin/env python3
"""
Quick Validation Script for Step 2-Lite Upgrade
================================================
Tests that all 11 decision variables affect objectives.

Author: RMTwin Research Team
"""

import numpy as np
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_model_params():
    """Test model parameters module"""
    print("\n" + "="*60)
    print("TEST 1: Model Parameters")
    print("="*60)
    
    from model_params import MODEL_PARAMS, run_sanity_checks, get_param
    
    # Run sanity checks
    run_sanity_checks()
    
    # Test parameter access
    bw = get_param('comm_bandwidth_GBps', '5G')
    print(f"  5G bandwidth: {bw} GB/s")
    
    compute = get_param('deployment_compute_factor', 'Edge')
    print(f"  Edge compute factor: {compute}")
    
    print("  ✓ Model parameters OK")
    return True


def test_evaluator():
    """Test evaluator with variable sensitivity"""
    print("\n" + "="*60)
    print("TEST 2: Evaluator Variable Sensitivity")
    print("="*60)
    
    from config_manager import ConfigManager
    from ontology_manager import OntologyManager
    from evaluation import EnhancedFitnessEvaluatorV3, validate_model_consistency
    
    # Load config and ontology
    config = ConfigManager('config.json')
    ontology = OntologyManager()
    graph = ontology.populate_from_csv_files(
        'sensors_data.txt',
        'algorithms_data.txt', 
        'infrastructure_data.txt',
        'cost_benefit_data.txt'
    )
    
    # Initialize evaluator
    evaluator = EnhancedFitnessEvaluatorV3(graph, config)
    
    # Run consistency checks
    validate_model_consistency(evaluator)
    
    # Test variable sensitivity
    print("\nVariable Sensitivity Analysis:")
    base_x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.15])
    base_obj, base_con = evaluator._evaluate_single(base_x)
    
    variable_names = [
        'sensor', 'data_rate', 'geo_lod', 'cond_lod', 'algorithm',
        'detection_threshold', 'storage', 'communication', 'deployment',
        'crew_size', 'inspection_cycle'
    ]
    
    objective_names = ['Cost', 'Recall', 'Latency', 'Disruption', 'Carbon', 'Reliability']
    
    sensitivity = {}
    
    for i, var_name in enumerate(variable_names):
        # Test low and high values
        x_low = base_x.copy()
        x_low[i] = 0.1
        x_high = base_x.copy()
        x_high[i] = 0.9
        
        obj_low, _ = evaluator._evaluate_single(x_low)
        obj_high, _ = evaluator._evaluate_single(x_high)
        
        # Calculate relative change for each objective
        changes = []
        for j in range(6):
            if base_obj[j] != 0:
                change = abs(obj_high[j] - obj_low[j]) / abs(base_obj[j]) * 100
            else:
                change = abs(obj_high[j] - obj_low[j]) * 100
            changes.append(change)
        
        sensitivity[var_name] = changes
        
        # Find most affected objective
        max_idx = np.argmax(changes)
        print(f"  {var_name:20s}: max impact on {objective_names[max_idx]:12s} ({changes[max_idx]:6.1f}%)")
    
    # Check that all variables affect at least one objective
    all_active = True
    for var_name, changes in sensitivity.items():
        if max(changes) < 0.1:
            print(f"  WARNING: {var_name} has minimal impact!")
            all_active = False
    
    if all_active:
        print("\n  ✓ All 11 variables affect objectives")
    else:
        print("\n  ✗ Some variables have minimal impact")
    
    return all_active


def test_trace_function():
    """Test the explain/trace function"""
    print("\n" + "="*60)
    print("TEST 3: Trace/Explain Function")
    print("="*60)
    
    from config_manager import ConfigManager
    from ontology_manager import OntologyManager
    from evaluation import EnhancedFitnessEvaluatorV3
    
    config = ConfigManager('config.json')
    ontology = OntologyManager()
    graph = ontology.populate_from_csv_files(
        'sensors_data.txt',
        'algorithms_data.txt',
        'infrastructure_data.txt',
        'cost_benefit_data.txt'
    )
    
    evaluator = EnhancedFitnessEvaluatorV3(graph, config)
    
    # Decode a test solution
    x = np.array([0.88, 0.4, 0.5, 0.5, 0.65, 0.6, 0.0, 0.5, 0.85, 0.3, 0.12])
    config_decoded = evaluator.solution_mapper.decode_solution(x)
    
    # Get trace
    trace = evaluator.explain(config_decoded)
    
    print("\nTrace output:")
    print(f"  Sensor type: {trace['types']['sensor']}")
    print(f"  Algorithm type: {trace['types']['algorithm']}")
    print(f"  Raw data: {trace['data_volume']['raw_total_gb_year']:.1f} GB/year")
    print(f"  Sent data: {trace['data_volume']['sent_total_gb_year']:.1f} GB/year")
    print(f"  Data reduction: {trace['data_volume']['data_reduction_ratio']:.2f}")
    print(f"  Recall model z: {trace['recall_model']['sigmoid_input_z']:.3f}")
    print(f"  Final recall: {trace['recall_model']['final_recall']:.4f}")
    print(f"  Cost: ${trace['objectives']['f1_cost_USD']:,.0f}")
    print(f"  Latency: {trace['objectives']['f3_latency_s']:.1f} seconds")
    
    print("\n  ✓ Trace function works correctly")
    return True


def test_baseline_expert():
    """Test that Expert baseline produces feasible solutions"""
    print("\n" + "="*60)
    print("TEST 4: Expert Baseline Feasibility")
    print("="*60)
    
    from config_manager import ConfigManager
    from ontology_manager import OntologyManager
    from baseline_methods import ExpertHeuristicBaseline
    from evaluation import EnhancedFitnessEvaluatorV3
    
    config = ConfigManager('config.json')
    ontology = OntologyManager()
    graph = ontology.populate_from_csv_files(
        'sensors_data.txt',
        'algorithms_data.txt',
        'infrastructure_data.txt',
        'cost_benefit_data.txt'
    )
    
    evaluator = EnhancedFitnessEvaluatorV3(graph, config)
    
    # Run expert baseline
    expert = ExpertHeuristicBaseline(evaluator, config)
    df = expert.optimize()
    
    feasible_count = df['is_feasible'].sum()
    total_count = len(df)
    
    print(f"\n  Total configurations: {total_count}")
    print(f"  Feasible configurations: {feasible_count}")
    
    if feasible_count > 0:
        print(f"  Best cost: ${df[df['is_feasible']]['f1_total_cost_USD'].min():,.0f}")
        print(f"  Best recall: {df[df['is_feasible']]['detection_recall'].max():.4f}")
        print("\n  ✓ Expert baseline produces feasible solutions")
        return True
    else:
        print("\n  ✗ No feasible Expert solutions found")
        return False


def test_optimization_mini():
    """Quick optimization test (few generations)"""
    print("\n" + "="*60)
    print("TEST 5: Mini Optimization Run (5 generations)")
    print("="*60)
    
    from config_manager import ConfigManager
    from ontology_manager import OntologyManager
    from optimization_core import RMTwinOptimizer
    
    # Modify config for quick test
    config = ConfigManager('config.json')
    config.n_generations = 5
    config.population_size = 50
    
    ontology = OntologyManager()
    graph = ontology.populate_from_csv_files(
        'sensors_data.txt',
        'algorithms_data.txt',
        'infrastructure_data.txt',
        'cost_benefit_data.txt'
    )
    
    # Run optimization
    optimizer = RMTwinOptimizer(graph, config)
    pareto_df, history = optimizer.optimize()
    
    n_solutions = len(pareto_df)
    print(f"\n  Pareto solutions found: {n_solutions}")
    
    if n_solutions > 0:
        print(f"  Cost range: ${pareto_df['f1_total_cost_USD'].min():,.0f} - ${pareto_df['f1_total_cost_USD'].max():,.0f}")
        print(f"  Recall range: {pareto_df['detection_recall'].min():.4f} - {pareto_df['detection_recall'].max():.4f}")
        
        # Check trace columns exist
        if 'raw_data_gb_per_year' in pareto_df.columns:
            print("  ✓ Trace columns included in output")
        else:
            print("  ✗ Trace columns missing")
        
        print("\n  ✓ Mini optimization completed successfully")
        return True
    else:
        print("\n  ✗ No Pareto solutions found")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  STEP 2-LITE UPGRADE VALIDATION")
    print("="*60)
    
    results = {}
    
    try:
        results['model_params'] = test_model_params()
    except Exception as e:
        print(f"  ✗ Model params test failed: {e}")
        results['model_params'] = False
    
    try:
        results['evaluator'] = test_evaluator()
    except Exception as e:
        print(f"  ✗ Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        results['evaluator'] = False
    
    try:
        results['trace'] = test_trace_function()
    except Exception as e:
        print(f"  ✗ Trace test failed: {e}")
        results['trace'] = False
    
    try:
        results['expert'] = test_baseline_expert()
    except Exception as e:
        print(f"  ✗ Expert baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        results['expert'] = False
    
    try:
        results['optimization'] = test_optimization_mini()
    except Exception as e:
        print(f"  ✗ Optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        results['optimization'] = False
    
    # Summary
    print("\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("  ALL TESTS PASSED - Upgrade validated!")
        return 0
    else:
        print("  SOME TESTS FAILED - Check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

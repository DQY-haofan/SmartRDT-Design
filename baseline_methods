#!/usr/bin/env python3
"""
Baseline Methods for RMTwin Configuration
Complete implementation of all baseline approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import time
from abc import ABC, abstractmethod

# Import will be done in the classes to avoid circular import

logger = logging.getLogger(__name__)


class BaselineMethod(ABC):
    """Abstract base class for baseline methods"""
    
    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config
        self.results = []
        self.execution_time = 0
        
    @abstractmethod
    def optimize(self) -> pd.DataFrame:
        """Run optimization and return results"""
        pass
    
    def _create_result_entry(self, x: np.ndarray, objectives: np.ndarray, 
                           constraints: np.ndarray, solution_id: int) -> Dict:
        """Create standardized result entry"""
        config = self.evaluator.solution_mapper.decode_solution(x)
        is_feasible = np.all(constraints <= 0)
        
        return {
            'solution_id': solution_id,
            'method': self.__class__.__name__,
            
            # Configuration
            'sensor': config['sensor'].split('#')[-1],
            'algorithm': config['algorithm'].split('#')[-1],
            'data_rate_Hz': config['data_rate'],
            'geometric_LOD': config['geo_lod'],
            'condition_LOD': config['cond_lod'],
            'detection_threshold': config['detection_threshold'],
            'storage': config['storage'].split('#')[-1],
            'communication': config['communication'].split('#')[-1],
            'deployment': config['deployment'].split('#')[-1],
            'crew_size': config['crew_size'],
            'inspection_cycle_days': config['inspection_cycle'],
            
            # Objectives
            'f1_total_cost_USD': float(objectives[0]),
            'f2_one_minus_recall': float(objectives[1]),
            'f3_latency_seconds': float(objectives[2]),
            'f4_traffic_disruption_hours': float(objectives[3]),
            'f5_carbon_emissions_kgCO2e_year': float(objectives[4]),
            'f6_system_reliability_inverse_MTBF': float(objectives[5]),
            
            # Derived metrics
            'detection_recall': float(1 - objectives[1]),
            'system_MTBF_hours': float(1/objectives[5] if objectives[5] > 0 else 1e6),
            'system_MTBF_years': float(1/objectives[5]/8760 if objectives[5] > 0 else 100),
            'annual_cost_USD': float(objectives[0] / self.config.planning_horizon_years),
            'cost_per_km_year': float(objectives[0] / self.config.planning_horizon_years / 
                                    self.config.road_network_length_km),
            'carbon_footprint_tons_CO2_year': float(objectives[4] / 1000),
            
            # Feasibility
            'is_feasible': bool(is_feasible),
            'time_seconds': self.execution_time
        }


class RandomSearchBaseline(BaselineMethod):
    """Random search baseline"""
    
    def optimize(self) -> pd.DataFrame:
        """Generate random solutions"""
        n_samples = self.config.n_random_samples
        logger.info(f"Running Random Search with {n_samples} samples...")
        
        start_time = time.time()
        
        for i in range(n_samples):
            # Generate random solution
            x = np.random.rand(11)
            
            # Evaluate
            objectives, constraints = self.evaluator._evaluate_single(x)
            
            # Store result
            self.results.append(
                self._create_result_entry(x, objectives, constraints, i+1)
            )
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"  Evaluated {i + 1} random solutions...")
        
        self.execution_time = time.time() - start_time
        
        # Update time for all results
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Random Search completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Found {df['is_feasible'].sum()} feasible solutions")
        
        return df


class GridSearchBaseline(BaselineMethod):
    """Grid search on key variables"""
    
    def optimize(self) -> pd.DataFrame:
        """Grid search on discrete variables"""
        resolution = self.config.grid_resolution
        logger.info(f"Running Grid Search with resolution {resolution}...")
        
        start_time = time.time()
        
        # Select subsets for tractability
        n_sensors = min(resolution, len(self.evaluator.solution_mapper.sensors))
        n_algos = min(resolution, len(self.evaluator.solution_mapper.algorithms))
        
        sensors_subset = self.evaluator.solution_mapper.sensors[:n_sensors]
        algos_subset = self.evaluator.solution_mapper.algorithms[:n_algos]
        lod_options = ['Micro', 'Meso', 'Macro']
        deployment_options = self.evaluator.solution_mapper.deployments[:3]
        
        solution_id = 0
        
        for sensor_idx, sensor in enumerate(sensors_subset):
            for algo_idx, algo in enumerate(algos_subset):
                for geo_lod in lod_options:
                    for deployment_idx, deployment in enumerate(deployment_options):
                        solution_id += 1
                        
                        # Create solution vector
                        x = np.zeros(11)
                        x[0] = sensor_idx / max(n_sensors - 1, 1)
                        x[1] = 0.5  # Medium data rate
                        x[2] = lod_options.index(geo_lod) / 2
                        x[3] = x[2]  # Same LOD
                        x[4] = algo_idx / max(n_algos - 1, 1)
                        x[5] = 0.7  # High threshold
                        x[6] = 0.5  # Middle storage
                        x[7] = 0.5  # Middle comm
                        x[8] = deployment_idx / max(len(deployment_options) - 1, 1)
                        x[9] = 0.3  # 3-person crew
                        x[10] = 0.1  # ~36 day cycle
                        
                        # Evaluate
                        objectives, constraints = self.evaluator._evaluate_single(x)
                        
                        # Store result
                        self.results.append(
                            self._create_result_entry(x, objectives, constraints, solution_id)
                        )
        
        self.execution_time = time.time() - start_time
        
        # Update time
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Grid Search completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Evaluated {len(df)} grid points")
        logger.info(f"  Found {df['is_feasible'].sum()} feasible solutions")
        
        return df


class WeightedSumBaseline(BaselineMethod):
    """Single-objective weighted sum optimization"""
    
    def optimize(self) -> pd.DataFrame:
        """Optimize using different weight combinations"""
        n_weights = self.config.weight_combinations
        logger.info(f"Running Weighted Sum with {n_weights} weight sets...")
        
        start_time = time.time()
        
        # Generate weight combinations
        weight_sets = self._generate_weight_sets(n_weights)
        
        for weight_idx, weights in enumerate(weight_sets):
            # Find best solution for this weight set
            best_score = float('inf')
            best_solution = None
            best_objectives = None
            best_constraints = None
            
            # Random search with current weights
            for _ in range(100):
                x = np.random.rand(11)
                objectives, constraints = self.evaluator._evaluate_single(x)
                
                if np.all(constraints <= 0):  # Only consider feasible
                    # Normalize and compute weighted sum
                    norm_obj = self._normalize_objectives(objectives)
                    score = np.dot(weights, norm_obj)
                    
                    if score < best_score:
                        best_score = score
                        best_solution = x
                        best_objectives = objectives
                        best_constraints = constraints
            
            # Store best solution for this weight set
            if best_solution is not None:
                result = self._create_result_entry(
                    best_solution, best_objectives, best_constraints, weight_idx + 1
                )
                result['weights'] = weights.tolist()
                self.results.append(result)
            
            # Progress logging
            if (weight_idx + 1) % 10 == 0:
                logger.info(f"  Evaluated {weight_idx + 1} weight combinations...")
        
        self.execution_time = time.time() - start_time
        
        # Update time
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Weighted Sum completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Found {len(df)} solutions")
        
        return df
    
    def _generate_weight_sets(self, n_sets: int) -> np.ndarray:
        """Generate diverse weight combinations"""
        weights = []
        
        # Uniform weights
        weights.append(np.ones(6) / 6)
        
        # Single objective focus
        for i in range(6):
            w = np.zeros(6)
            w[i] = 1.0
            weights.append(w)
        
        # Balanced pairs
        for i in range(6):
            for j in range(i+1, 6):
                w = np.zeros(6)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
                if len(weights) >= n_sets:
                    break
            if len(weights) >= n_sets:
                break
        
        # Random weights for remaining
        while len(weights) < n_sets:
            w = np.random.rand(6)
            w = w / w.sum()  # Normalize
            weights.append(w)
        
        return np.array(weights[:n_sets])
    
    def _normalize_objectives(self, obj: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0,1]"""
        # Approximate ranges based on problem knowledge
        ranges = [
            (1e5, 2e7),      # Cost
            (0, 0.3),        # 1-Recall
            (0.1, 200),      # Latency
            (0, 200),        # Disruption
            (100, 50000),    # Carbon emissions
            (1e-6, 1e-3)     # 1/MTBF
        ]
        
        norm = np.zeros_like(obj)
        for i, (min_val, max_val) in enumerate(ranges):
            norm[i] = (obj[i] - min_val) / (max_val - min_val + 1e-10)
        
        return np.clip(norm, 0, 1)


class ExpertHeuristicBaseline(BaselineMethod):
    """Expert-defined configurations"""
    
    def optimize(self) -> pd.DataFrame:
        """Generate expert-recommended configurations"""
        logger.info("Running Expert Heuristic baseline...")
        
        start_time = time.time()
        
        # Define expert configurations
        expert_configs = [
            ("HighPerformance", self._high_performance_config()),
            ("LowCost", self._low_cost_config()),
            ("Balanced", self._balanced_config()),
            ("Sustainable", self._sustainable_config()),
            ("Reliable", self._reliable_config()),
            ("RealTime", self._real_time_config()),
            ("LongRange", self._long_range_config()),
            ("HighPrecision", self._high_precision_config()),
            ("MinimalDisruption", self._minimal_disruption_config()),
            ("FiberOptic", self._fiber_optic_config())
        ]
        
        for idx, (name, x) in enumerate(expert_configs):
            # Evaluate
            objectives, constraints = self.evaluator._evaluate_single(x)
            
            # Store result
            result = self._create_result_entry(x, objectives, constraints, idx + 1)
            result['method'] = f"ExpertHeuristic-{name}"
            self.results.append(result)
        
        self.execution_time = time.time() - start_time
        
        # Update time
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Expert Heuristic completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Generated {len(df)} expert configurations")
        logger.info(f"  {df['is_feasible'].sum()} are feasible")
        
        return df
    
    def _high_performance_config(self) -> np.ndarray:
        """Maximum detection performance"""
        return np.array([0.1, 0.9, 0, 0, 0.1, 0.8, 0.8, 0.9, 0.2, 0.5, 0.05])
    
    def _low_cost_config(self) -> np.ndarray:
        """Minimum cost configuration"""
        return np.array([0.9, 0.2, 0.9, 0.9, 0.9, 0.5, 0.2, 0.2, 0.8, 0.1, 0.5])
    
    def _balanced_config(self) -> np.ndarray:
        """Balanced across all objectives"""
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2])
    
    def _sustainable_config(self) -> np.ndarray:
        """Minimum environmental impact"""
        return np.array([0.7, 0.3, 0.7, 0.7, 0.6, 0.5, 0.3, 0.3, 0.9, 0.2, 0.3])
    
    def _reliable_config(self) -> np.ndarray:
        """Maximum system reliability"""
        return np.array([0.2, 0.4, 0.3, 0.3, 0.4, 0.6, 0.6, 0.7, 0.4, 0.4, 0.15])
    
    def _real_time_config(self) -> np.ndarray:
        """Minimum latency for real-time response"""
        return np.array([0.3, 0.8, 0.7, 0.7, 0.3, 0.7, 0.7, 0.8, 0.9, 0.3, 0.1])
    
    def _long_range_config(self) -> np.ndarray:
        """Maximum coverage efficiency"""
        return np.array([0.15, 0.7, 0.8, 0.8, 0.5, 0.6, 0.5, 0.6, 0.5, 0.2, 0.4])
    
    def _high_precision_config(self) -> np.ndarray:
        """Maximum measurement precision"""
        return np.array([0.05, 0.6, 0, 0, 0.2, 0.9, 0.7, 0.7, 0.3, 0.4, 0.08])
    
    def _minimal_disruption_config(self) -> np.ndarray:
        """Minimum traffic disruption"""
        return np.array([0.6, 0.4, 0.6, 0.6, 0.7, 0.5, 0.4, 0.4, 0.7, 0.15, 0.7])
    
    def _fiber_optic_config(self) -> np.ndarray:
        """Fiber optic sensor configuration"""
        # Find FOS sensor index
        fos_idx = 0.6  # Default
        for i, sensor in enumerate(self.evaluator.solution_mapper.sensors):
            if 'FOS' in sensor or 'Fiber' in sensor:
                fos_idx = i / len(self.evaluator.solution_mapper.sensors)
                break
        
        return np.array([fos_idx, 0.1, 0, 0, 0.3, 0.8, 0.7, 0.8, 0.5, 0.1, 0.9])


class BaselineRunner:
    """Orchestrates all baseline methods"""
    
    def __init__(self, ontology_graph, config):
        self.ontology_graph = ontology_graph
        self.config = config
        
        # Import here to avoid circular import
        from evaluation import EnhancedFitnessEvaluatorV2
        
        # Initialize shared evaluator
        self.evaluator = EnhancedFitnessEvaluatorV2(ontology_graph, config)
        
        # Initialize baseline methods
        self.methods = {
            'random': RandomSearchBaseline(self.evaluator, config),
            'grid': GridSearchBaseline(self.evaluator, config),
            'weighted': WeightedSumBaseline(self.evaluator, config),
            'expert': ExpertHeuristicBaseline(self.evaluator, config)
        }
    
    def run_all_methods(self) -> Dict[str, pd.DataFrame]:
        """Run all baseline methods"""
        results = {}
        
        for name, method in self.methods.items():
            logger.info(f"\nRunning baseline method: {name}")
            try:
                df = method.optimize()
                results[name] = df
                
                # Log summary
                logger.info(f"  Total solutions: {len(df)}")
                logger.info(f"  Feasible solutions: {df['is_feasible'].sum()}")
                if df['is_feasible'].sum() > 0:
                    feasible = df[df['is_feasible']]
                    logger.info(f"  Best cost: ${feasible['f1_total_cost_USD'].min():,.0f}")
                    logger.info(f"  Best recall: {feasible['detection_recall'].max():.3f}")
                    logger.info(f"  Best carbon: {feasible['f5_carbon_emissions_kgCO2e_year'].min():.0f} kgCO2e/year")
                
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                results[name] = pd.DataFrame()
        
        return results
    
    def run_method(self, method_name: str) -> pd.DataFrame:
        """Run a specific baseline method"""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}. "
                           f"Available: {list(self.methods.keys())}")
        
        return self.methods[method_name].optimize()
    
    def compare_with_pareto(self, pareto_df: pd.DataFrame, 
                           baseline_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare baseline results with Pareto front"""
        comparison = []
        
        # Analyze Pareto results
        comparison.append({
            'Method': 'NSGA-II',
            'Total_Solutions': len(pareto_df),
            'Feasible_Solutions': len(pareto_df),  # All Pareto solutions are feasible
            'Min_Cost': pareto_df['f1_total_cost_USD'].min(),
            'Max_Recall': pareto_df['detection_recall'].max(),
            'Min_Carbon': pareto_df['f5_carbon_emissions_kgCO2e_year'].min(),
            'Min_Latency': pareto_df['f3_latency_seconds'].min(),
            'Max_MTBF': pareto_df['system_MTBF_hours'].max()
        })
        
        # Analyze baseline results
        for method, df in baseline_results.items():
            if len(df) > 0:
                feasible = df[df['is_feasible']]
                comparison.append({
                    'Method': method.title(),
                    'Total_Solutions': len(df),
                    'Feasible_Solutions': len(feasible),
                    'Min_Cost': feasible['f1_total_cost_USD'].min() if len(feasible) > 0 else np.nan,
                    'Max_Recall': feasible['detection_recall'].max() if len(feasible) > 0 else np.nan,
                    'Min_Carbon': feasible['f5_carbon_emissions_kgCO2e_year'].min() if len(feasible) > 0 else np.nan,
                    'Min_Latency': feasible['f3_latency_seconds'].min() if len(feasible) > 0 else np.nan,
                    'Max_MTBF': feasible['system_MTBF_hours'].max() if len(feasible) > 0 else np.nan
                })
        
        return pd.DataFrame(comparison)
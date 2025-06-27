#!/usr/bin/env python3
"""
Core Optimization Module - NSGA-II/III Implementation
Fixed version with 6 objectives and proper attribute access
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

logger = logging.getLogger(__name__)


#!/usr/bin/env python3
"""
Core Optimization Module - NSGA-II/III Implementation
Fixed version with proper NSGA-III configuration
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

logger = logging.getLogger(__name__)


class RMTwinProblem(Problem):
    """Multi-objective optimization problem formulation with 6 objectives"""
    
    def __init__(self, evaluator, n_objectives=6):  # 改为6
        # Problem dimensions
        n_var = 11  # Decision variables
        n_constr = 5  # 5 constraints for 6 objectives
        
        # Variable bounds
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        
        super().__init__(
            n_var=n_var,
            n_obj=n_objectives,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )
        
        self.evaluator = evaluator
        self._eval_count = 0
        
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population"""
        objectives, constraints = self.evaluator.evaluate_batch(X)
        
        out["F"] = objectives
        out["G"] = constraints
        
        # Log progress
        self._eval_count += len(X)
        if self._eval_count % 1000 == 0:
            logger.info(f"Evaluated {self._eval_count} solutions...")
            
            # Log current best values
            best_indices = {
                'cost': objectives[:, 0].argmin(),
                'recall': objectives[:, 1].argmin(),
                'latency': objectives[:, 2].argmin(),
                'disruption': objectives[:, 3].argmin(),
                'carbon': objectives[:, 4].argmin() if objectives.shape[1] > 4 else None,
                'reliability': objectives[:, 5].argmin() if objectives.shape[1] > 5 else None
            }
            
            for name, idx in best_indices.items():
                if idx is not None:
                    logger.debug(f"Current best {name}: {objectives[idx]}")


class RMTwinOptimizer:
    """Main optimization orchestrator for 6-objective optimization"""
    
    def __init__(self, ontology_graph, config):
        self.ontology_graph = ontology_graph
        self.config = config
        
        # Import here to avoid circular import
        from evaluation import EnhancedFitnessEvaluatorV2
        
        # Initialize evaluator
        self.evaluator = EnhancedFitnessEvaluatorV2(ontology_graph, config)
        
        # Initialize problem
        self.problem = RMTwinProblem(self.evaluator, n_objectives=config.n_objectives)
        
        # Configure algorithm
        self.algorithm = self._configure_algorithm()
        
    def _configure_algorithm(self):
        """Configure NSGA-II/III algorithm based on number of objectives"""
        if self.config.n_objectives <= 3:
            # Use NSGA-II for few objectives
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=self.config.crossover_eta, 
                             prob=self.config.crossover_prob),
                mutation=PM(eta=self.config.mutation_eta, 
                           prob=1.0/self.problem.n_var),
                eliminate_duplicates=True
            )
        else:
            # Use NSGA-III for many objectives (4-6)
            from pymoo.util.ref_dirs import get_reference_directions
            
            # Adjust partitions to ensure ref_dirs <= pop_size
            if self.config.n_objectives == 4:
                # 8 partitions gives 165 ref points
                ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=8)
            elif self.config.n_objectives == 5:
                # 5 partitions gives 126 ref points
                ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=5)
            else:  # 6 objectives
                # 3 partitions gives 84 ref points
                ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=3)
            
            # Ensure population size is at least as large as ref_dirs
            pop_size = max(self.config.population_size, len(ref_dirs) + 50)
            
            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=self.config.crossover_eta, 
                             prob=self.config.crossover_prob),
                mutation=PM(eta=self.config.mutation_eta, 
                           prob=1.0/self.problem.n_var),
                eliminate_duplicates=True
            )
            
            logger.info(f"NSGA-III configured with {len(ref_dirs)} reference directions "
                       f"and population size {pop_size}")
        
        return algorithm
    
    def optimize(self) -> Tuple[pd.DataFrame, Dict]:
        """Run optimization and return results"""
        logger.info(f"Starting {self.algorithm.__class__.__name__} optimization with {self.config.n_objectives} objectives...")
        
        # Configure termination
        termination = get_termination("n_gen", self.config.n_generations)
        
        # Run optimization
        res = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=42,
            save_history=True,
            verbose=True
        )
        
        # Process results
        pareto_df = self._process_results(res)
        
        # Extract history (fixed n_eval access)
        history = {
            'n_evals': res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else self.problem._eval_count,
            'exec_time': res.exec_time if hasattr(res, 'exec_time') else 0,
            'n_gen': res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else self.config.n_generations,
            'history': res.history if hasattr(res, 'history') else None
        }
        
        return pareto_df, history
    
    def _process_results(self, res) -> pd.DataFrame:
        """Convert optimization results to DataFrame"""
        if res.X is None or (hasattr(res.X, '__len__') and len(res.X) == 0):
            logger.warning("No feasible solutions found! Check constraints.")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'solution_id', 'sensor', 'algorithm', 'data_rate_Hz',
                'geometric_LOD', 'condition_LOD', 'detection_threshold',
                'storage', 'communication', 'deployment', 'crew_size',
                'inspection_cycle_days', 'f1_total_cost_USD', 'f2_one_minus_recall',
                'f3_latency_seconds', 'f4_traffic_disruption_hours',
                'f5_carbon_emissions_kgCO2e_year', 'f6_system_reliability_inverse_MTBF',
                'detection_recall', 'system_MTBF_hours', 'system_MTBF_years',
                'annual_cost_USD', 'cost_per_km_year', 'carbon_footprint_tons_CO2_year',
                'is_feasible'
            ])
        
        # Ensure arrays are 2D
        X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
        F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
        
        results = []
        
        for i in range(len(X)):
            # Decode configuration
            config = self.evaluator.solution_mapper.decode_solution(X[i])
            
            # Get objectives
            objectives = F[i]
            
            # Create result dictionary based on number of objectives
            result = {
                'solution_id': i + 1,
                
                # Configuration details
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
                
                # Raw objectives (4 base objectives)
                'f1_total_cost_USD': float(objectives[0]),
                'f2_one_minus_recall': float(objectives[1]),
                'f3_latency_seconds': float(objectives[2]),
                'f4_traffic_disruption_hours': float(objectives[3]),
                
                # Derived metrics
                'detection_recall': float(1 - objectives[1]),
                'annual_cost_USD': float(objectives[0] / self.config.planning_horizon_years),
                'cost_per_km_year': float(objectives[0] / self.config.planning_horizon_years / 
                                         self.config.road_network_length_km),
            }
            
            # Add additional objectives if present
            if len(objectives) > 4:
                result['f5_carbon_emissions_kgCO2e_year'] = float(objectives[4])
                result['carbon_footprint_tons_CO2_year'] = float(objectives[4] / 1000)
            
            if len(objectives) > 5:
                result['f6_system_reliability_inverse_MTBF'] = float(objectives[5])
                result['system_MTBF_hours'] = float(1/objectives[5] if objectives[5] > 0 else 1e6)
                result['system_MTBF_years'] = float(1/objectives[5]/8760 if objectives[5] > 0 else 100)
            
            # All Pareto solutions are feasible
            result['is_feasible'] = True
            
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('f1_total_cost_USD')
            
            # Add rankings
            df['cost_rank'] = df['f1_total_cost_USD'].rank()
            df['recall_rank'] = df['detection_recall'].rank(ascending=False)
            
            if 'f5_carbon_emissions_kgCO2e_year' in df.columns:
                df['carbon_rank'] = df['f5_carbon_emissions_kgCO2e_year'].rank()
            
            if 'system_MTBF_hours' in df.columns:
                df['reliability_rank'] = df['system_MTBF_hours'].rank(ascending=False)
        
        logger.info(f"Processed {len(df)} Pareto-optimal solutions")
        
        return df
    

class OptimizationAnalyzer:
    """Analyze optimization results"""
    
    @staticmethod
    def calculate_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate hypervolume indicator"""
        try:
            from pymoo.indicators.hv import HV
            ind = HV(ref_point=ref_point)
            return ind(F)
        except ImportError:
            logger.warning("pymoo HV indicator not available")
            return 0.0
    
    @staticmethod
    def calculate_igd(F: np.ndarray, pareto_front: np.ndarray) -> float:
        """Calculate Inverted Generational Distance"""
        try:
            from pymoo.indicators.igd import IGD
            ind = IGD(pareto_front)
            return ind(F)
        except ImportError:
            logger.warning("pymoo IGD indicator not available")
            return 0.0
    
    @staticmethod
    def analyze_convergence(history) -> Dict:
        """Analyze convergence characteristics"""
        if not history or not hasattr(history, 'data'):
            return {}
        
        results = {
            'n_generations': len(history),
            'convergence_metrics': []
        }
        
        # Track metrics over generations
        for gen, algo in enumerate(history):
            if hasattr(algo, 'pop') and algo.pop is not None:
                F = algo.pop.get('F')
                if F is not None:
                    metrics = {
                        'generation': gen,
                        'n_solutions': len(F),
                        'best_cost': F[:, 0].min(),
                        'best_recall': 1 - F[:, 1].min(),
                        'avg_cost': F[:, 0].mean(),
                        'avg_recall': 1 - F[:, 1].mean()
                    }
                    
                    # Add 6-objective metrics if available
                    if F.shape[1] > 4:
                        metrics['best_carbon'] = F[:, 4].min()
                        metrics['avg_carbon'] = F[:, 4].mean()
                    
                    if F.shape[1] > 5:
                        metrics['best_reliability'] = 1/F[:, 5].max() if F[:, 5].max() > 0 else 0
                        metrics['avg_reliability'] = np.mean([1/x if x > 0 else 0 for x in F[:, 5]])
                    
                    results['convergence_metrics'].append(metrics)
        
        return results
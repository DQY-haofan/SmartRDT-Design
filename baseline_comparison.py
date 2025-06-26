#!/usr/bin/env python3
"""
Baseline Methods for RMTwin Configuration
Updated for 6-objective comparison with the enhanced framework

Provides simple baseline approaches:
1. Random Search
2. Grid Search
3. Single-Objective Weighted Sum
4. Expert Heuristic Rules
"""
from enhanced_evaluation_v2 import EnhancedFitnessEvaluatorV2, AdvancedEvaluationConfig
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Import from main framework
from rdflib import Graph
# Import from main framework
import sys
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline methods"""
    # Method parameters
    n_random_samples: int = 1000
    grid_resolution: int = 5  # Points per dimension for grid search
    weight_combinations: int = 50  # Number of weight combinations for weighted sum
    
    # Optimization parameters (matching main framework)
    road_network_length_km: float = 500.0  # Updated to match config.json
    planning_horizon_years: int = 10
    budget_cap_usd: float = 20_000_000  # Updated to match config.json
    min_recall_threshold: float = 0.70  # Updated to match config.json
    max_latency_seconds: float = 180.0  # Updated to match config.json
    
    # New constraints for 6 objectives
    max_energy_kwh_year: float = 50_000
    min_mtbf_hours: float = 5_000
    
    # Output
    output_dir: str = './results/baseline'

# ============================================================================
# BASELINE EVALUATOR (Updated for fiber optic sensors)
# ============================================================================

class BaselineEvaluator:
    """Simplified evaluator for baseline methods"""
    
    def __init__(self, ontology_graph: Graph, config: BaselineConfig, 
                    main_evaluator=None):  # 新增参数
            self.g = ontology_graph
            self.config = config
            self._cache_components()
            
            # 直接使用主框架的评估器
            if main_evaluator:
                self.main_evaluator = main_evaluator
            else:
                # 创建完整的配置
                adv_config = AdvancedEvaluationConfig(
                    road_network_length_km=config.road_network_length_km,
                    planning_horizon_years=config.planning_horizon_years,
                    budget_cap_usd=config.budget_cap_usd,
                    daily_wage_per_person=1500,  # 添加缺失的参数
                    fos_sensor_spacing_km=0.1,   # 添加FOS间距
                    depreciation_rate=0.1,        # 添加折旧率
                    scenario_type='urban',        # 添加场景类型
                    carbon_intensity_factor=0.417 # 添加碳强度
                )
                self.main_evaluator = EnhancedFitnessEvaluatorV2(ontology_graph, adv_config)
        
    def _cache_components(self):
        """Cache available components"""
        # Simplified component extraction
        self.sensors = []
        self.algorithms = []
        self.storage_systems = []
        self.comm_systems = []
        self.deployments = []
        
        # Extract from ontology (simplified)
        for s, p, o in self.g:
            subject_str = str(s)
            object_str = str(o)
            
            if subject_str.startswith('http://example.org/rmtwin#'):
                if 'Sensor' in object_str:
                    self.sensors.append(subject_str)
                elif 'Algorithm' in object_str:
                    self.algorithms.append(subject_str)
                elif 'StorageSystem' in object_str:
                    self.storage_systems.append(subject_str)
                elif 'CommunicationSystem' in object_str:
                    self.comm_systems.append(subject_str)
                elif 'Deployment' in object_str:
                    self.deployments.append(subject_str)
        
        # Remove duplicates and limit
        self.sensors = list(set(self.sensors))[:15]
        self.algorithms = list(set(self.algorithms))[:22]
        self.storage_systems = list(set(self.storage_systems))[:5]
        self.comm_systems = list(set(self.comm_systems))[:4]
        self.deployments = list(set(self.deployments))[:4]
        
        # Ensure we have at least some components
        if not self.sensors:
            self.sensors = ["http://example.org/rmtwin#Vehicle_LowCost_Sensors"]
        if not self.algorithms:
            self.algorithms = ["http://example.org/rmtwin#ML_SVM"]
        if not self.storage_systems:
            self.storage_systems = ["http://example.org/rmtwin#Storage_Cloud_AWS_S3"]
        if not self.comm_systems:
            self.comm_systems = ["http://example.org/rmtwin#Communication_5G_Network"]
        if not self.deployments:
            self.deployments = ["http://example.org/rmtwin#Deployment_Cloud_Computing"]
        
    def decode_solution(self, x: np.ndarray) -> Dict:
        """Decode solution vector"""
        return {
            'sensor': self.sensors[int(x[0] * len(self.sensors)) % len(self.sensors)],
            'data_rate': 10 + x[1] * 90,
            'geo_lod': ['Micro', 'Meso', 'Macro'][int(x[2] * 3) % 3],
            'cond_lod': ['Micro', 'Meso', 'Macro'][int(x[3] * 3) % 3],
            'algorithm': self.algorithms[int(x[4] * len(self.algorithms)) % len(self.algorithms)],
            'detection_threshold': 0.1 + x[5] * 0.8,
            'storage': self.storage_systems[int(x[6] * len(self.storage_systems)) % len(self.storage_systems)],
            'communication': self.comm_systems[int(x[7] * len(self.comm_systems)) % len(self.comm_systems)],
            'deployment': self.deployments[int(x[8] * len(self.deployments)) % len(self.deployments)],
            'crew_size': int(1 + x[9] * 9),
            'inspection_cycle': int(1 + x[10] * 364)
        }
    
    # 在 BaselineEvaluator 类中添加
    def evaluate_enhanced(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """使用增强版评估逻辑"""
        config = self.decode_solution(x)
        
        # 创建临时的高级配置
        adv_config = AdvancedEvaluationConfig(
            road_network_length_km=self.config.road_network_length_km,
            planning_horizon_years=self.config.planning_horizon_years,
            budget_cap_usd=self.config.budget_cap_usd
        )
        
        # 创建增强评估器
        evaluator = EnhancedFitnessEvaluatorV2(self.g, adv_config)
        
        # 评估
        objectives, constraints = evaluator.evaluate_solution(x, config)
        
        # 检查可行性
        is_feasible = np.all(constraints <= 0)
        
        return objectives, is_feasible

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """直接使用主评估器"""
        config = self.decode_solution(x)
        return self.main_evaluator.evaluate_solution(x, config)
    
    def _calculate_cost(self, config: Dict) -> float:
        """Simplified cost calculation"""
        # Base costs by component type
        sensor_costs = {
            'LiDAR': 500000, 'Camera': 100000, 'IoT': 50000, 'TLS': 200000,
            'FOS': 80000, 'Vehicle': 10000  # Added FOS cost
        }
        algo_costs = {
            'Deep': 50000, 'Machine': 20000, 'Traditional': 5000, 'PC': 10000
        }
        
        # Estimate based on component names
        sensor_cost = 200000  # default
        for key, cost in sensor_costs.items():
            if key.lower() in config['sensor'].lower():
                sensor_cost = cost
                break
                
        algo_cost = 20000  # default
        for key, cost in algo_costs.items():
            if key.lower() in config['algorithm'].lower():
                algo_cost = cost
                break
        
        # Operational costs
        # Check if stationary sensor (FOS)
        if 'FOS' in config['sensor']:
            # Minimal crew costs for stationary sensors
            crew_annual_cost = config['crew_size'] * 1000 * 10  # Only 10 days/year maintenance
        else:
            inspections_per_year = 365 / config['inspection_cycle']
            crew_daily_cost = config['crew_size'] * 1000 * 1.5  # skill multiplier
            days_per_inspection = self.config.road_network_length_km / 80  # assume 80 km/day
            crew_annual_cost = crew_daily_cost * days_per_inspection * inspections_per_year
        
        total_cost = sensor_cost + algo_cost + crew_annual_cost * self.config.planning_horizon_years
        
        return total_cost
    
    def _calculate_detection_performance(self, config: Dict) -> float:
        """Simplified performance calculation"""
        # Base recall by algorithm type
        if 'Deep' in config['algorithm'] or 'DL' in config['algorithm']:
            base_recall = 0.90
        elif 'Machine' in config['algorithm'] or 'ML' in config['algorithm']:
            base_recall = 0.80
        elif 'PC' in config['algorithm']:
            base_recall = 0.85
        else:
            base_recall = 0.65
            
        # Adjust by LOD
        if config['geo_lod'] == 'Micro':
            base_recall *= 1.1
        elif config['geo_lod'] == 'Macro':
            base_recall *= 0.9
            
        # Fiber optic sensors have high accuracy
        if 'FOS' in config['sensor']:
            base_recall *= 1.2
            
        return 1 - min(base_recall, 0.99)  # Return 1-recall
    
    def _calculate_latency(self, config: Dict) -> float:
        """Simplified latency calculation"""
        # Base latency by deployment
        if 'Edge' in config['deployment']:
            base_latency = 0.5
        elif 'Cloud' in config['deployment']:
            base_latency = 2.0
        else:
            base_latency = 1.0
            
        # Adjust by data rate
        latency = base_latency + (config['data_rate'] / 100) * 0.5
        
        # Fiber optic sensors have very low latency
        if 'FOS' in config['sensor']:
            latency *= 0.1
            
        return latency
    
    def _calculate_disruption(self, config: Dict) -> float:
        """Simplified disruption calculation"""
        # Fiber optic sensors cause minimal disruption (stationary)
        if 'FOS' in config['sensor']:
            return 0.1  # Minimal disruption
            
        inspections_per_year = 365 / config['inspection_cycle']
        hours_per_inspection = 4
        
        # Adjust by sensor type
        if 'MMS' in config['sensor'] or 'Mobile' in config['sensor']:
            hours_per_inspection *= 0.5  # Faster
        
        return inspections_per_year * hours_per_inspection
    
    def _calculate_environmental_impact(self, config: Dict) -> float:
        """Simplified environmental impact (kWh/year)"""
        # Base consumption by sensor type
        sensor_energy = {
            'LiDAR': 200, 'Camera': 50, 'IoT': 10, 'TLS': 100,
            'FOS': 1, 'Vehicle': 20  # FOS very low power
        }
        
        # Estimate sensor power
        sensor_power = 100  # default
        for key, power in sensor_energy.items():
            if key.lower() in config['sensor'].lower():
                sensor_power = power
                break
        
        # Backend power (simplified)
        if 'Cloud' in config['deployment']:
            backend_power = 500
        else:
            backend_power = 300
            
        # Annual consumption
        if 'FOS' in config['sensor']:
            # Continuous monitoring
            sensor_hours = 365 * 24
            vehicle_kwh = 0  # No vehicle needed
        else:
            sensor_hours = (365 / config['inspection_cycle']) * 8 * 10
            vehicle_kwh = (self.config.road_network_length_km * 365 / config['inspection_cycle']) * 0.8
            
        backend_hours = 365 * 24
        
        total_kwh = (sensor_power * sensor_hours + backend_power * backend_hours) / 1000 + vehicle_kwh
        
        return total_kwh
    
    def _calculate_reliability(self, config: Dict) -> float:
        """Simplified reliability (1/MTBF)"""
        # Base MTBF by component type
        if 'FOS' in config['sensor']:
            sensor_mtbf = 100000  # Very high for fiber optic
        elif 'LiDAR' in config['sensor']:
            sensor_mtbf = 20000
        elif 'Camera' in config['sensor']:
            sensor_mtbf = 10000
        else:
            sensor_mtbf = 50000
            
        # System reliability (simplified)
        system_mtbf = sensor_mtbf * 0.8  # Account for other components
        
        return 1 / system_mtbf if system_mtbf > 0 else 1.0

# ============================================================================
# BASELINE METHODS (keep the same)
# ============================================================================

class RandomSearchBaseline:
    """Random search baseline"""
    
    def __init__(self, evaluator: BaselineEvaluator, config: BaselineConfig):
        self.evaluator = evaluator
        self.config = config
        
    def optimize(self) -> pd.DataFrame:
        """Run random search"""
        logger.info(f"Running Random Search with {self.config.n_random_samples} samples...")
        
        results = []
        start_time = time.time()
        
        for i in range(self.config.n_random_samples):
            # Generate random solution
            x = np.random.rand(11)
            
            # Evaluate
            objectives, is_feasible = self.evaluator.evaluate_enhanced(x)
            
            # Decode for storage
            config = self.evaluator.decode_solution(x)
            
            result = {
                'method': 'RandomSearch',
                'solution_id': i + 1,
                'sensor': config['sensor'].split('#')[-1],
                'algorithm': config['algorithm'].split('#')[-1],
                'f1_total_cost_USD': objectives[0],
                'f2_one_minus_recall': objectives[1],
                'f3_latency_seconds': objectives[2],
                'f4_traffic_disruption_hours': objectives[3],
                'f5_environmental_impact_kWh_year': objectives[4],
                'f6_system_reliability_inverse_MTBF': objectives[5],
                'detection_recall': 1 - objectives[1],
                'system_MTBF_hours': 1/objectives[5] if objectives[5] > 0 else float('inf'),
                'is_feasible': is_feasible,
                'time_seconds': time.time() - start_time
            }
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Evaluated {i + 1} random solutions...")
        
        df = pd.DataFrame(results)
        logger.info(f"Random Search completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"  Found {df['is_feasible'].sum()} feasible solutions")
        
        return df

class GridSearchBaseline:
    """Grid search baseline"""
    
    def __init__(self, evaluator: BaselineEvaluator, config: BaselineConfig):
        self.evaluator = evaluator
        self.config = config
        
    def optimize(self) -> pd.DataFrame:
        """Run grid search (simplified on subset of variables)"""
        logger.info(f"Running Grid Search with resolution {self.config.grid_resolution}...")
        
        # For tractability, only grid search on key discrete variables
        sensors_subset = self.evaluator.sensors[:3]
        algos_subset = self.evaluator.algorithms[:3]
        lod_options = ['Micro', 'Meso', 'Macro']
        
        results = []
        start_time = time.time()
        solution_id = 0
        
        for sensor_idx, sensor in enumerate(sensors_subset):
            for algo_idx, algo in enumerate(algos_subset):
                for geo_lod in lod_options:
                    for deployment_idx in range(min(3, len(self.evaluator.deployments))):
                        # Create solution vector
                        x = np.zeros(11)
                        x[0] = sensor_idx / max(len(sensors_subset) - 1, 1)
                        x[2] = lod_options.index(geo_lod) / 2
                        x[3] = x[2]  # Same LOD for simplicity
                        x[4] = algo_idx / max(len(algos_subset) - 1, 1)
                        x[5] = 0.5  # Fixed threshold
                        x[6] = 0.5  # Middle storage option
                        x[7] = 0.5  # Middle comm option
                        x[8] = deployment_idx / max(2, 1)
                        x[9] = 0.3  # 3-person crew
                        x[10] = 0.1  # ~36 day cycle
                        x[1] = 0.5  # Middle data rate
                        
                        # Evaluate
                        objectives, is_feasible = self.evaluator.evaluate_enhanced(x)
                        config = self.evaluator.decode_solution(x)
                        
                        solution_id += 1
                        result = {
                            'method': 'GridSearch',
                            'solution_id': solution_id,
                            'sensor': config['sensor'].split('#')[-1],
                            'algorithm': config['algorithm'].split('#')[-1],
                            'f1_total_cost_USD': objectives[0],
                            'f2_one_minus_recall': objectives[1],
                            'f3_latency_seconds': objectives[2],
                            'f4_traffic_disruption_hours': objectives[3],
                            'f5_environmental_impact_kWh_year': objectives[4],
                            'f6_system_reliability_inverse_MTBF': objectives[5],
                            'detection_recall': 1 - objectives[1],
                            'system_MTBF_hours': 1/objectives[5] if objectives[5] > 0 else float('inf'),
                            'is_feasible': is_feasible,
                            'time_seconds': time.time() - start_time
                        }
                        results.append(result)
        
        df = pd.DataFrame(results)
        logger.info(f"Grid Search completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"  Evaluated {len(df)} grid points")
        logger.info(f"  Found {df['is_feasible'].sum()} feasible solutions")
        
        return df

class WeightedSumBaseline:
    """Weighted sum single-objective baseline"""
    
    def __init__(self, evaluator: BaselineEvaluator, config: BaselineConfig):
        self.evaluator = evaluator
        self.config = config
        
    def optimize(self) -> pd.DataFrame:
        """Run weighted sum optimization"""
        logger.info(f"Running Weighted Sum with {self.config.weight_combinations} weight sets...")
        
        results = []
        start_time = time.time()
        
        # Generate weight combinations
        weight_sets = self._generate_weight_sets()
        
        for weight_idx, weights in enumerate(weight_sets):
            # Random search with this weight set
            best_score = float('inf')
            best_solution = None
            best_objectives = None
            
            for _ in range(100):  # 100 random samples per weight set
                x = np.random.rand(11)
                objectives, is_feasible = self.evaluator.evaluate_enhanced(x)
                
                if is_feasible:
                    # Normalize objectives
                    norm_obj = self._normalize_objectives(objectives)
                    
                    # Calculate weighted sum
                    score = np.dot(weights, norm_obj)
                    
                    if score < best_score:
                        best_score = score
                        best_solution = x
                        best_objectives = objectives
            
            if best_solution is not None:
                config = self.evaluator.decode_solution(best_solution)
                
                result = {
                    'method': 'WeightedSum',
                    'solution_id': weight_idx + 1,
                    'weights': weights.tolist(),
                    'sensor': config['sensor'].split('#')[-1],
                    'algorithm': config['algorithm'].split('#')[-1],
                    'f1_total_cost_USD': best_objectives[0],
                    'f2_one_minus_recall': best_objectives[1],
                    'f3_latency_seconds': best_objectives[2],
                    'f4_traffic_disruption_hours': best_objectives[3],
                    'f5_environmental_impact_kWh_year': best_objectives[4],
                    'f6_system_reliability_inverse_MTBF': best_objectives[5],
                    'detection_recall': 1 - best_objectives[1],
                    'system_MTBF_hours': 1/best_objectives[5] if best_objectives[5] > 0 else float('inf'),
                    'is_feasible': True,
                    'time_seconds': time.time() - start_time
                }
                results.append(result)
            
            if (weight_idx + 1) % 10 == 0:
                logger.info(f"  Evaluated {weight_idx + 1} weight combinations...")
        
        df = pd.DataFrame(results)
        logger.info(f"Weighted Sum completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"  Found {len(df)} solutions")
        
        return df
    
    def _generate_weight_sets(self) -> np.ndarray:
        """Generate diverse weight combinations"""
        weight_sets = []
        
        # Uniform weights
        weight_sets.append(np.ones(6) / 6)
        
        # Single objective focus
        for i in range(6):
            w = np.zeros(6)
            w[i] = 1.0
            weight_sets.append(w)
        
        # Random weights
        for _ in range(self.config.weight_combinations - 7):
            w = np.random.rand(6)
            w = w / w.sum()  # Normalize
            weight_sets.append(w)
        
        return np.array(weight_sets)
    
    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0, 1]"""
        # Simple min-max normalization with assumed ranges
        ranges = [
            (100000, 20000000),   # Cost (updated range)
            (0, 0.3),             # 1-Recall
            (0.1, 200),           # Latency
            (0, 200),             # Disruption
            (1000, 100000),       # Energy
            (0, 0.001)            # 1/MTBF
        ]
        
        norm_obj = np.zeros_like(objectives)
        for i, (min_val, max_val) in enumerate(ranges):
            norm_obj[i] = (objectives[i] - min_val) / (max_val - min_val)
            norm_obj[i] = np.clip(norm_obj[i], 0, 1)
        
        return norm_obj

class ExpertHeuristicBaseline:
    """Expert heuristic rules baseline"""
    
    def __init__(self, evaluator: BaselineEvaluator, config: BaselineConfig):
        self.evaluator = evaluator
        self.config = config
        
    def optimize(self) -> pd.DataFrame:
        """Generate solutions based on expert rules"""
        logger.info("Running Expert Heuristic baseline...")
        
        results = []
        start_time = time.time()
        
        # Expert Rule 1: High-performance configuration
        x1 = self._create_high_performance_config()
        obj1, feas1 = self.evaluator.evaluate_enhanced(x1)
        results.append(self._create_result('HighPerformance', 1, x1, obj1, feas1, start_time))
        
        # Expert Rule 2: Low-cost configuration
        x2 = self._create_low_cost_config()
        obj2, feas2 = self.evaluator.evaluate_enhanced(x2)
        results.append(self._create_result('LowCost', 2, x2, obj2, feas2, start_time))
        
        # Expert Rule 3: Balanced configuration
        x3 = self._create_balanced_config()
        obj3, feas3 = self.evaluator.evaluate_enhanced(x3)
        results.append(self._create_result('Balanced', 3, x3, obj3, feas3, start_time))
        
        # Expert Rule 4: Sustainable configuration
        x4 = self._create_sustainable_config()
        obj4, feas4 = self.evaluator.evaluate_enhanced(x4)
        results.append(self._create_result('Sustainable', 4, x4, obj4, feas4, start_time))
        
        # Expert Rule 5: Reliable configuration
        x5 = self._create_reliable_config()
        obj5, feas5 = self.evaluator.evaluate_enhanced(x5)
        results.append(self._create_result('Reliable', 5, x5, obj5, feas5, start_time))
        
        # Expert Rule 6: Fiber Optic configuration (new)
        x6 = self._create_fiber_optic_config()
        obj6, feas6 = self.evaluator.evaluate_enhanced(x6)
        results.append(self._create_result('FiberOptic', 6, x6, obj6, feas6, start_time))
        
        df = pd.DataFrame(results)
        logger.info(f"Expert Heuristic completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"  Generated {len(df)} expert configurations")
        
        return df
    
    def _create_high_performance_config(self) -> np.ndarray:
        """Create high-performance configuration"""
        x = np.zeros(11)
        x[0] = 0.1  # High-end sensor (LiDAR)
        x[1] = 0.9  # High data rate
        x[2] = 0.0  # Micro LOD
        x[3] = 0.0  # Micro LOD
        x[4] = 0.1  # Deep learning algorithm
        x[5] = 0.7  # High threshold
        x[6] = 0.8  # High-end storage
        x[7] = 0.9  # Fiber communication
        x[8] = 0.2  # Cloud deployment
        x[9] = 0.5  # Medium crew
        x[10] = 0.05  # Frequent inspection
        return x
    
    def _create_low_cost_config(self) -> np.ndarray:
        """Create low-cost configuration"""
        x = np.zeros(11)
        x[0] = 0.9  # Low-cost sensor (Vehicle)
        x[1] = 0.2  # Low data rate
        x[2] = 0.9  # Macro LOD
        x[3] = 0.9  # Macro LOD
        x[4] = 0.9  # Traditional algorithm
        x[5] = 0.5  # Medium threshold
        x[6] = 0.2  # Low-cost storage
        x[7] = 0.2  # LoRaWAN
        x[8] = 0.8  # Edge deployment
        x[9] = 0.1  # Small crew
        x[10] = 0.5  # Infrequent inspection
        return x
    
    def _create_balanced_config(self) -> np.ndarray:
        """Create balanced configuration"""
        x = np.zeros(11)
        x[0] = 0.5  # Mid-range sensor
        x[1] = 0.5  # Medium data rate
        x[2] = 0.5  # Meso LOD
        x[3] = 0.5  # Meso LOD
        x[4] = 0.5  # ML algorithm
        x[5] = 0.5  # Medium threshold
        x[6] = 0.5  # Hybrid storage
        x[7] = 0.5  # 4G/5G
        x[8] = 0.5  # Hybrid deployment
        x[9] = 0.3  # Medium crew
        x[10] = 0.2  # Regular inspection
        return x
    
    def _create_sustainable_config(self) -> np.ndarray:
        """Create sustainable configuration"""
        x = np.zeros(11)
        x[0] = 0.7  # Low-power sensor
        x[1] = 0.3  # Low data rate
        x[2] = 0.7  # Coarse LOD
        x[3] = 0.7  # Coarse LOD
        x[4] = 0.6  # Efficient algorithm
        x[5] = 0.5  # Medium threshold
        x[6] = 0.3  # Efficient storage
        x[7] = 0.3  # Low-power comm
        x[8] = 0.9  # Edge deployment
        x[9] = 0.2  # Small crew
        x[10] = 0.3  # Moderate inspection
        return x
    
    def _create_reliable_config(self) -> np.ndarray:
        """Create reliable configuration"""
        x = np.zeros(11)
        x[0] = 0.2  # Industrial sensor
        x[1] = 0.4  # Moderate data rate
        x[2] = 0.3  # Fine LOD
        x[3] = 0.3  # Fine LOD
        x[4] = 0.4  # Proven algorithm
        x[5] = 0.6  # Conservative threshold
        x[6] = 0.6  # Redundant storage
        x[7] = 0.7  # Reliable comm
        x[8] = 0.4  # On-premise
        x[9] = 0.4  # Adequate crew
        x[10] = 0.15  # Regular inspection
        return x
    
    def _create_fiber_optic_config(self) -> np.ndarray:
        """Create fiber optic configuration"""
        x = np.zeros(11)
        # Find FOS sensor index
        fos_idx = None
        for i, sensor in enumerate(self.evaluator.sensors):
            if 'FOS' in sensor:
                fos_idx = i / len(self.evaluator.sensors)
                break
        
        x[0] = fos_idx if fos_idx is not None else 0.6  # FOS sensor
        x[1] = 0.1  # Low data rate (continuous monitoring)
        x[2] = 0.0  # Micro LOD (high precision)
        x[3] = 0.0  # Micro LOD
        x[4] = 0.3  # Efficient algorithm
        x[5] = 0.8  # High threshold (low false positives)
        x[6] = 0.7  # Good storage
        x[7] = 0.8  # Fiber communication
        x[8] = 0.5  # Hybrid deployment
        x[9] = 0.1  # Minimal crew (stationary)
        x[10] = 0.9  # Continuous monitoring
        return x
    
    def _create_result(self, rule_name: str, sol_id: int, x: np.ndarray, 
                       objectives: np.ndarray, is_feasible: bool, 
                       start_time: float) -> Dict:
        """Create result dictionary"""
        config = self.evaluator.decode_solution(x)
        
        return {
            'method': f'ExpertHeuristic-{rule_name}',
            'solution_id': sol_id,
            'sensor': config['sensor'].split('#')[-1],
            'algorithm': config['algorithm'].split('#')[-1],
            'f1_total_cost_USD': objectives[0],
            'f2_one_minus_recall': objectives[1],
            'f3_latency_seconds': objectives[2],
            'f4_traffic_disruption_hours': objectives[3],
            'f5_environmental_impact_kWh_year': objectives[4],
            'f6_system_reliability_inverse_MTBF': objectives[5],
            'detection_recall': 1 - objectives[1],
            'system_MTBF_hours': 1/objectives[5] if objectives[5] > 0 else float('inf'),
            'is_feasible': is_feasible,
            'time_seconds': time.time() - start_time
        }

# ============================================================================
# COMPARISON AND VISUALIZATION (keep most of it)
# ============================================================================

def compare_baseline_methods(baseline_results: Dict[str, pd.DataFrame], 
                           pareto_results: pd.DataFrame,
                           output_dir: str = './results/baseline'):
    """Compare baseline methods with Pareto-optimal results"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Combine all results
    all_baseline = pd.concat([df for df in baseline_results.values()], 
                            ignore_index=True)
    
    # Create comparison visualizations
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Method comparison - feasible solutions
    ax1 = plt.subplot(3, 3, 1)
    method_feasible = all_baseline.groupby('method')['is_feasible'].agg(['sum', 'count'])
    method_feasible['percentage'] = method_feasible['sum'] / method_feasible['count'] * 100
    
    method_feasible['percentage'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_ylabel('Feasible Solutions (%)', fontsize=12)
    ax1.set_title('Feasibility Rate by Method', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Objective space coverage - Cost vs Recall
    ax2 = plt.subplot(3, 3, 2)
    
    # Plot Pareto front
    ax2.scatter(pareto_results['f1_total_cost_USD']/1000, 
               pareto_results['detection_recall'],
               c='red', s=100, alpha=0.6, label='NSGA-II Pareto', 
               edgecolors='black', linewidth=1)
    
    # Plot baseline methods
    colors = plt.cm.tab10(np.arange(len(baseline_results)))
    for i, (method, df) in enumerate(baseline_results.items()):
        feasible = df[df['is_feasible']]
        if len(feasible) > 0:
            ax2.scatter(feasible['f1_total_cost_USD']/1000,
                       feasible['detection_recall'],
                       c=[colors[i]], s=50, alpha=0.5, 
                       label=method, marker='o')
    
    ax2.set_xlabel('Total Cost (k$)', fontsize=12)
    ax2.set_ylabel('Detection Recall', fontsize=12)
    ax2.set_title('Cost vs Performance Trade-off', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sustainability comparison
    ax3 = plt.subplot(3, 3, 3)
    
    ax3.scatter(pareto_results['f5_environmental_impact_kWh_year']/1000,
               pareto_results['system_MTBF_hours']/8760,
               c='red', s=100, alpha=0.6, label='NSGA-II Pareto',
               edgecolors='black', linewidth=1)
    
    for i, (method, df) in enumerate(baseline_results.items()):
        feasible = df[df['is_feasible']]
        if len(feasible) > 0:
            # Replace inf with very large value for plotting
            mtbf_hours = feasible['system_MTBF_hours'].replace([np.inf], 1e6)
            ax3.scatter(feasible['f5_environmental_impact_kWh_year']/1000,
                       mtbf_hours/8760,
                       c=[colors[i]], s=50, alpha=0.5,
                       label=method, marker='o')
    
    ax3.set_xlabel('Annual Energy (MWh)', fontsize=12)
    ax3.set_ylabel('System MTBF (years)', fontsize=12)
    ax3.set_title('Sustainability vs Reliability', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 20)  # Limit y-axis for better visualization
    
    # 4. Hypervolume comparison (simplified)
    ax4 = plt.subplot(3, 3, 4)
    
    hypervolumes = {}
    ref_point = np.array([2e7, 0.3, 200, 200, 100000, 0.001])  # Reference point
    
    # Calculate hypervolume for each method
    for method, df in baseline_results.items():
        feasible = df[df['is_feasible']]
        if len(feasible) > 0:
            # Simplified hypervolume (product of normalized ranges)
            obj_cols = ['f1_total_cost_USD', 'f2_one_minus_recall', 
                       'f3_latency_seconds', 'f4_traffic_disruption_hours',
                       'f5_environmental_impact_kWh_year', 'f6_system_reliability_inverse_MTBF']
            
            min_point = feasible[obj_cols].min()
            hv = np.prod(np.maximum(ref_point - min_point, 0))
            hypervolumes[method] = hv
    
    # Add NSGA-II
    min_point_pareto = pareto_results[['f1_total_cost_USD', 'f2_one_minus_recall',
                                      'f3_latency_seconds', 'f4_traffic_disruption_hours',
                                      'f5_environmental_impact_kWh_year', 
                                      'f6_system_reliability_inverse_MTBF']].min()
    hypervolumes['NSGA-II'] = np.prod(np.maximum(ref_point - min_point_pareto, 0))
    
    # Normalize and plot
    max_hv = max(hypervolumes.values())
    if max_hv > 0:
        norm_hv = {k: v/max_hv for k, v in hypervolumes.items()}
    else:
        norm_hv = {k: 0 for k in hypervolumes.keys()}
    
    bars = ax4.bar(norm_hv.keys(), norm_hv.values(), color='lightgreen')
    bars[-1].set_color('red')  # Highlight NSGA-II
    ax4.set_ylabel('Normalized Hypervolume', fontsize=12)
    ax4.set_title('Solution Quality Comparison', fontsize=14)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_ylim(0, 1.1)
    
    # 5. Computation time comparison
    ax5 = plt.subplot(3, 3, 5)
    
    time_data = []
    for method, df in baseline_results.items():
        time_data.append({
            'Method': method,
            'Total Time (s)': df['time_seconds'].max(),
            'Solutions': len(df),
            'Time per Solution (s)': df['time_seconds'].max() / len(df)
        })
    
    time_df = pd.DataFrame(time_data)
    
    ax5_twin = ax5.twinx()
    
    x = np.arange(len(time_df))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, time_df['Total Time (s)'], width, 
                    label='Total Time', color='skyblue')
    bars2 = ax5_twin.bar(x + width/2, time_df['Solutions'], width,
                        label='# Solutions', color='lightcoral')
    
    ax5.set_xlabel('Method', fontsize=12)
    ax5.set_ylabel('Total Time (seconds)', fontsize=12)
    ax5_twin.set_ylabel('Number of Solutions', fontsize=12)
    ax5.set_title('Computational Efficiency', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(time_df['Method'], rotation=45, ha='right')
    
    # Combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 6. Best solutions comparison table
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Find best solution for each method
    best_solutions = []
    for method, df in baseline_results.items():
        feasible = df[df['is_feasible']]
        if len(feasible) > 0:
            # Simple quality metric (lower is better)
            feasible['quality'] = (
                feasible['f1_total_cost_USD'] / 2e7 +
                feasible['f2_one_minus_recall'] * 10 +
                feasible['f3_latency_seconds'] / 200 +
                feasible['f5_environmental_impact_kWh_year'] / 100000
            )
            best = feasible.loc[feasible['quality'].idxmin()]
            best_solutions.append({
                'Method': method,
                'Cost (k$)': f"{best['f1_total_cost_USD']/1000:.0f}",
                'Recall': f"{best['detection_recall']:.3f}",
                'Energy (MWh)': f"{best['f5_environmental_impact_kWh_year']/1000:.1f}",
                'MTBF (yr)': f"{min(best['system_MTBF_hours']/8760, 100):.1f}"
            })
    
    # Add best NSGA-II
    best_pareto = pareto_results.loc[
        (pareto_results['f1_total_cost_USD'] / 2e7 +
         pareto_results['f2_one_minus_recall'] * 10 +
         pareto_results['f3_latency_seconds'] / 200 +
         pareto_results['f5_environmental_impact_kWh_year'] / 100000).idxmin()
    ]
    
    best_solutions.append({
        'Method': 'NSGA-II',
        'Cost (k$)': f"{best_pareto['f1_total_cost_USD']/1000:.0f}",
        'Recall': f"{best_pareto['detection_recall']:.3f}",
        'Energy (MWh)': f"{best_pareto['f5_environmental_impact_kWh_year']/1000:.1f}",
        'MTBF (yr)': f"{best_pareto['system_MTBF_hours']/8760:.1f}"
    })
    
    best_df = pd.DataFrame(best_solutions)
    
    table = ax6.table(cellText=best_df.values,
                     colLabels=best_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color NSGA-II row
    for j in range(len(best_df.columns)):
        table[(len(best_df), j)].set_facecolor('#ffcccc')
    
    ax6.set_title('Best Solution Comparison', fontsize=14)
    
    # 7-9. Additional objective comparisons
    for idx, (obj_name, obj_col, scale) in enumerate([
        ('Latency', 'f3_latency_seconds', 1),
        ('Traffic Disruption', 'f4_traffic_disruption_hours', 1),
        ('All Objectives', None, None)
    ]):
        ax = plt.subplot(3, 3, 7 + idx)
        
        if obj_col:
            # Box plot for single objective
            data_to_plot = []
            labels_to_plot = []
            
            for method, df in baseline_results.items():
                feasible = df[df['is_feasible']]
                if len(feasible) > 0:
                    data_to_plot.append(feasible[obj_col] / scale)
                    labels_to_plot.append(method)
            
            # Add NSGA-II
            data_to_plot.append(pareto_results[obj_col] / scale)
            labels_to_plot.append('NSGA-II')
            
            bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
            
            # Color boxes
            for i, patch in enumerate(bp['boxes']):
                if i < len(baseline_results):
                    patch.set_facecolor('lightblue')
                else:
                    patch.set_facecolor('lightcoral')
            
            ax.set_ylabel(f'{obj_name} ({scale}x)' if scale != 1 else obj_name, fontsize=12)
            ax.set_title(f'{obj_name} Distribution', fontsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
        else:
            # Radar chart for all objectives
            objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Energy', 'Reliability']
            
            # Calculate average normalized performance
            method_performance = {}
            
            for method, df in list(baseline_results.items())[:3]:  # Top 3 baselines
                feasible = df[df['is_feasible']]
                if len(feasible) > 0:
                    avg_perf = []
                    avg_perf.append(1 - (feasible['f1_total_cost_USD'].mean() - 100000) / 19900000)
                    avg_perf.append(feasible['detection_recall'].mean())
                    avg_perf.append(1 - (feasible['f3_latency_seconds'].mean() - 0.1) / 199.9)
                    avg_perf.append(1 - (feasible['f4_traffic_disruption_hours'].mean() / 200))
                    avg_perf.append(1 - (feasible['f5_environmental_impact_kWh_year'].mean() - 1000) / 99000)
                    mtbf_mean = feasible['system_MTBF_hours'].replace([np.inf], 1e6).mean()
                    avg_perf.append(min(mtbf_mean / 100000, 1))
                    
                    method_performance[method] = np.clip(avg_perf, 0, 1)
            
            # Add NSGA-II
            avg_perf = []
            avg_perf.append(1 - (pareto_results['f1_total_cost_USD'].mean() - 100000) / 19900000)
            avg_perf.append(pareto_results['detection_recall'].mean())
            avg_perf.append(1 - (pareto_results['f3_latency_seconds'].mean() - 0.1) / 199.9)
            avg_perf.append(1 - (pareto_results['f4_traffic_disruption_hours'].mean() / 200))
            avg_perf.append(1 - (pareto_results['f5_environmental_impact_kWh_year'].mean() - 1000) / 99000)
            avg_perf.append(pareto_results['system_MTBF_hours'].mean() / 100000)
            method_performance['NSGA-II'] = np.clip(avg_perf, 0, 1)
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])
            
            ax = plt.subplot(3, 3, 9, projection='polar')
            
            for method, performance in method_performance.items():
                values = np.concatenate([performance, [performance[0]]])
                ax.plot(angles, values, 'o-', linewidth=2, label=method)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(objectives)
            ax.set_ylim(0, 1)
            ax.set_title('Multi-Objective Performance', fontsize=14, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
    
    plt.suptitle('Baseline Methods vs NSGA-II Comparison (6 Objectives)', fontsize=18)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(f'{output_dir}/baseline_comparison_6obj.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/baseline_comparison_6obj.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # Save detailed comparison data
    comparison_summary = {
        'method_performance': {},
        'computation_time': {},
        'solution_quality': {}
    }
    
    for method, df in baseline_results.items():
        feasible = df[df['is_feasible']]
        comparison_summary['method_performance'][method] = {
            'total_solutions': len(df),
            'feasible_solutions': len(feasible),
            'feasibility_rate': len(feasible) / len(df) if len(df) > 0 else 0,
            'avg_cost': feasible['f1_total_cost_USD'].mean() if len(feasible) > 0 else np.nan,
            'avg_recall': feasible['detection_recall'].mean() if len(feasible) > 0 else np.nan,
            'avg_energy': feasible['f5_environmental_impact_kWh_year'].mean() if len(feasible) > 0 else np.nan,
            'computation_time': df['time_seconds'].max()
        }
    
    # Add NSGA-II stats
    comparison_summary['method_performance']['NSGA-II'] = {
        'total_solutions': len(pareto_results),
        'feasible_solutions': len(pareto_results),  # All Pareto solutions are feasible
        'feasibility_rate': 1.0,
        'avg_cost': pareto_results['f1_total_cost_USD'].mean(),
        'avg_recall': pareto_results['detection_recall'].mean(),
        'avg_energy': pareto_results['f5_environmental_impact_kWh_year'].mean(),
        'computation_time': 278.20  # From your log
    }
    
    # Save summary
    with open(f'{output_dir}/comparison_summary.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    
    logger.info(f"Baseline comparison completed. Results saved to {output_dir}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_baseline_comparison(ontology_graph: Graph, 
                          pareto_csv_path: str = './results/pareto_solutions_6d.csv'):
    """Run all baseline methods and compare with NSGA-II results"""
    
    # Configuration
    config = BaselineConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(ontology_graph, config)
    
    # Load Pareto results
    try:
        pareto_results = pd.read_csv(pareto_csv_path)
        logger.info(f"Loaded {len(pareto_results)} Pareto-optimal solutions for comparison")
    except FileNotFoundError:
        logger.error(f"Could not find Pareto results at {pareto_csv_path}")
        logger.error("Please run the main optimization first")
        return
    
    # Run baseline methods
    baseline_results = {}
    
    # 1. Random Search
    logger.info("\n" + "="*60)
    logger.info("Running Random Search Baseline...")
    random_baseline = RandomSearchBaseline(evaluator, config)
    baseline_results['RandomSearch'] = random_baseline.optimize()
    
    # 2. Grid Search
    logger.info("\n" + "="*60)
    logger.info("Running Grid Search Baseline...")
    grid_baseline = GridSearchBaseline(evaluator, config)
    baseline_results['GridSearch'] = grid_baseline.optimize()
    
    # 3. Weighted Sum
    logger.info("\n" + "="*60)
    logger.info("Running Weighted Sum Baseline...")
    weighted_baseline = WeightedSumBaseline(evaluator, config)
    baseline_results['WeightedSum'] = weighted_baseline.optimize()
    
    # 4. Expert Heuristic
    logger.info("\n" + "="*60)
    logger.info("Running Expert Heuristic Baseline...")
    expert_baseline = ExpertHeuristicBaseline(evaluator, config)
    baseline_results['ExpertHeuristic'] = expert_baseline.optimize()
    
    # Save individual results
    for method, df in baseline_results.items():
        df.to_csv(f"{config.output_dir}/{method.lower()}_results.csv", index=False)
    
    # Compare methods
    logger.info("\n" + "="*60)
    logger.info("Comparing baseline methods with NSGA-II...")
    compare_baseline_methods(baseline_results, pareto_results, config.output_dir)
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("BASELINE COMPARISON SUMMARY:")
    logger.info("-"*60)
    
    for method, df in baseline_results.items():
        feasible = df[df['is_feasible']]
        logger.info(f"\n{method}:")
        logger.info(f"  Total solutions: {len(df)}")
        logger.info(f"  Feasible solutions: {len(feasible)} ({len(feasible)/len(df)*100:.1f}%)")
        
        if len(feasible) > 0:
            logger.info(f"  Best cost: ${feasible['f1_total_cost_USD'].min():,.0f}")
            logger.info(f"  Best recall: {feasible['detection_recall'].max():.3f}")
            logger.info(f"  Best energy: {feasible['f5_environmental_impact_kWh_year'].min():.0f} kWh/year")
    
    logger.info(f"\nNSGA-II (Enhanced Framework):")
    logger.info(f"  Total solutions: {len(pareto_results)}")
    logger.info(f"  All Pareto-optimal (100% non-dominated)")
    logger.info(f"  Cost range: ${pareto_results['f1_total_cost_USD'].min():,.0f} - ${pareto_results['f1_total_cost_USD'].max():,.0f}")
    logger.info(f"  Recall range: {pareto_results['detection_recall'].min():.3f} - {pareto_results['detection_recall'].max():.3f}")
    logger.info(f"  Energy range: {pareto_results['f5_environmental_impact_kWh_year'].min():.0f} - {pareto_results['f5_environmental_impact_kWh_year'].max():.0f} kWh/year")
    
    logger.info("\n" + "="*60)
    logger.info("Baseline comparison completed!")
    
    return baseline_results

# Run if called directly
if __name__ == "__main__":
    # This would typically be called from the main framework
    # For standalone testing, you would need to load the ontology first
    print("Please run this through the main framework or provide an ontology graph")
    print("Example usage:")
    print("  from baseline_comparison import run_baseline_comparison")
    print("  run_baseline_comparison(ontology_graph, './results/pareto_solutions_6d.csv')")
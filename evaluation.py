#!/usr/bin/env python3
"""
Enhanced Fitness Evaluation Module V3
6-objective optimization with ProcessPoolExecutor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS, RDF, OWL, XSD
import pickle

logger = logging.getLogger(__name__)

RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class SolutionMapper:
    """Maps between optimization variables and RMTwin configurations"""
    
    def __init__(self, ontology_graph: Graph):
        self.g = ontology_graph
        self._cache_components()
        
    def _cache_components(self):
        """Cache all available components from ontology"""
        self.sensors = []
        self.algorithms = []
        self.storage_systems = []
        self.comm_systems = []
        self.deployments = []
        
        logger.info("Caching components from ontology...")
        
        # Query for all instances
        for s, p, o in self.g:
            subject_str = str(s)
            object_str = str(o)
            
            if subject_str.startswith('http://example.org/'):
                if 'SensorSystem' in object_str or 'Sensor' in object_str:
                    self.sensors.append(subject_str)
                elif 'Algorithm' in object_str:
                    self.algorithms.append(subject_str)
                elif 'StorageSystem' in object_str:
                    self.storage_systems.append(subject_str)
                elif 'CommunicationSystem' in object_str:
                    self.comm_systems.append(subject_str)
                elif 'ComputeDeployment' in object_str or 'Deployment' in object_str:
                    self.deployments.append(subject_str)
        
        # Remove duplicates
        self.sensors = list(set(self.sensors))
        self.algorithms = list(set(self.algorithms))
        self.storage_systems = list(set(self.storage_systems))
        self.comm_systems = list(set(self.comm_systems))
        self.deployments = list(set(self.deployments))
        
        # Ensure we have at least some components
        if not self.sensors:
            self.sensors = ["http://example.org/rmtwin#MMS_LiDAR_Riegl_VUX1HA"]
        if not self.algorithms:
            self.algorithms = ["http://example.org/rmtwin#DL_YOLOv5s_Enhanced"]
        if not self.storage_systems:
            self.storage_systems = ["http://example.org/rmtwin#Storage_Cloud_AWS_S3"]
        if not self.comm_systems:
            self.comm_systems = ["http://example.org/rmtwin#Communication_5G_Network"]
        if not self.deployments:
            self.deployments = ["http://example.org/rmtwin#Deployment_Cloud_Computing"]
        
        logger.info(f"Cached components: {len(self.sensors)} sensors, "
                   f"{len(self.algorithms)} algorithms, {len(self.storage_systems)} storage, "
                   f"{len(self.comm_systems)} communication, {len(self.deployments)} deployment")
    
    def decode_solution(self, x: np.ndarray) -> Dict:
        """Decode solution vector to configuration"""
        x = x.flatten() if isinstance(x, np.ndarray) else x
        
        config = {
            'sensor': self.sensors[int(x[0] * len(self.sensors)) % len(self.sensors)],
            'data_rate': 10 + x[1] * 90,  # 10-100 Hz
            'geo_lod': ['Micro', 'Meso', 'Macro'][int(x[2] * 3) % 3],
            'cond_lod': ['Micro', 'Meso', 'Macro'][int(x[3] * 3) % 3],
            'algorithm': self.algorithms[int(x[4] * len(self.algorithms)) % len(self.algorithms)],
            'detection_threshold': 0.1 + x[5] * 0.8,  # 0.1-0.9
            'storage': self.storage_systems[int(x[6] * len(self.storage_systems)) % len(self.storage_systems)],
            'communication': self.comm_systems[int(x[7] * len(self.comm_systems)) % len(self.comm_systems)],
            'deployment': self.deployments[int(x[8] * len(self.deployments)) % len(self.deployments)],
            'crew_size': int(1 + x[9] * 9),  # 1-10
            'inspection_cycle': int(1 + x[10] * 364)  # 1-365 days
        }
        
        return config


def _evaluate_single_wrapper(args):
    """Wrapper function for process pool evaluation"""
    x, property_cache, config_dict, mapper_data = args
    
    # Reconstruct mapper with cached data
    mapper = SolutionMapper.__new__(SolutionMapper)
    mapper.sensors = mapper_data['sensors']
    mapper.algorithms = mapper_data['algorithms']
    mapper.storage_systems = mapper_data['storage_systems']
    mapper.comm_systems = mapper_data['comm_systems']
    mapper.deployments = mapper_data['deployments']
    
    # Decode configuration
    config = mapper.decode_solution(x)
    
    # Create a simple evaluator for calculation
    evaluator = SimpleEvaluator(property_cache, config_dict)
    
    # Calculate objectives
    objectives = evaluator.calculate_all_objectives(config)
    constraints = evaluator.calculate_constraints(objectives)
    
    return objectives, constraints


class SimpleEvaluator:
    """Simplified evaluator for process pool execution"""
    
    def __init__(self, property_cache, config):
        self._property_cache = property_cache
        self.config = config
        
    def _query_property(self, subject: str, predicate: str, default=None):
        """Get property value from cache"""
        if subject in self._property_cache:
            if predicate in self._property_cache[subject]:
                return self._property_cache[subject][predicate]
        return default
    
    def calculate_all_objectives(self, config: Dict) -> np.ndarray:
        """Calculate all 6 objectives"""
        f1 = self._calculate_total_cost_v2(config)
        f2 = self._calculate_detection_performance_v2(config)
        f3 = self._calculate_latency_v2(config)
        f4 = self._calculate_traffic_disruption_v2(config)
        f5 = self._calculate_environmental_impact_v2(config)
        f6 = self._calculate_system_reliability_v2(config)
        
        return np.array([f1, f2, f3, f4, f5, f6])
    
    def calculate_constraints(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate constraints based on objectives"""
        f1, f2, f3, f4, f5, f6 = objectives
        recall = 1 - f2
        
        constraints = np.array([
            f3 - self.config['max_latency_seconds'],  # Max latency
            self.config['min_recall_threshold'] - recall,  # Min recall
            f1 - self.config['budget_cap_usd'],  # Budget
            f5 - self.config.get('max_carbon_emissions_kgCO2e_year', 50000),  # Max carbon
            self.config.get('min_mtbf_hours', 5000) - (1/f6 if f6 > 0 else 1e6)  # Min MTBF
        ])
        
        return constraints
    
    def _calculate_total_cost_v2(self, config: Dict) -> float:
        """Enhanced cost calculation including all factors"""
        sensor_name = str(config['sensor']).split('#')[-1]
        
        # Initial investment
        sensor_initial_cost = self._query_property(
            config['sensor'], 'hasInitialCostUSD', 100000)
        
        # Special handling for FOS
        if 'FOS' in sensor_name or 'Fiber' in sensor_name:
            sensor_spacing_km = self.config.get('fos_sensor_spacing_km', 0.1)
            sensors_needed = self.config['road_network_length_km'] / sensor_spacing_km
            actual_sensor_cost = sensor_initial_cost * sensors_needed
            installation_cost = 5000 * sensors_needed
            total_sensor_initial = actual_sensor_cost + installation_cost
        else:
            total_sensor_initial = sensor_initial_cost
        
        # Other components
        storage_initial = self._query_property(config['storage'], 'hasInitialCostUSD', 0)
        comm_initial = self._query_property(config['communication'], 'hasInitialCostUSD', 0)
        deployment_initial = self._query_property(config['deployment'], 'hasInitialCostUSD', 0)
        algo_initial = self._query_property(config['algorithm'], 'hasInitialCostUSD', 20000)
        
        total_initial_investment = (total_sensor_initial + storage_initial + 
                                  comm_initial + deployment_initial + algo_initial)
        
        # Annual operational costs
        depreciation_rate = self.config.get('depreciation_rate', 0.1)
        annual_capital_cost = total_initial_investment * depreciation_rate
        
        # Sensor operational costs
        sensor_daily_cost = self._query_property(
            config['sensor'], 'hasOperationalCostUSDPerDay', 100)
        coverage_km_day = self._query_property(
            config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        
        if coverage_km_day > 0:  # Mobile sensor
            inspections_per_year = 365 / config['inspection_cycle']
            days_per_inspection = self.config['road_network_length_km'] / coverage_km_day
            sensor_annual_operational = sensor_daily_cost * days_per_inspection * inspections_per_year
        else:  # Fixed sensor
            if 'FOS' in sensor_name:
                operational_cost_per_sensor_day = 0.5
                sensors_needed = self.config['road_network_length_km'] / sensor_spacing_km
                sensor_annual_operational = operational_cost_per_sensor_day * sensors_needed * 365
            else:
                sensor_annual_operational = sensor_daily_cost * 365
        
        # Other operational costs
        storage_annual = self._query_property(config['storage'], 'hasAnnualOpCostUSD', 5000)
        comm_annual = self._query_property(config['communication'], 'hasAnnualOpCostUSD', 2000)
        deployment_annual = self._query_property(config['deployment'], 'hasAnnualOpCostUSD', 10000)
        
        # Crew costs with skill multiplier
        skill_level = self._query_property(config['sensor'], 'hasOperatorSkillLevel', 'Basic')
        skill_multiplier = {
            'Basic': 1.0, 'Intermediate': 1.5, 'Expert': 2.0
        }.get(str(skill_level), 1.0)
        
        daily_wage = self.config['daily_wage_per_person'] * skill_multiplier
        
        if coverage_km_day > 0:
            crew_annual_cost = (config['crew_size'] * daily_wage * 
                              days_per_inspection * inspections_per_year)
        else:
            maintenance_days = 10 if 'FOS' in sensor_name else 20
            crew_annual_cost = config['crew_size'] * daily_wage * maintenance_days
        
        # Data annotation costs (for DL algorithms)
        data_annotation_annual = 0
        if 'DL' in str(config['algorithm']) or 'Deep' in str(config['algorithm']):
            annotation_cost = self._query_property(
                config['algorithm'], 'hasDataAnnotationCostUSD', 0.5)
            
            if 'Camera' in sensor_name:
                images_per_km = 100
                annual_images = images_per_km * self.config['road_network_length_km'] * inspections_per_year
            else:
                annual_images = 10000
            
            data_annotation_annual = annotation_cost * annual_images
        
        # Model retraining costs
        retrain_freq = self._query_property(config['algorithm'], 'hasModelRetrainingFreqMonths', 12)
        if retrain_freq > 0:
            retrainings_per_year = 12 / retrain_freq
            retraining_cost = 5000  # Base retraining cost
            model_retraining_annual = retraining_cost * retrainings_per_year
        else:
            model_retraining_annual = 0
        
        # Total annual cost
        total_annual_cost = (annual_capital_cost + sensor_annual_operational + 
                           storage_annual + comm_annual + deployment_annual + 
                           crew_annual_cost + data_annotation_annual + model_retraining_annual)
        
        # Apply seasonal adjustments
        if self.config.get('apply_seasonal_adjustments', True):
            winter_factor = 1.3  # 30% increase for winter operations
            seasonal_adjustment = 0.25  # 25% of operations in winter
            total_annual_cost *= (1 + (winter_factor - 1) * seasonal_adjustment)
        
        # Total lifecycle cost
        total_lifecycle_cost = total_annual_cost * self.config['planning_horizon_years']
        
        return total_lifecycle_cost
    
    def _calculate_detection_performance_v2(self, config: Dict) -> float:
        """Enhanced detection performance calculation"""
        base_recall = self._query_property(config['algorithm'], 'hasRecall', 0.7)
        
        # Sensor accuracy impact
        accuracy_mm = self._query_property(config['sensor'], 'hasAccuracyRangeMM', 10)
        accuracy_factor = 1 - (accuracy_mm / 100)
        
        # LOD impact with more nuanced factors
        lod_factors = {
            'Micro': {'factor': 1.15, 'threshold_adj': 0.05},
            'Meso': {'factor': 1.0, 'threshold_adj': 0},
            'Macro': {'factor': 0.85, 'threshold_adj': -0.05}
        }
        
        lod_data = lod_factors.get(config['geo_lod'], lod_factors['Meso'])
        lod_factor = lod_data['factor']
        
        # Detection threshold impact
        threshold_optimal = 0.5
        threshold_penalty = abs(config['detection_threshold'] - threshold_optimal) * 0.1
        
        # Algorithm-specific adjustments
        algo_name = str(config['algorithm']).split('#')[-1]
        
        # Class imbalance penalties
        class_imbalance_penalties = self.config.get('class_imbalance_penalties', {
            'Traditional': 0.05, 'ML': 0.02, 'DL': 0.01, 'PC': 0.03
        })
        
        penalty = 0
        for algo_type, pen_value in class_imbalance_penalties.items():
            if algo_type in algo_name:
                penalty = pen_value
                break
        
        # Hardware requirements impact
        hardware_req = self._query_property(config['algorithm'], 'hasHardwareRequirement', 'CPU')
        if 'HighEnd_GPU' in str(hardware_req):
            # High-end GPU enables better performance
            hardware_bonus = 0.02
        else:
            hardware_bonus = 0
        
        # Calculate final recall
        final_recall = (base_recall * accuracy_factor * lod_factor + 
                       hardware_bonus - penalty - threshold_penalty)
        
        # Add noise for realism
        noise = np.random.normal(0, 0.01)
        final_recall += noise
        
        return 1 - np.clip(final_recall, 0.01, 0.99)
    
    def _calculate_latency_v2(self, config: Dict) -> float:
        """Enhanced latency calculation with detailed modeling"""
        # Data acquisition time
        data_rate = config['data_rate']
        acq_time = 1 / data_rate if data_rate > 0 else 1.0
        
        # Data volume based on sensor and LOD
        base_data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        
        lod_multipliers = {'Micro': 2.0, 'Meso': 1.0, 'Macro': 0.5}
        data_gb = base_data_gb * lod_multipliers.get(config['geo_lod'], 1.0)
        
        # Communication latency
        comm_type = str(config['communication']).split('#')[-1]
        comm_specs = {
            'Communication_5G_Network': {'bandwidth': 1000, 'latency': 0.01},
            'Communication_LoRaWAN': {'bandwidth': 0.05, 'latency': 1.0},
            'Communication_Fiber_Optic': {'bandwidth': 10000, 'latency': 0.002},
            'Communication_4G_LTE': {'bandwidth': 100, 'latency': 0.05}
        }
        
        comm_data = comm_specs.get(comm_type, {'bandwidth': 100, 'latency': 0.05})
        
        # Apply scenario-based network quality
        scenario_type = self.config.get('scenario_type', 'urban')
        network_quality_factors = self.config.get('network_quality_factors', {
            'rural': {'Fiber': 0.8, '5G': 0.7, '4G': 0.9, 'LoRaWAN': 1.0},
            'urban': {'Fiber': 1.0, '5G': 1.0, '4G': 1.0, 'LoRaWAN': 0.9},
            'mixed': {'Fiber': 0.9, '5G': 0.85, '4G': 0.95, 'LoRaWAN': 0.95}
        })
        
        tech = None
        for t in ['Fiber', '5G', '4G', 'LoRaWAN']:
            if t in comm_type:
                tech = t
                break
        
        scenario_factor = 1.0
        if tech and scenario_type in network_quality_factors:
            scenario_factor = network_quality_factors[scenario_type].get(tech, 1.0)
        
        effective_bandwidth = comm_data['bandwidth'] * scenario_factor
        network_latency = comm_data['latency'] / scenario_factor
        
        # Communication time
        comm_time = (data_gb * 1000) / effective_bandwidth if effective_bandwidth > 0 else 100
        
        # Processing time based on algorithm and deployment
        algo_fps = self._query_property(config['algorithm'], 'hasFPS', 10)
        base_proc_time = 1 / algo_fps if algo_fps > 0 else 0.1
        
        # Deployment impact
        deploy_factors = {
            'Edge': {'factor': 1.5, 'overhead': 0.02},
            'Cloud': {'factor': 1.0, 'overhead': 0.05},
            'Hybrid': {'factor': 1.2, 'overhead': 0.03},
            'OnPremise': {'factor': 1.3, 'overhead': 0.01}
        }
        
        deploy_type = config['deployment'].split('_')[-1]
        deploy_data = deploy_factors.get(deploy_type, {'factor': 1.0, 'overhead': 0.05})
        
        proc_time = base_proc_time * deploy_data['factor'] + deploy_data['overhead']
        
        # Total latency
        total_latency = acq_time + network_latency + comm_time + proc_time
        
        return total_latency
    
    def _calculate_traffic_disruption_v2(self, config: Dict) -> float:
        """Enhanced traffic disruption calculation"""
        # Base disruption per inspection
        base_disruption = 4.0  # hours
        
        # Sensor-specific adjustments
        sensor_name = str(config['sensor']).split('#')[-1]
        speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
        
        if 'UAV' in sensor_name:
            # UAVs cause minimal traffic disruption
            disruption_factor = 0.1
        elif 'FOS' in sensor_name or 'IoT' in sensor_name:
            # Fixed sensors - disruption only during installation/maintenance
            disruption_factor = 0.05
        elif speed > 0:
            # Mobile sensors - disruption based on speed
            speed_factor = 80 / speed if speed < 80 else 1.0
            disruption_factor = speed_factor
        else:
            disruption_factor = 1.0
        
        # Calculate annual disruption
        inspections_per_year = 365 / config['inspection_cycle']
        
        if 'FOS' in sensor_name or 'IoT' in sensor_name:
            # Fixed sensors - maintenance visits only
            maintenance_visits = 4  # Quarterly maintenance
            annual_disruption = base_disruption * disruption_factor * maintenance_visits
        else:
            annual_disruption = base_disruption * disruption_factor * inspections_per_year
        
        # Traffic volume impact
        traffic_volume = self.config.get('traffic_volume_hourly', 2000)
        traffic_factor = (traffic_volume / 1000) ** 0.5  # Square root for diminishing impact
        
        # Lane closure impact
        default_lane_closure = self.config.get('default_lane_closure_ratio', 0.3)
        lane_factor = 1 + default_lane_closure
        
        # Time of day optimization
        night_work_ratio = 0.3  # 30% of work can be done at night
        night_impact_reduction = 0.7  # 70% reduction in impact for night work
        time_factor = 1 - (night_work_ratio * night_impact_reduction)
        
        # Total disruption
        total_disruption = annual_disruption * lane_factor * traffic_factor * time_factor
        
        return total_disruption
    
    def _calculate_environmental_impact_v2(self, config: Dict) -> float:
        """Calculate environmental impact in kgCO2e/year (f5)"""
        # Energy consumption from components
        total_power_w = 0
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                power = self._query_property(config[comp], 'hasEnergyConsumptionW', 0)
                total_power_w += power if power else 0
        
        # Operating hours calculation
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        
        if coverage > 0:
            # Mobile sensor operation
            inspections_per_year = 365 / config['inspection_cycle']
            days_per_inspection = self.config['road_network_length_km'] / coverage
            sensor_hours = days_per_inspection * inspections_per_year * 8  # 8 hours/day
            
            # Vehicle emissions
            vehicle_km = self.config['road_network_length_km'] * inspections_per_year
            vehicle_fuel_l = vehicle_km * 0.08  # 8L/100km fuel consumption
            vehicle_emissions_kg = vehicle_fuel_l * 2.31  # 2.31 kg CO2/L gasoline
        else:
            # Fixed sensor - continuous operation
            sensor_hours = 365 * 24
            vehicle_emissions_kg = 0
        
        # Backend/cloud operations - always running
        backend_hours = 365 * 24
        
        # Energy consumption breakdown
        sensor_energy_kwh = (total_power_w * 0.3 * sensor_hours) / 1000
        backend_energy_kwh = (total_power_w * 0.7 * backend_hours) / 1000
        total_energy_kwh = sensor_energy_kwh + backend_energy_kwh
        
        # Carbon intensity
        carbon_intensity = self.config.get('carbon_intensity_factor', 0.417)  # kg CO2/kWh
        electricity_emissions_kg = total_energy_kwh * carbon_intensity
        
        # Manufacturing emissions (amortized)
        equipment_cost = self._query_property(config['sensor'], 'hasInitialCostUSD', 100000)
        manufacturing_emissions = equipment_cost * 0.001  # Rough estimate: 1 kg CO2/$1000
        annual_manufacturing = manufacturing_emissions / 10  # 10-year lifespan
        
        # Total emissions
        total_emissions = electricity_emissions_kg + vehicle_emissions_kg + annual_manufacturing
        
        return total_emissions
    
    def _calculate_system_reliability_v2(self, config: Dict) -> float:
        """Calculate system reliability as 1/MTBF (f6)"""
        # Component reliability (series system model)
        component_mtbfs = []
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                base_mtbf = self._query_property(config[comp], 'hasMTBFHours', 10000)
                
                # Apply redundancy factors based on deployment type
                comp_type = str(config[comp]).split('#')[-1]
                
                redundancy_multipliers = self.config.get('redundancy_multipliers', {
                    'Cloud': 10.0,      # High redundancy
                    'OnPremise': 1.5,   # Some redundancy
                    'Edge': 2.0,        # Moderate redundancy
                    'Hybrid': 5.0       # Good redundancy
                })
                
                redundancy = 1.0
                for red_type, mult in redundancy_multipliers.items():
                    if red_type in comp_type:
                        redundancy = mult
                        break
                
                # Environmental factors
                if 'outdoor' in str(self._query_property(config[comp], 'hasEnvironment', '')).lower():
                    environmental_factor = 0.8  # 20% reduction for outdoor equipment
                else:
                    environmental_factor = 1.0
                
                effective_mtbf = base_mtbf * redundancy * environmental_factor
                
                if effective_mtbf > 0:
                    component_mtbfs.append(effective_mtbf)
        
        # Algorithm reliability (software)
        algo_mtbf = self._query_property(config['algorithm'], 'hasMTBFHours', 50000)
        if algo_mtbf > 0:
            component_mtbfs.append(algo_mtbf)
        
        # System MTBF calculation (series system)
        if component_mtbfs:
            # For series system: 1/MTBF_system = sum(1/MTBF_i)
            inverse_mtbf_sum = sum(1/mtbf for mtbf in component_mtbfs)
            
            # Add common cause failures
            common_cause_rate = 1 / (365 * 24 * 10)  # Once per 10 years
            inverse_mtbf_sum += common_cause_rate
            
            return inverse_mtbf_sum
        else:
            return 1e-6  # Default to very high reliability if no data


class EnhancedFitnessEvaluatorV2:
    """Enhanced fitness evaluator with 6 objectives and process pool support"""
    
    def __init__(self, ontology_graph: Graph, config):
        self.g = ontology_graph
        self.config = config
        self.solution_mapper = SolutionMapper(ontology_graph)
        
        # Component property cache
        self._property_cache = {}
        
        # Pre-cache all properties at initialization
        self._initialize_cache()
        
        # Prepare data for process pool
        self._mapper_data = {
            'sensors': self.solution_mapper.sensors,
            'algorithms': self.solution_mapper.algorithms,
            'storage_systems': self.solution_mapper.storage_systems,
            'comm_systems': self.solution_mapper.comm_systems,
            'deployments': self.solution_mapper.deployments
        }
        
        # Convert config to dict for serialization
        self._config_dict = vars(config) if hasattr(config, '__dict__') else config
        
        # Statistics
        self._evaluation_count = 0
        
    def _initialize_cache(self):
        """Cache all properties at initialization time"""
        logger.info("Initializing property cache...")
        
        # Define properties to cache
        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasAnnualOpCostUSD',
            'hasEnergyConsumptionW', 'hasMTBFHours', 'hasOperatorSkillLevel',
            'hasCalibrationFreqMonths', 'hasDataAnnotationCostUSD',
            'hasModelRetrainingFreqMonths', 'hasExplainabilityScore',
            'hasIntegrationComplexity', 'hasCybersecurityVulnerability',
            'hasAccuracyRangeMM', 'hasDataVolumeGBPerKm',
            'hasCoverageEfficiencyKmPerDay', 'hasOperatingSpeedKmh',
            'hasRecall', 'hasPrecision', 'hasFPS', 'hasHardwareRequirement',
            'hasEnvironment'
        ]
        
        # Get all components
        all_components = (
            self.solution_mapper.sensors + 
            self.solution_mapper.algorithms + 
            self.solution_mapper.storage_systems + 
            self.solution_mapper.comm_systems + 
            self.solution_mapper.deployments
        )
        
        # Cache properties for each component
        for component in all_components:
            if component not in self._property_cache:
                self._property_cache[component] = {}
            
            for prop in properties:
                query = f"""
                PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
                SELECT ?value WHERE {{
                    <{component}> rdtco:{prop} ?value .
                }}
                """
                
                try:
                    results = list(self.g.query(query))
                    if results:
                        value = results[0][0]
                        try:
                            self._property_cache[component][prop] = float(value)
                        except:
                            self._property_cache[component][prop] = str(value)
                except Exception as e:
                    # Skip if property doesn't exist
                    pass
        
        logger.info(f"Cached properties for {len(self._property_cache)} components")
    
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of solutions using process pool if enabled"""
        n_solutions = len(X)
        objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 5))
        
        # Use process pool for large batches
        if self.config.use_parallel and n_solutions > 10:
            # Prepare arguments for process pool
            args_list = []
            for x in X:
                args = (x, self._property_cache, self._config_dict, self._mapper_data)
                args_list.append(args)
            
            # Execute in parallel
            with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
                futures = {executor.submit(_evaluate_single_wrapper, args): i 
                          for i, args in enumerate(args_list)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        obj, const = future.result()
                        objectives[idx] = obj
                        constraints[idx] = const
                    except Exception as e:
                        logger.error(f"Error evaluating solution {idx}: {e}")
                        # Set high penalty values
                        objectives[idx] = np.array([1e10, 1, 1000, 1000, 100000, 1])
                        constraints[idx] = np.array([1000, 1, 1e10, 50000, -1000])
        else:
            # Sequential evaluation for small batches
            for i, x in enumerate(X):
                try:
                    args = (x, self._property_cache, self._config_dict, self._mapper_data)
                    objectives[i], constraints[i] = _evaluate_single_wrapper(args)
                except Exception as e:
                    logger.error(f"Error evaluating solution {i}: {e}")
                    objectives[i] = np.array([1e10, 1, 1000, 1000, 100000, 1])
                    constraints[i] = np.array([1000, 1, 1e10, 50000, -1000])
        
        # Update statistics
        self._evaluation_count += n_solutions
        
        # Log progress periodically
        if self._evaluation_count % 1000 == 0:
            logger.info(f"Evaluated {self._evaluation_count} solutions.")
            
            # Log objective ranges
            logger.debug(f"Objective ranges:")
            logger.debug(f"  Cost: ${objectives[:, 0].min():.0f} - ${objectives[:, 0].max():.0f}")
            logger.debug(f"  1-Recall: {objectives[:, 1].min():.3f} - {objectives[:, 1].max():.3f}")
            logger.debug(f"  Latency: {objectives[:, 2].min():.1f} - {objectives[:, 2].max():.1f}s")
            logger.debug(f"  Disruption: {objectives[:, 3].min():.1f} - {objectives[:, 3].max():.1f}h")
            logger.debug(f"  Carbon: {objectives[:, 4].min():.0f} - {objectives[:, 4].max():.0f} kgCO2e")
            logger.debug(f"  1/MTBF: {objectives[:, 5].min():.6f} - {objectives[:, 5].max():.6f}")
        
        return objectives, constraints
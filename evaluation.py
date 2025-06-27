#!/usr/bin/env python3
"""
Enhanced Fitness Evaluation Module V2
Fixed version without threading issues
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from functools import lru_cache
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS, RDF, OWL, XSD

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


class EnhancedFitnessEvaluatorV2:
    """Enhanced fitness evaluator - simplified without threading"""
    
    def __init__(self, ontology_graph: Graph, config):
        self.g = ontology_graph
        self.config = config
        self.solution_mapper = SolutionMapper(ontology_graph)
        
        # Component property cache
        self._property_cache = {}
        
        # Pre-cache all properties at initialization (sequential)
        self._initialize_cache()
        
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
            'hasRecall', 'hasPrecision', 'hasFPS'
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
    
    def _query_property(self, subject: str, predicate: str, default=None):
        """Get property value from cache only"""
        if subject in self._property_cache:
            if predicate in self._property_cache[subject]:
                return self._property_cache[subject][predicate]
        return default
    
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of solutions - sequential only"""
        n_solutions = len(X)
        objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 3))
        
        # Always use sequential evaluation to avoid threading issues
        for i, x in enumerate(X):
            obj, const = self._evaluate_single(x)
            objectives[i] = obj
            constraints[i] = const
        
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
            logger.debug(f"  Carbon: {objectives[:, 4].min():.0f} - {objectives[:, 4].max():.0f} kgCO2e")
        
        return objectives, constraints
    
    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate single solution"""
        config = self.solution_mapper.decode_solution(x)
        
        # Calculate objectives (all from cache, no SPARQL queries)
        f1 = self._calculate_total_cost_v2(config)
        f2 = self._calculate_detection_performance_v2(config)
        f3 = self._calculate_latency_v2(config)
        f4 = self._calculate_traffic_disruption_v2(config)
        f5 = self._calculate_environmental_impact_v2(config)
        f6 = self._calculate_system_reliability_v2(config)
        
        objectives = np.array([f1, f2, f3, f4, f5, f6])
        
        # Calculate constraints
        recall = 1 - f2
        constraints = np.array([
            f3 - self.config.max_latency_seconds,  # Max latency
            self.config.min_recall_threshold - recall,  # Min recall
            f1 - self.config.budget_cap_usd  # Budget
        ])
        
        return objectives, constraints
    
    def _calculate_total_cost_v2(self, config: Dict) -> float:
        """Enhanced cost calculation including all factors"""
        sensor_name = str(config['sensor']).split('#')[-1]
        
        # Initial investment
        sensor_initial_cost = self._query_property(
            config['sensor'], 'hasInitialCostUSD', 100000)
        
        # Special handling for FOS
        if 'FOS' in sensor_name or 'Fiber' in sensor_name:
            sensor_spacing_km = getattr(self.config, 'fos_sensor_spacing_km', 0.1)
            sensors_needed = self.config.road_network_length_km / sensor_spacing_km
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
        depreciation_rate = getattr(self.config, 'depreciation_rate', 0.1)
        annual_capital_cost = total_initial_investment * depreciation_rate
        
        # Sensor operational costs
        sensor_daily_cost = self._query_property(
            config['sensor'], 'hasOperationalCostUSDPerDay', 100)
        coverage_km_day = self._query_property(
            config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        
        if coverage_km_day > 0:  # Mobile sensor
            inspections_per_year = 365 / config['inspection_cycle']
            days_per_inspection = self.config.road_network_length_km / coverage_km_day
            sensor_annual_operational = sensor_daily_cost * days_per_inspection * inspections_per_year
        else:  # Fixed sensor
            if 'FOS' in sensor_name:
                operational_cost_per_sensor_day = 0.5
                sensors_needed = self.config.road_network_length_km / sensor_spacing_km
                sensor_annual_operational = operational_cost_per_sensor_day * sensors_needed * 365
            else:
                sensor_annual_operational = sensor_daily_cost * 365
        
        # Other operational costs
        storage_annual = self._query_property(config['storage'], 'hasAnnualOpCostUSD', 5000)
        comm_annual = self._query_property(config['communication'], 'hasAnnualOpCostUSD', 2000)
        deployment_annual = self._query_property(config['deployment'], 'hasAnnualOpCostUSD', 10000)
        
        # Crew costs
        skill_level = self._query_property(config['sensor'], 'hasOperatorSkillLevel', 'Basic')
        skill_multiplier = {
            'Basic': 1.0, 'Intermediate': 1.5, 'Expert': 2.0
        }.get(str(skill_level), 1.0)
        
        daily_wage = self.config.daily_wage_per_person * skill_multiplier
        
        if coverage_km_day > 0:
            crew_annual_cost = (config['crew_size'] * daily_wage * 
                              days_per_inspection * inspections_per_year)
        else:
            maintenance_days = 10 if 'FOS' in sensor_name else 20
            crew_annual_cost = config['crew_size'] * daily_wage * maintenance_days
        
        # Data annotation costs (for DL)
        data_annotation_annual = 0
        if 'DL' in str(config['algorithm']) or 'Deep' in str(config['algorithm']):
            annotation_cost = self._query_property(
                config['algorithm'], 'hasDataAnnotationCostUSD', 0.5)
            
            if 'Camera' in sensor_name:
                images_per_km = 100
                annual_images = images_per_km * self.config.road_network_length_km * inspections_per_year
            else:
                annual_images = 10000
            
            data_annotation_annual = annotation_cost * annual_images
        
        # Total annual cost
        total_annual_cost = (annual_capital_cost + sensor_annual_operational + 
                           storage_annual + comm_annual + deployment_annual + 
                           crew_annual_cost + data_annotation_annual)
        
        # Total lifecycle cost
        total_lifecycle_cost = total_annual_cost * self.config.planning_horizon_years
        
        return total_lifecycle_cost
    
    def _calculate_detection_performance_v2(self, config: Dict) -> float:
        """Enhanced detection performance with class imbalance"""
        base_recall = self._query_property(config['algorithm'], 'hasRecall', 0.7)
        
        # Sensor accuracy impact
        accuracy_mm = self._query_property(config['sensor'], 'hasAccuracyRangeMM', 10)
        accuracy_factor = 1 - (accuracy_mm / 100)
        
        # LOD impact
        lod_factor = {'Micro': 1.1, 'Meso': 1.0, 'Macro': 0.9}.get(config['geo_lod'], 1.0)
        
        # Class imbalance penalty
        algo_name = str(config['algorithm']).split('#')[-1]
        class_imbalance_penalties = getattr(self.config, 'class_imbalance_penalties', {
            'Traditional': 0.05, 'ML': 0.02, 'DL': 0.01, 'PC': 0.03
        })
        
        penalty = 0
        for algo_type, pen_value in class_imbalance_penalties.items():
            if algo_type in algo_name:
                penalty = pen_value
                break
        
        final_recall = base_recall * accuracy_factor * lod_factor - penalty
        return 1 - np.clip(final_recall, 0.01, 0.99)
    
    def _calculate_latency_v2(self, config: Dict) -> float:
        """Enhanced latency with scenario-aware networking"""
        # Data volume
        data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        
        # Base bandwidth
        comm_type = str(config['communication']).split('#')[-1]
        base_bw = {
            'Communication_5G_Network': 1000,
            'Communication_LoRaWAN': 0.05,
            'Communication_Fiber_Optic': 10000,
            'Communication_4G_LTE': 100
        }.get(comm_type, 100)
        
        # Scenario impact
        scenario_type = getattr(self.config, 'scenario_type', 'urban')
        network_quality_factors = getattr(self.config, 'network_quality_factors', {
            'rural': {'Fiber': 0.8, '5G': 0.7, '4G': 0.9, 'LoRaWAN': 1.0},
            'urban': {'Fiber': 1.0, '5G': 1.0, '4G': 1.0, 'LoRaWAN': 0.9}
        })
        
        tech = None
        for t in ['Fiber', '5G', '4G', 'LoRaWAN']:
            if t in comm_type:
                tech = t
                break
        
        scenario_factor = 1.0
        if tech and scenario_type in network_quality_factors:
            scenario_factor = network_quality_factors[scenario_type].get(tech, 1.0)
        
        effective_bw = base_bw * scenario_factor
        
        # Communication time
        comm_time = (data_gb * 1000) / effective_bw if effective_bw > 0 else 100
        
        # Processing time
        deploy_factor = {
            'Edge': 1.5, 'Cloud': 1.0, 'Hybrid': 1.2, 'OnPremise': 1.3
        }.get(config['deployment'].split('_')[-1], 1.0)
        
        proc_time = 0.1 * deploy_factor
        
        return 1/config['data_rate'] + comm_time + proc_time
    
    def _calculate_traffic_disruption_v2(self, config: Dict) -> float:
        """Enhanced traffic disruption model"""
        base_hours = 4.0
        inspections_year = 365 / config['inspection_cycle']
        
        speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
        
        if speed > 0:
            speed_factor = 80 / speed if speed < 80 else 1.0
            disruption = base_hours * speed_factor * inspections_year
        else:
            disruption = 0.1 * inspections_year
        
        # Traffic impact factors
        default_lane_closure = getattr(self.config, 'default_lane_closure_ratio', 0.3)
        traffic_volume = getattr(self.config, 'traffic_volume_hourly', 2000)
        
        lane_factor = 1 + default_lane_closure
        traffic_factor = traffic_volume / 1000
        
        return disruption * lane_factor * traffic_factor
    
    def _calculate_environmental_impact_v2(self, config: Dict) -> float:
        """Enhanced environmental impact - carbon footprint"""
        total_power_w = 0
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                power = self._query_property(config[comp], 'hasEnergyConsumptionW', 0)
                total_power_w += power if power else 0
        
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        
        if coverage > 0:
            sensor_hours = (self.config.road_network_length_km / coverage) * \
                         (365 / config['inspection_cycle']) * 8
            vehicle_km = self.config.road_network_length_km * (365 / config['inspection_cycle'])
            vehicle_kwh = vehicle_km * 0.8
        else:
            sensor_hours = 365 * 24
            vehicle_kwh = 0
        
        backend_hours = 365 * 24
        
        total_kwh = (total_power_w * 0.3 * sensor_hours + 
                    total_power_w * 0.7 * backend_hours) / 1000 + vehicle_kwh
        
        # Carbon emissions
        carbon_intensity = getattr(self.config, 'carbon_intensity_factor', 0.417)
        return total_kwh * carbon_intensity
    
    def _calculate_system_reliability_v2(self, config: Dict) -> float:
        """Enhanced reliability with redundancy factors"""
        inverse_mtbf = 0
        
        redundancy_multipliers = getattr(self.config, 'redundancy_multipliers', {
            'Cloud': 10.0, 'OnPremise': 1.5, 'Edge': 2.0, 'Hybrid': 5.0
        })
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                base_mtbf = self._query_property(config[comp], 'hasMTBFHours', 10000)
                
                # Apply redundancy
                comp_type = str(config[comp]).split('#')[-1]
                redundancy = 1.0
                for red_type, mult in redundancy_multipliers.items():
                    if red_type in comp_type:
                        redundancy = mult
                        break
                
                effective_mtbf = base_mtbf * redundancy
                if effective_mtbf > 0:
                    inverse_mtbf += 1 / effective_mtbf
        
        return inverse_mtbf if inverse_mtbf > 0 else 1e-6
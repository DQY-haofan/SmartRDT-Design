#!/usr/bin/env python3
"""
Enhanced Ontology-Driven Multi-Objective Optimization Framework for 
Road Maintenance Digital Twin (RMTwin) Configuration

Version: 8.0 - Academic Edition for Automation in Construction
Features: 6 objectives including sustainability metrics, parallel computing, 
         caching, and publication-quality visualizations

Author: [Your Name]
Date: June 2025
"""

import os
import sys
import json
import logging
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Optimization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Semantic Web
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
import pySHACL

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND GLOBAL SETTINGS
# ============================================================================

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 14,                # Increased base font size
    'axes.titlesize': 16,           # Title font size
    'axes.labelsize': 14,           # Axis label font size
    'xtick.labelsize': 12,          # X-tick label size
    'ytick.labelsize': 12,          # Y-tick label size
    'legend.fontsize': 12,          # Legend font size
    'figure.titlesize': 18,         # Figure title size
    'font.family': 'serif',         # Use serif font for academic look
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,           # Set True if LaTeX is available
    'figure.dpi': 100,              # Display DPI
    'savefig.dpi': 300,            # Save DPI for publication
    'savefig.bbox': 'tight',        # Tight bounding box
    'lines.linewidth': 2,           # Line width
    'lines.markersize': 8,          # Marker size
})

# Configure seaborn for better aesthetics
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rmtwin_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Namespaces
RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")

# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class OptimizationConfig:
    """Central configuration for the optimization framework"""
    
    # File paths
    sensor_csv: str = 'sensors_data_enhanced.txt'
    algorithm_csv: str = 'algorithms_data_enhanced.txt'
    infrastructure_csv: str = 'infrastructure_data_enhanced.txt'
    cost_benefit_csv: str = 'cost_benefit_data.txt'
    
    # Network parameters
    road_network_length_km: float = 100.0
    planning_horizon_years: int = 10
    
    # Constraints
    budget_cap_usd: float = 1_000_000
    min_recall_threshold: float = 0.80
    max_latency_seconds: float = 60.0
    max_disruption_hours: float = 100.0
    max_energy_kwh_year: float = 50_000
    min_mtbf_hours: float = 5_000
    
    # Algorithm parameters
    population_size: int = 150
    n_generations: int = 200
    n_objectives: int = 6
    crossover_prob: float = 0.9
    crossover_eta: float = 15
    mutation_eta: float = 20
    
    # Parallel computing
    n_processes: int = mp.cpu_count() - 1
    use_parallel: bool = True
    cache_size: int = 10000
    
    # Visualization
    output_dir: str = './results'
    figure_format: List[str] = field(default_factory=lambda: ['png', 'pdf', 'svg'])
    color_palette: str = 'viridis'
    
    # Weights for weighted sum (if needed)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'cost': 0.25,
        'performance': 0.20,
        'latency': 0.15,
        'disruption': 0.10,
        'environmental': 0.20,
        'reliability': 0.10
    })
    
    # Sustainability parameters
    carbon_intensity_kwh: float = 0.4  # kg CO2 per kWh
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath: str):
        """Save configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationConfig':
        """Load configuration from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

# ============================================================================
# ENHANCED ONTOLOGY POPULATOR
# ============================================================================

class EnhancedOntologyPopulator:
    """Populates RDTcO-Maint ontology with enhanced sensor, algorithm, and infrastructure data"""
    
    def __init__(self):
        self.g = Graph()
        self.g.bind("rdtco", RDTCO)
        self.g.bind("ex", EX)
        self._setup_base_ontology()
        
    def _setup_base_ontology(self):
        """Set up base ontology structure"""
        # Define core classes
        self.g.add((RDTCO.DigitalTwinConfiguration, RDF.type, OWL.Class))
        self.g.add((RDTCO.SensorSystem, RDF.type, OWL.Class))
        self.g.add((RDTCO.Algorithm, RDF.type, OWL.Class))
        self.g.add((RDTCO.StorageSystem, RDF.type, OWL.Class))
        self.g.add((RDTCO.CommunicationSystem, RDF.type, OWL.Class))
        self.g.add((RDTCO.ComputeDeployment, RDF.type, OWL.Class))
        
        # Define properties with domains and ranges
        properties = [
            ('hasInitialCostUSD', 'Initial cost in USD', XSD.decimal),
            ('hasOperationalCostUSDPerDay', 'Daily operational cost', XSD.decimal),
            ('hasEnergyConsumptionW', 'Energy consumption in watts', XSD.decimal),
            ('hasMTBFHours', 'Mean time between failures in hours', XSD.decimal),
            ('hasOperatorSkillLevel', 'Required operator skill level', XSD.string),
            ('hasCalibrationFreqMonths', 'Calibration frequency in months', XSD.decimal),
            ('hasDataAnnotationCostUSD', 'Data annotation cost per image', XSD.decimal),
            ('hasModelRetrainingFreqMonths', 'Model retraining frequency', XSD.decimal),
            ('hasExplainabilityScore', 'Model explainability score 1-5', XSD.integer),
            ('hasIntegrationComplexity', 'Integration complexity 1-5', XSD.integer),
            ('hasCybersecurityVulnerability', 'Security vulnerability 1-5', XSD.integer),
        ]
        
        for prop_name, comment, range_type in properties:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))
            self.g.add((prop_uri, RDFS.range, range_type))
            
    def populate_from_csvs(self, sensor_csv: str, algorithm_csv: str, 
                          infrastructure_csv: str, cost_csv: str) -> Graph:
        """Populate ontology from enhanced CSV files"""
        logger.info("Populating ontology from enhanced CSV files...")
        
        # Load and process each CSV
        self._load_sensors(sensor_csv)
        self._load_algorithms(algorithm_csv)
        self._load_infrastructure(infrastructure_csv)
        self._load_cost_benefit(cost_csv)
        
        # Add SHACL constraints
        self._add_shacl_constraints()
        
        logger.info(f"Ontology populated with {len(self.g)} triples")
        return self.g
    
    def _load_sensors(self, filepath: str):
        """Load enhanced sensor data"""
        df = pd.read_csv(filepath)
        logger.info(f"Loading {len(df)} sensor instances...")
        
        for _, row in df.iterrows():
            sensor_uri = EX[row['Sensor_Instance_Name']]
            sensor_type = RDTCO[row['Sensor_RDF_Type']]
            
            self.g.add((sensor_uri, RDF.type, sensor_type))
            self.g.add((sensor_uri, RDFS.label, Literal(row['Sensor_Instance_Name'])))
            
            # Add enhanced properties
            self.g.add((sensor_uri, RDTCO.hasInitialCostUSD, 
                       Literal(row['Initial_Cost_USD'], datatype=XSD.decimal)))
            self.g.add((sensor_uri, RDTCO.hasOperationalCostUSDPerDay, 
                       Literal(row['Operational_Cost_USD_per_day'], datatype=XSD.decimal)))
            self.g.add((sensor_uri, RDTCO.hasEnergyConsumptionW, 
                       Literal(row['Energy_Consumption_W'], datatype=XSD.decimal)))
            self.g.add((sensor_uri, RDTCO.hasMTBFHours, 
                       Literal(row['MTBF_hours'], datatype=XSD.decimal)))
            self.g.add((sensor_uri, RDTCO.hasOperatorSkillLevel, 
                       Literal(row['Operator_Skill_Level'])))
            
            if pd.notna(row['Calibration_Freq_months']):
                self.g.add((sensor_uri, RDTCO.hasCalibrationFreqMonths, 
                           Literal(row['Calibration_Freq_months'], datatype=XSD.decimal)))
            
            # Add other properties
            self.g.add((sensor_uri, RDTCO.hasAccuracyRangeMM, 
                       Literal(row['Accuracy_Range_mm'], datatype=XSD.decimal)))
            self.g.add((sensor_uri, RDTCO.hasDataVolumeGBPerKm, 
                       Literal(row['Data_Volume_GB_per_km'], datatype=XSD.decimal)))
            self.g.add((sensor_uri, RDTCO.hasCoverageEfficiencyKmPerDay, 
                       Literal(row['Coverage_Efficiency_km_per_day'], datatype=XSD.decimal)))
    
    def _load_algorithms(self, filepath: str):
        """Load enhanced algorithm data"""
        df = pd.read_csv(filepath)
        logger.info(f"Loading {len(df)} algorithm instances...")
        
        for _, row in df.iterrows():
            algo_uri = EX[row['Algorithm_Instance_Name']]
            algo_type = RDTCO[row['Algorithm_RDF_Type']]
            
            self.g.add((algo_uri, RDF.type, algo_type))
            self.g.add((algo_uri, RDFS.label, Literal(row['Algorithm_Instance_Name'])))
            
            # Add performance metrics
            self.g.add((algo_uri, RDTCO.hasPrecision, 
                       Literal(row['Precision'], datatype=XSD.decimal)))
            self.g.add((algo_uri, RDTCO.hasRecall, 
                       Literal(row['Recall'], datatype=XSD.decimal)))
            self.g.add((algo_uri, RDTCO.hasFPS, 
                       Literal(row['FPS'], datatype=XSD.decimal)))
            
            # Add enhanced properties
            self.g.add((algo_uri, RDTCO.hasHardwareRequirement, 
                       Literal(row['Hardware_Requirement'])))
            self.g.add((algo_uri, RDTCO.hasDataAnnotationCostUSD, 
                       Literal(row['Data_Annotation_Cost_USD'], datatype=XSD.decimal)))
            self.g.add((algo_uri, RDTCO.hasModelRetrainingFreqMonths, 
                       Literal(row['Model_Retraining_Freq_months'], datatype=XSD.decimal)))
            self.g.add((algo_uri, RDTCO.hasExplainabilityScore, 
                       Literal(row['Explainability_Score'], datatype=XSD.integer)))
    
    def _load_infrastructure(self, filepath: str):
        """Load enhanced infrastructure data"""
        df = pd.read_csv(filepath)
        logger.info(f"Loading {len(df)} infrastructure instances...")
        
        for _, row in df.iterrows():
            comp_uri = EX[row['Component_Instance_Name']]
            comp_type = RDTCO[row['Component_RDF_Type']]
            
            self.g.add((comp_uri, RDF.type, comp_type))
            self.g.add((comp_uri, RDFS.label, Literal(row['Component_Instance_Name'])))
            
            # Add costs
            self.g.add((comp_uri, RDTCO.hasInitialCostUSD, 
                       Literal(row['Initial_Cost_USD'], datatype=XSD.decimal)))
            self.g.add((comp_uri, RDTCO.hasAnnualOpCostUSD, 
                       Literal(row['Annual_OpCost_USD'], datatype=XSD.decimal)))
            
            # Add enhanced properties
            if pd.notna(row['Energy_Consumption_W']):
                self.g.add((comp_uri, RDTCO.hasEnergyConsumptionW, 
                           Literal(row['Energy_Consumption_W'], datatype=XSD.decimal)))
            if pd.notna(row['MTBF_hours']):
                self.g.add((comp_uri, RDTCO.hasMTBFHours, 
                           Literal(row['MTBF_hours'], datatype=XSD.decimal)))
            if pd.notna(row['Integration_Complexity']):
                self.g.add((comp_uri, RDTCO.hasIntegrationComplexity, 
                           Literal(row['Integration_Complexity'], datatype=XSD.integer)))
            if pd.notna(row['Cybersecurity_Vulnerability']):
                self.g.add((comp_uri, RDTCO.hasCybersecurityVulnerability, 
                           Literal(row['Cybersecurity_Vulnerability'], datatype=XSD.integer)))
                           
    def _load_cost_benefit(self, filepath: str):
        """Load cost-benefit data"""
        df = pd.read_csv(filepath)
        logger.info(f"Loading {len(df)} cost-benefit entries...")
        
        # Store as configuration parameters
        for _, row in df.iterrows():
            param_uri = EX[f"Parameter_{row['Metric_Name']}"]
            self.g.add((param_uri, RDF.type, RDTCO.ConfigurationParameter))
            self.g.add((param_uri, RDFS.label, Literal(row['Metric_Name'])))
            self.g.add((param_uri, RDTCO.hasValue, Literal(row['Value'], datatype=XSD.decimal)))
            self.g.add((param_uri, RDTCO.hasUnit, Literal(row['Unit'])))
            
    def _add_shacl_constraints(self):
        """Add SHACL constraints for validation"""
        # This is a placeholder - implement specific SHACL shapes as needed
        pass

# ============================================================================
# SOLUTION MAPPER WITH CACHING
# ============================================================================

class CachedSolutionMapper:
    """Maps between optimization variables and RMTwin configurations with caching"""
    
    def __init__(self, ontology_graph: Graph):
        self.g = ontology_graph
        self._cache_components()
        
    @lru_cache(maxsize=1000)
    def _cache_components(self):
        """Cache all available components from ontology"""
        # Cache sensors
        self.sensors = []
        for s in self.g.subjects(RDF.type, None):
            for t in self.g.objects(s, RDF.type):
                if 'Sensor' in str(t) and 'System' in str(t):
                    self.sensors.append(str(s))
        
        # Cache algorithms
        self.algorithms = []
        for s in self.g.subjects(RDF.type, None):
            for t in self.g.objects(s, RDF.type):
                if 'Algorithm' in str(t):
                    self.algorithms.append(str(s))
        
        # Cache storage
        self.storage_systems = []
        for s in self.g.subjects(RDF.type, RDTCO.StorageSystem):
            self.storage_systems.append(str(s))
        
        # Cache communication
        self.comm_systems = []
        for s in self.g.subjects(RDF.type, RDTCO.CommunicationSystem):
            self.comm_systems.append(str(s))
        
        # Cache deployment
        self.deployments = []
        for s in self.g.subjects(RDF.type, RDTCO.ComputeDeployment):
            self.deployments.append(str(s))
            
        logger.info(f"Cached components: {len(self.sensors)} sensors, "
                   f"{len(self.algorithms)} algorithms, {len(self.storage_systems)} storage, "
                   f"{len(self.comm_systems)} communication, {len(self.deployments)} deployment")
    
    @lru_cache(maxsize=10000)
    def decode_solution(self, x: np.ndarray) -> Dict:
        """Decode solution vector to configuration with caching"""
        # Convert numpy array to tuple for hashing
        x_tuple = tuple(x)
        
        return {
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

# ============================================================================
# ENHANCED FITNESS EVALUATOR WITH PARALLEL PROCESSING
# ============================================================================

class ParallelFitnessEvaluator:
    """Evaluates fitness with 6 objectives including sustainability metrics"""
    
    def __init__(self, ontology_graph: Graph, config: OptimizationConfig):
        self.g = ontology_graph
        self.config = config
        self.mapper = CachedSolutionMapper(ontology_graph)
        
        # Cache for expensive queries
        self._query_cache = {}
        
        # Normalization parameters
        self.norm_params = {
            'cost': {'min': 100_000, 'max': 2_000_000},
            'recall': {'min': 0.0, 'max': 0.4},  # 1-recall
            'latency': {'min': 0.1, 'max': 300.0},
            'disruption': {'min': 0.0, 'max': 500.0},
            'environmental': {'min': 1_000, 'max': 100_000},  # kWh/year
            'reliability': {'min': 0.0, 'max': 0.001}  # 1/MTBF
        }
        
    @lru_cache(maxsize=10000)
    def _query_property(self, subject: str, predicate: str) -> Optional[float]:
        """Cached SPARQL query for properties"""
        query_key = f"{subject}_{predicate}"
        
        if query_key in self._query_cache:
            return self._query_cache[query_key]
        
        query = f"""
        SELECT ?value WHERE {{
            <{subject}> <{predicate}> ?value .
        }}
        """
        
        result = None
        for row in self.g.query(query):
            if row.value:
                result = float(row.value)
                break
                
        self._query_cache[query_key] = result
        return result
    
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of solutions in parallel"""
        n_solutions = len(X)
        objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 3))
        
        if self.config.use_parallel and n_solutions > 10:
            # Use parallel processing for large batches
            with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
                futures = {executor.submit(self._evaluate_single, x): i 
                          for i, x in enumerate(X)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    obj, const = future.result()
                    objectives[idx] = obj
                    constraints[idx] = const
        else:
            # Sequential processing for small batches
            for i, x in enumerate(X):
                objectives[i], constraints[i] = self._evaluate_single(x)
                
        return objectives, constraints
    
    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single solution"""
        config = self.mapper.decode_solution(x)
        
        # Calculate all 6 objectives
        f1 = self._calculate_total_cost(config)
        f2 = self._calculate_detection_performance(config)
        f3 = self._calculate_latency(config)
        f4 = self._calculate_traffic_disruption(config)
        f5 = self._calculate_environmental_impact(config)
        f6 = self._calculate_system_reliability(config)
        
        # Normalize objectives
        objectives = np.array([
            self._normalize(f1, 'cost'),
            self._normalize(f2, 'recall'),
            self._normalize(f3, 'latency'),
            self._normalize(f4, 'disruption'),
            self._normalize(f5, 'environmental'),
            self._normalize(f6, 'reliability')
        ])
        
        # Calculate constraints
        recall = 1 - f2
        g1 = f3 - self.config.max_latency_seconds
        g2 = self.config.min_recall_threshold - recall
        g3 = f1 - self.config.budget_cap_usd
        
        constraints = np.array([g1, g2, g3])
        
        return objectives, constraints
    
    def _normalize(self, value: float, objective: str) -> float:
        """Normalize objective value to [0, 1]"""
        min_val = self.norm_params[objective]['min']
        max_val = self.norm_params[objective]['max']
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def _calculate_total_cost(self, config: Dict) -> float:
        """Calculate total cost over planning horizon (f1)"""
        total_cost = 0
        
        # Initial costs
        for component in ['sensor', 'storage', 'communication', 'deployment']:
            if component in config:
                cost = self._query_property(config[component], str(RDTCO.hasInitialCostUSD))
                if cost:
                    total_cost += cost
        
        # Operational costs
        sensor_op_cost = self._query_property(config['sensor'], 
                                            str(RDTCO.hasOperationalCostUSDPerDay))
        if sensor_op_cost:
            inspections_per_year = 365 / config['inspection_cycle']
            days_per_inspection = self.config.road_network_length_km / 80  # Assume 80 km/day
            annual_sensor_cost = sensor_op_cost * days_per_inspection * inspections_per_year
            total_cost += annual_sensor_cost * self.config.planning_horizon_years
        
        # Crew costs with skill multiplier
        skill_level = self._query_property(config['sensor'], str(RDTCO.hasOperatorSkillLevel))
        skill_multiplier = {'Basic': 1.0, 'Intermediate': 1.5, 'Expert': 2.0}.get(
            skill_level, 1.0) if skill_level else 1.0
        
        crew_daily_cost = config['crew_size'] * 1000 * skill_multiplier  # $1000/person/day base
        crew_annual_cost = crew_daily_cost * days_per_inspection * inspections_per_year
        total_cost += crew_annual_cost * self.config.planning_horizon_years
        
        # Data annotation costs
        annotation_cost = self._query_property(config['algorithm'], 
                                             str(RDTCO.hasDataAnnotationCostUSD))
        if annotation_cost:
            total_annotation = annotation_cost * 10000  # Assume 10k images
            total_cost += total_annotation
        
        # Model retraining costs
        retrain_freq = self._query_property(config['algorithm'], 
                                          str(RDTCO.hasModelRetrainingFreqMonths))
        if retrain_freq and retrain_freq > 0:
            retrainings = (self.config.planning_horizon_years * 12) / retrain_freq
            total_cost += retrainings * 5000  # $5k per retraining
        
        return total_cost
    
    def _calculate_detection_performance(self, config: Dict) -> float:
        """Calculate 1 - recall (f2)"""
        base_recall = self._query_property(config['algorithm'], str(RDTCO.hasRecall))
        if not base_recall:
            base_recall = 0.7  # Default
        
        # Adjust for sensor accuracy
        sensor_accuracy = self._query_property(config['sensor'], 
                                             str(RDTCO.hasAccuracyRangeMM))
        if sensor_accuracy:
            accuracy_factor = 1 - (sensor_accuracy / 100)  # Better accuracy -> higher factor
            base_recall *= (0.8 + 0.2 * accuracy_factor)
        
        # Adjust for LOD
        lod_factors = {'Micro': 1.1, 'Meso': 1.0, 'Macro': 0.9}
        base_recall *= lod_factors.get(config['geo_lod'], 1.0)
        
        # Adjust for detection threshold
        threshold_factor = 1 - abs(config['detection_threshold'] - 0.5) * 0.2
        base_recall *= threshold_factor
        
        return 1 - min(base_recall, 0.99)  # Return 1-recall for minimization
    
    def _calculate_latency(self, config: Dict) -> float:
        """Calculate data-to-decision latency in seconds (f3)"""
        # Data acquisition time
        data_rate = config['data_rate']
        acq_time = 1 / data_rate if data_rate > 0 else 1.0
        
        # Data volume
        data_volume = self._query_property(config['sensor'], 
                                         str(RDTCO.hasDataVolumeGBPerKm))
        if not data_volume:
            data_volume = 1.0  # Default 1 GB/km
        
        # Communication time
        comm_bandwidth = {'5G_Network': 1000, 'LoRaWAN': 0.05, 
                         'Fiber_Optic': 10000, '4G_LTE': 100}.get(
                         config['communication'].split('/')[-1], 100)
        
        comm_time = (data_volume * 1000) / comm_bandwidth  # Convert GB to MB
        
        # Processing time
        algo_fps = self._query_property(config['algorithm'], str(RDTCO.hasFPS))
        if algo_fps and algo_fps > 0:
            proc_time = 1 / algo_fps
        else:
            proc_time = 0.1  # Default
        
        # Deployment factor
        deploy_factors = {'Edge_Computing': 1.5, 'Cloud_Computing': 1.0, 
                         'Hybrid_Edge_Cloud': 1.2}
        deploy_factor = deploy_factors.get(config['deployment'].split('/')[-1], 1.0)
        proc_time *= deploy_factor
        
        return acq_time + comm_time + proc_time
    
    def _calculate_traffic_disruption(self, config: Dict) -> float:
        """Calculate traffic disruption in hours (f4)"""
        # Base disruption per inspection
        base_disruption = 4.0  # hours
        
        # Adjust for sensor type
        sensor_speed = self._query_property(config['sensor'], 
                                          str(RDTCO.hasOperatingSpeedKmh))
        if sensor_speed and sensor_speed > 0:
            # Faster sensors -> less disruption
            speed_factor = 80 / sensor_speed  # Normalized to 80 km/h
            base_disruption *= speed_factor
        
        # Total annual disruption
        inspections_per_year = 365 / config['inspection_cycle']
        annual_disruption = base_disruption * inspections_per_year
        
        # Adjust for time of day (simplified)
        # Assume 30% of inspections can be done at night with 0.3x impact
        annual_disruption *= 0.79  # Weighted average
        
        return annual_disruption
    
    def _calculate_environmental_impact(self, config: Dict) -> float:
        """Calculate environmental impact in kWh/year (f5)"""
        total_energy_w = 0
        
        # Component energy consumption
        components = ['sensor', 'storage', 'communication', 'deployment']
        for comp in components:
            if comp in config:
                energy = self._query_property(config[comp], 
                                            str(RDTCO.hasEnergyConsumptionW))
                if energy:
                    total_energy_w += energy
        
        # Calculate operational hours
        inspections_per_year = 365 / config['inspection_cycle']
        coverage_efficiency = self._query_property(config['sensor'], 
                                                 str(RDTCO.hasCoverageEfficiencyKmPerDay))
        if not coverage_efficiency:
            coverage_efficiency = 50  # Default
            
        days_per_inspection = self.config.road_network_length_km / coverage_efficiency
        sensor_hours = days_per_inspection * inspections_per_year * 8  # 8 hours/day
        
        # Backend runs 24/7
        backend_hours = 365 * 24
        
        # Calculate total energy
        sensor_energy_kwh = (total_energy_w * 0.3 * sensor_hours) / 1000
        backend_energy_kwh = (total_energy_w * 0.7 * backend_hours) / 1000
        
        # Add vehicle emissions (converted to kWh equivalent)
        vehicle_km = self.config.road_network_length_km * inspections_per_year
        vehicle_kwh_equiv = vehicle_km * 0.8  # 0.8 kWh/km equivalent
        
        return sensor_energy_kwh + backend_energy_kwh + vehicle_kwh_equiv
    
    def _calculate_system_reliability(self, config: Dict) -> float:
        """Calculate system reliability as 1/MTBF (f6)"""
        inverse_mtbf_sum = 0
        
        # Series reliability model: 1/MTBF_system = sum(1/MTBF_i)
        components = ['sensor', 'storage', 'communication', 'deployment']
        for comp in components:
            if comp in config:
                mtbf = self._query_property(config[comp], str(RDTCO.hasMTBFHours))
                if mtbf and mtbf > 0:
                    inverse_mtbf_sum += 1 / mtbf
        
        # Avoid division by zero
        if inverse_mtbf_sum == 0:
            return 1.0  # Poor reliability
            
        return inverse_mtbf_sum

# ============================================================================
# ENHANCED OPTIMIZATION PROBLEM
# ============================================================================

class EnhancedRMTwinProblem(Problem):
    """Multi-objective optimization problem with 6 objectives"""
    
    def __init__(self, ontology_graph: Graph, config: OptimizationConfig):
        self.g = ontology_graph
        self.config = config
        self.evaluator = ParallelFitnessEvaluator(ontology_graph, config)
        
        # Variable bounds
        xl = np.zeros(11)
        xu = np.ones(11)
        
        super().__init__(
            n_var=11,
            n_obj=6,
            n_constr=3,
            xl=xl,
            xu=xu
        )
        
        logger.info(f"Initialized optimization problem with {self.n_obj} objectives")
        
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population"""
        objectives, constraints = self.evaluator.evaluate_batch(X)
        out["F"] = objectives
        out["G"] = constraints

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

class EnhancedVisualizer:
    """Create publication-quality visualizations for 6-objective optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self._setup_directories()
        
    def _setup_directories(self):
        """Create output directories"""
        for fmt in self.config.figure_format:
            (self.output_dir / fmt).mkdir(parents=True, exist_ok=True)
            
    def create_all_visualizations(self, df: pd.DataFrame, res: Any):
        """Create all visualizations"""
        logger.info("Creating enhanced visualizations...")
        
        # 1. Parallel coordinates plot (best for 6D)
        self.create_parallel_coordinates(df)
        
        # 2. Scatter matrix
        self.create_scatter_matrix(df)
        
        # 3. Story-based 3D plots
        self.create_story_3d_plots(df)
        
        # 4. Interactive Plotly visualizations
        self.create_interactive_parallel_coordinates(df)
        
        # 5. Radar charts for selected solutions
        self.create_radar_charts(df)
        
        # 6. Convergence analysis
        self.create_convergence_plot(res)
        
        # 7. Decision variable impact
        self.create_decision_variable_analysis(df)
        
        # 8. Sustainability dashboard
        self.create_sustainability_dashboard(df)
        
    def create_parallel_coordinates(self, df: pd.DataFrame):
        """Create parallel coordinates plot for 6D Pareto front"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Prepare data - normalize for visualization
        plot_data = df[['solution_id', 'f1_total_cost_USD', 'detection_recall',
                       'f3_latency_seconds', 'f4_traffic_disruption_hours',
                       'f5_environmental_impact_kWh_year', 'system_MTBF_hours']].copy()
        
        # Normalize columns
        for col in plot_data.columns[1:]:
            if col in ['detection_recall', 'system_MTBF_hours']:
                # These are "larger is better"
                plot_data[col] = (plot_data[col] - plot_data[col].min()) / \
                               (plot_data[col].max() - plot_data[col].min())
            else:
                # These are "smaller is better"
                plot_data[col] = 1 - (plot_data[col] - plot_data[col].min()) / \
                                   (plot_data[col].max() - plot_data[col].min())
        
        # Create parallel coordinates
        parallel_coordinates(plot_data, 'solution_id', 
                           colormap=self.config.color_palette, 
                           alpha=0.5, linewidth=2)
        
        # Customize
        ax.set_xlabel('')
        ax.set_ylabel('Normalized Performance\n(1 = Best, 0 = Worst)', fontsize=14)
        ax.set_title('6-Dimensional Pareto Front Visualization\nParallel Coordinates Plot', 
                    fontsize=18, pad=20)
        
        # Set custom labels
        labels = ['Total Cost\n($USD)', 'Detection\nRecall', 'Latency\n(seconds)',
                 'Traffic\nDisruption\n(hours)', 'Environmental\nImpact\n(kWh/year)',
                 'System\nReliability\n(MTBF hours)']
        ax.set_xticklabels(labels, fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)
        
        # Remove legend (too cluttered)
        ax.get_legend().remove()
        
        # Add annotation
        ax.text(0.02, 0.98, f'n = {len(df)} Pareto-optimal solutions',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure(fig, 'parallel_coordinates_6d')
        
    def create_scatter_matrix(self, df: pd.DataFrame):
        """Create scatter matrix for pairwise relationships"""
        # Select and rename columns
        plot_df = df[['f1_total_cost_USD', 'detection_recall', 
                     'f3_latency_seconds', 'f4_traffic_disruption_hours',
                     'f5_environmental_impact_kWh_year', 'system_MTBF_hours']].copy()
        
        plot_df.columns = ['Cost\n(k$)', 'Recall', 'Latency\n(s)', 
                          'Disruption\n(h)', 'Energy\n(MWh/y)', 'MTBF\n(k hours)']
        
        # Scale for better display
        plot_df['Cost\n(k$)'] /= 1000
        plot_df['Energy\n(MWh/y)'] /= 1000
        plot_df['MTBF\n(k hours)'] /= 1000
        
        # Create figure
        g = sns.pairplot(plot_df, diag_kind='kde', 
                        plot_kws={'alpha': 0.6, 's': 50},
                        diag_kws={'linewidth': 2})
        
        # Customize
        g.fig.suptitle('6-Objective Pairwise Trade-off Analysis', 
                      fontsize=18, y=1.02)
        
        # Adjust layout
        g.fig.set_size_inches(16, 16)
        
        self._save_figure(g.fig, 'scatter_matrix_6d')
        
    def create_story_3d_plots(self, df: pd.DataFrame):
        """Create narrative 3D visualizations"""
        # Story 1: Classic Performance Trade-off
        fig1 = plt.figure(figsize=(14, 10))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # Color by environmental impact
        colors = df['f5_environmental_impact_kWh_year']
        scatter1 = ax1.scatter(df['f1_total_cost_USD']/1000,
                             df['detection_recall'],
                             df['f3_latency_seconds'],
                             c=colors, cmap='RdYlGn_r',
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Total Cost (k$)', fontsize=14, labelpad=10)
        ax1.set_ylabel('Detection Recall', fontsize=14, labelpad=10)
        ax1.set_zlabel('Latency (seconds)', fontsize=14, labelpad=10)
        ax1.set_title('Performance-Cost-Efficiency Trade-off\n(Color: Environmental Impact)',
                     fontsize=16, pad=20)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.8)
        cbar1.set_label('Annual Energy (kWh)', fontsize=12, rotation=270, labelpad=20)
        
        # Improve viewing angle
        ax1.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        self._save_figure(fig1, 'story_performance_tradeoff_3d')
        
        # Story 2: Sustainability Focus
        fig2 = plt.figure(figsize=(14, 10))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # Color by detection recall
        colors2 = df['detection_recall']
        scatter2 = ax2.scatter(df['f1_total_cost_USD']/1000,
                             df['f5_environmental_impact_kWh_year']/1000,
                             df['system_MTBF_hours']/1000,
                             c=colors2, cmap='viridis',
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Total Cost (k$)', fontsize=14, labelpad=10)
        ax2.set_ylabel('Environmental Impact\n(MWh/year)', fontsize=14, labelpad=10)
        ax2.set_zlabel('System Reliability\n(k hours MTBF)', fontsize=14, labelpad=10)
        ax2.set_title('Sustainability-Reliability-Cost Trade-off\n(Color: Detection Performance)',
                     fontsize=16, pad=20)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8)
        cbar2.set_label('Detection Recall', fontsize=12, rotation=270, labelpad=20)
        
        ax2.view_init(elev=25, azim=135)
        
        plt.tight_layout()
        self._save_figure(fig2, 'story_sustainability_tradeoff_3d')
        
    def create_interactive_parallel_coordinates(self, df: pd.DataFrame):
        """Create interactive Plotly parallel coordinates"""
        # Prepare data
        dimensions = []
        
        # Add each objective as a dimension
        objectives = [
            ('f1_total_cost_USD', 'Total Cost ($)', False),
            ('detection_recall', 'Detection Recall', True),
            ('f3_latency_seconds', 'Latency (s)', False),
            ('f4_traffic_disruption_hours', 'Traffic Disruption (h)', False),
            ('f5_environmental_impact_kWh_year', 'Environmental Impact (kWh/y)', False),
            ('system_MTBF_hours', 'System MTBF (hours)', True)
        ]
        
        for col, label, reverse in objectives:
            dimensions.append(
                dict(
                    range=[df[col].min(), df[col].max()],
                    label=label,
                    values=df[col],
                    # Reverse scale for "larger is better" metrics
                    constraintrange=[df[col].max(), df[col].min()] if reverse else None
                )
            )
        
        # Create figure
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df['f1_total_cost_USD'],
                    colorscale='Viridis',
                    showscale=True,
                    reversescale=True,
                    cmin=df['f1_total_cost_USD'].min(),
                    cmax=df['f1_total_cost_USD'].max(),
                    colorbar=dict(
                        title='Total Cost ($)',
                        titleside='right',
                        tickmode='linear',
                        tick0=df['f1_total_cost_USD'].min(),
                        dtick=(df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min()) / 5
                    )
                ),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title={
                'text': 'Interactive 6D Pareto Front Explorer<br><sub>Brush axes to filter solutions</sub>',
                'font': {'size': 20}
            },
            width=1400,
            height=800,
            font={'size': 14}
        )
        
        # Save as HTML
        fig.write_html(str(self.output_dir / 'interactive_parallel_coordinates.html'))
        logger.info("Created interactive parallel coordinates plot")
        
    def create_radar_charts(self, df: pd.DataFrame):
        """Create radar charts for extreme solutions"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                               subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        # Define extreme solutions
        extremes = [
            ('Lowest Cost', df['f1_total_cost_USD'].idxmin()),
            ('Highest Recall', df['detection_recall'].idxmax()),
            ('Lowest Latency', df['f3_latency_seconds'].idxmin()),
            ('Lowest Disruption', df['f4_traffic_disruption_hours'].idxmin()),
            ('Most Sustainable', df['f5_environmental_impact_kWh_year'].idxmin()),
            ('Most Reliable', df['system_MTBF_hours'].idxmax())
        ]
        
        # Objectives for radar
        objectives = ['Cost', 'Recall', 'Latency', 'Disruption', 'Energy', 'Reliability']
        
        for idx, (title, sol_idx) in enumerate(extremes):
            ax = axes[idx]
            
            # Get normalized values
            values = []
            solution = df.iloc[sol_idx]
            
            # Normalize each objective to [0, 1] where 1 is best
            values.append(1 - (solution['f1_total_cost_USD'] - df['f1_total_cost_USD'].min()) /
                         (df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min()))
            values.append((solution['detection_recall'] - df['detection_recall'].min()) /
                         (df['detection_recall'].max() - df['detection_recall'].min()))
            values.append(1 - (solution['f3_latency_seconds'] - df['f3_latency_seconds'].min()) /
                         (df['f3_latency_seconds'].max() - df['f3_latency_seconds'].min()))
            values.append(1 - (solution['f4_traffic_disruption_hours'] - df['f4_traffic_disruption_hours'].min()) /
                         (df['f4_traffic_disruption_hours'].max() - df['f4_traffic_disruption_hours'].min()))
            values.append(1 - (solution['f5_environmental_impact_kWh_year'] - df['f5_environmental_impact_kWh_year'].min()) /
                         (df['f5_environmental_impact_kWh_year'].max() - df['f5_environmental_impact_kWh_year'].min()))
            values.append((solution['system_MTBF_hours'] - df['system_MTBF_hours'].min()) /
                         (df['system_MTBF_hours'].max() - df['system_MTBF_hours'].min()))
            
            # Complete the circle
            values += values[:1]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(objectives, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(f'{title}\n{solution["sensor"].split("/")[-1]}',
                        fontsize=12, pad=20)
            ax.grid(True, alpha=0.3)
            
        plt.suptitle('Radar Charts of Extreme Solutions\n(1 = Best, 0 = Worst)',
                    fontsize=16)
        plt.tight_layout()
        self._save_figure(fig, 'radar_charts_extreme_solutions')
        
    def create_convergence_plot(self, res):
        """Create convergence analysis plot"""
        if not hasattr(res, 'history') or not res.history:
            logger.warning("No convergence history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract history
        n_evals = []
        hypervolumes = []
        n_nondominated = []
        
        for entry in res.history:
            n_evals.append(entry.evaluator.n_eval)
            
            # Calculate hypervolume (simplified)
            F = entry.pop.get("F")
            if F is not None and len(F) > 0:
                # Normalize objectives
                F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
                # Simple hypervolume approximation
                hv = np.prod(1 - F_norm.min(axis=0))
                hypervolumes.append(hv)
                
                # Count non-dominated solutions
                nds = NonDominatedSorting()
                fronts = nds.do(F)
                n_nondominated.append(len(fronts[0]))
        
        # Plot hypervolume
        ax1.plot(n_evals, hypervolumes, 'b-', linewidth=2)
        ax1.set_xlabel('Function Evaluations', fontsize=14)
        ax1.set_ylabel('Hypervolume Indicator', fontsize=14)
        ax1.set_title('Convergence Analysis', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Plot number of non-dominated solutions
        ax2.plot(n_evals, n_nondominated, 'r-', linewidth=2)
        ax2.set_xlabel('Function Evaluations', fontsize=14)
        ax2.set_ylabel('Number of Non-dominated Solutions', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'convergence_analysis')
        
    def create_decision_variable_analysis(self, df: pd.DataFrame):
        """Analyze impact of decision variables"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        # Decision variables to analyze
        variables = [
            ('sensor', 'Sensor Type'),
            ('algorithm', 'Algorithm'),
            ('storage', 'Storage System'),
            ('communication', 'Communication'),
            ('deployment', 'Deployment'),
            ('geometric_LOD', 'Geometric LOD'),
            ('condition_LOD', 'Condition LOD'),
            ('crew_size', 'Crew Size'),
            ('inspection_cycle_days', 'Inspection Cycle'),
            ('data_rate_Hz', 'Data Rate (Hz)'),
            ('detection_threshold', 'Detection Threshold')
        ]
        
        for idx, (var, label) in enumerate(variables):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            if var in df.columns:
                if df[var].dtype == 'object' or df[var].nunique() < 10:
                    # Categorical variable - use box plot
                    data_to_plot = []
                    labels_to_plot = []
                    
                    for value in df[var].unique():
                        subset = df[df[var] == value]['f1_total_cost_USD'] / 1000
                        if len(subset) > 0:
                            data_to_plot.append(subset)
                            labels_to_plot.append(str(value).split('/')[-1][:15])
                    
                    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.set_xlabel(label, fontsize=12)
                    ax.set_ylabel('Total Cost (k$)', fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    # Continuous variable - use scatter plot
                    ax.scatter(df[var], df['f1_total_cost_USD']/1000, 
                              alpha=0.6, s=30)
                    ax.set_xlabel(label, fontsize=12)
                    ax.set_ylabel('Total Cost (k$)', fontsize=12)
                
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(variables), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Decision Variable Impact Analysis', fontsize=18)
        plt.tight_layout()
        self._save_figure(fig, 'decision_variable_analysis')
        
    def create_sustainability_dashboard(self, df: pd.DataFrame):
        """Create comprehensive sustainability analysis dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Carbon footprint distribution
        ax1 = fig.add_subplot(gs[0, 0])
        carbon_footprint = df['f5_environmental_impact_kWh_year'] * self.config.carbon_intensity_kwh / 1000
        ax1.hist(carbon_footprint, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Annual CO₂ Emissions (tons)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Carbon Footprint Distribution', fontsize=14)
        ax1.axvline(carbon_footprint.mean(), color='red', linestyle='--', 
                   label=f'Mean: {carbon_footprint.mean():.1f} tons')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy vs Performance trade-off
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(df['f5_environmental_impact_kWh_year']/1000, 
                            df['detection_recall'],
                            c=df['f1_total_cost_USD']/1000, 
                            cmap='plasma', s=60, alpha=0.7)
        ax2.set_xlabel('Annual Energy (MWh)', fontsize=12)
        ax2.set_ylabel('Detection Recall', fontsize=12)
        ax2.set_title('Energy-Performance Trade-off', fontsize=14)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Total Cost (k$)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Sensor technology sustainability comparison
        ax3 = fig.add_subplot(gs[0, 2])
        sensor_energy = df.groupby('sensor')['f5_environmental_impact_kWh_year'].agg(['mean', 'std'])
        sensor_energy['sensor_short'] = [s.split('/')[-1][:20] for s in sensor_energy.index]
        sensor_energy = sensor_energy.sort_values('mean')
        
        ax3.barh(sensor_energy['sensor_short'], sensor_energy['mean']/1000, 
                xerr=sensor_energy['std']/1000, color='skyblue', capsize=5)
        ax3.set_xlabel('Average Annual Energy (MWh)', fontsize=12)
        ax3.set_title('Sensor Technology Energy Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Reliability vs Sustainability
        ax4 = fig.add_subplot(gs[1, :2])
        scatter2 = ax4.scatter(df['system_MTBF_hours']/8760,  # Convert to years
                             df['f5_environmental_impact_kWh_year']/1000,
                             c=df['f1_total_cost_USD']/1000,
                             cmap='viridis', s=80, alpha=0.7)
        ax4.set_xlabel('System MTBF (years)', fontsize=12)
        ax4.set_ylabel('Annual Energy (MWh)', fontsize=12)
        ax4.set_title('Reliability-Sustainability Trade-off', fontsize=14)
        cbar2 = plt.colorbar(scatter2, ax=ax4)
        cbar2.set_label('Total Cost (k$)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Pareto frontier for sustainability objectives
        ax5 = fig.add_subplot(gs[1, 2])
        # Find Pareto frontier for cost vs environmental impact
        pareto_mask = self._find_pareto_frontier_2d(
            df['f1_total_cost_USD'].values,
            df['f5_environmental_impact_kWh_year'].values
        )
        
        ax5.scatter(df['f1_total_cost_USD']/1000, 
                   df['f5_environmental_impact_kWh_year']/1000,
                   c='lightgray', s=50, alpha=0.5, label='All solutions')
        ax5.scatter(df.loc[pareto_mask, 'f1_total_cost_USD']/1000,
                   df.loc[pareto_mask, 'f5_environmental_impact_kWh_year']/1000,
                   c='red', s=100, alpha=0.8, label='Pareto optimal')
        ax5.set_xlabel('Total Cost (k$)', fontsize=12)
        ax5.set_ylabel('Annual Energy (MWh)', fontsize=12)
        ax5.set_title('Cost-Sustainability Pareto Front', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate statistics
        best_energy = df.loc[df['f5_environmental_impact_kWh_year'].idxmin()]
        best_cost = df.loc[df['f1_total_cost_USD'].idxmin()]
        avg_carbon = carbon_footprint.mean()
        
        summary_text = f"""
SUSTAINABILITY ANALYSIS SUMMARY

Best Environmental Performance:
  • Configuration: {best_energy['sensor'].split('/')[-1]} + {best_energy['algorithm'].split('/')[-1]}
  • Annual Energy: {best_energy['f5_environmental_impact_kWh_year']:,.0f} kWh ({best_energy['f5_environmental_impact_kWh_year']/1000:.1f} MWh)
  • CO₂ Emissions: {best_energy['f5_environmental_impact_kWh_year'] * self.config.carbon_intensity_kwh / 1000:.1f} tons/year
  • Detection Recall: {best_energy['detection_recall']:.3f}
  • Total Cost: ${best_energy['f1_total_cost_USD']:,.0f}

Most Cost-Effective Configuration:
  • Configuration: {best_cost['sensor'].split('/')[-1]} + {best_cost['algorithm'].split('/')[-1]}
  • Annual Energy: {best_cost['f5_environmental_impact_kWh_year']:,.0f} kWh
  • CO₂ Emissions: {best_cost['f5_environmental_impact_kWh_year'] * self.config.carbon_intensity_kwh / 1000:.1f} tons/year

Portfolio Statistics:
  • Average Annual Energy: {df['f5_environmental_impact_kWh_year'].mean():,.0f} kWh
  • Average CO₂ Emissions: {avg_carbon:.1f} tons/year
  • Energy Range: {df['f5_environmental_impact_kWh_year'].min():,.0f} - {df['f5_environmental_impact_kWh_year'].max():,.0f} kWh/year
  • Solutions with <10 tons CO₂/year: {(carbon_footprint < 10).sum()} ({(carbon_footprint < 10).sum()/len(df)*100:.1f}%)
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Sustainability Analysis Dashboard', fontsize=20, y=0.98)
        self._save_figure(fig, 'sustainability_dashboard')
        
    def _find_pareto_frontier_2d(self, x, y):
        """Find 2D Pareto frontier (both objectives to minimize)"""
        n = len(x)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if x[j] <= x[i] and y[j] <= y[i]:
                        if x[j] < x[i] or y[j] < y[i]:
                            pareto_mask[i] = False
                            break
        
        return pareto_mask
    
    def _save_figure(self, fig, name: str):
        """Save figure in multiple formats"""
        for fmt in self.config.figure_format:
            filepath = self.output_dir / fmt / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=300 if fmt == 'png' else None,
                       bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved figure: {name}")

# ============================================================================
# RESULTS MANAGER
# ============================================================================

class ResultsManager:
    """Manage optimization results and generate reports"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, res, evaluator: ParallelFitnessEvaluator) -> pd.DataFrame:
        """Save optimization results to files"""
        results = []
        
        if res is None or res.X is None:
            logger.warning("No solutions found!")
            return pd.DataFrame()
            
        # Ensure arrays are 2D
        if res.X.ndim == 1:
            res.X = res.X.reshape(1, -1)
            res.F = res.F.reshape(1, -1)
            res.G = res.G.reshape(1, -1) if res.G is not None else None
            
        # Process each solution
        for i, x in enumerate(res.X):
            config = evaluator.mapper.decode_solution(x)
            
            # Get raw objective values
            obj, const = evaluator._evaluate_single(x)
            
            # Denormalize objectives
            f1_raw = obj[0] * (evaluator.norm_params['cost']['max'] - 
                             evaluator.norm_params['cost']['min']) + \
                    evaluator.norm_params['cost']['min']
            f2_raw = obj[1] * (evaluator.norm_params['recall']['max'] - 
                             evaluator.norm_params['recall']['min']) + \
                    evaluator.norm_params['recall']['min']
            f3_raw = obj[2] * (evaluator.norm_params['latency']['max'] - 
                             evaluator.norm_params['latency']['min']) + \
                    evaluator.norm_params['latency']['min']
            f4_raw = obj[3] * (evaluator.norm_params['disruption']['max'] - 
                             evaluator.norm_params['disruption']['min']) + \
                    evaluator.norm_params['disruption']['min']
            f5_raw = obj[4] * (evaluator.norm_params['environmental']['max'] - 
                             evaluator.norm_params['environmental']['min']) + \
                    evaluator.norm_params['environmental']['min']
            f6_raw = obj[5] * (evaluator.norm_params['reliability']['max'] - 
                             evaluator.norm_params['reliability']['min']) + \
                    evaluator.norm_params['reliability']['min']
            
            recall = 1 - f2_raw
            system_mtbf = 1 / f6_raw if f6_raw > 0 else float('inf')
            
            # Build result row
            row = {
                'solution_id': i + 1,
                # Configuration details
                'sensor': config['sensor'].split('/')[-1],
                'data_rate_Hz': round(config['data_rate'], 2),
                'geometric_LOD': config['geo_lod'],
                'condition_LOD': config['cond_lod'],
                'algorithm': config['algorithm'].split('/')[-1],
                'detection_threshold': round(config['detection_threshold'], 3),
                'storage': config['storage'].split('/')[-1],
                'communication': config['communication'].split('/')[-1],
                'deployment': config['deployment'].split('/')[-1],
                'crew_size': config['crew_size'],
                'inspection_cycle_days': config['inspection_cycle'],
                # Raw objectives
                'f1_total_cost_USD': round(f1_raw, 2),
                'f2_one_minus_recall': round(f2_raw, 4),
                'f3_latency_seconds': round(f3_raw, 2),
                'f4_traffic_disruption_hours': round(f4_raw, 2),
                'f5_environmental_impact_kWh_year': round(f5_raw, 2),
                'f6_system_reliability_inverse_MTBF': round(f6_raw, 8),
                # Derived metrics
                'detection_recall': round(recall, 4),
                'system_MTBF_hours': round(system_mtbf, 0),
                'annual_cost_USD': round(f1_raw / self.config.planning_horizon_years, 2),
                'cost_per_km_year': round(f1_raw / self.config.planning_horizon_years / 
                                         self.config.road_network_length_km, 2),
                'carbon_footprint_tons_CO2_year': round(f5_raw * self.config.carbon_intensity_kwh / 1000, 2),
                # Constraints
                'is_feasible': np.all(const <= 0) if const is not None else True
            }
            
            results.append(row)
            
        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('f1_total_cost_USD')
        
        # Save to CSV
        df.to_csv(self.output_dir / 'pareto_solutions_6d.csv', index=False)
        logger.info(f"Saved {len(df)} Pareto-optimal solutions")
        
        # Save summary statistics
        self._save_summary(df)
        
        return df
        
    def _save_summary(self, df: pd.DataFrame):
        """Save optimization summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.to_dict(),
            'results': {
                'total_solutions': len(df),
                'feasible_solutions': df['is_feasible'].sum() if 'is_feasible' in df else len(df),
                'objective_statistics': {
                    'cost': {
                        'min': float(df['f1_total_cost_USD'].min()),
                        'max': float(df['f1_total_cost_USD'].max()),
                        'mean': float(df['f1_total_cost_USD'].mean()),
                        'std': float(df['f1_total_cost_USD'].std())
                    },
                    'recall': {
                        'min': float(df['detection_recall'].min()),
                        'max': float(df['detection_recall'].max()),
                        'mean': float(df['detection_recall'].mean()),
                        'std': float(df['detection_recall'].std())
                    },
                    'latency': {
                        'min': float(df['f3_latency_seconds'].min()),
                        'max': float(df['f3_latency_seconds'].max()),
                        'mean': float(df['f3_latency_seconds'].mean()),
                        'std': float(df['f3_latency_seconds'].std())
                    },
                    'environmental_impact': {
                        'min': float(df['f5_environmental_impact_kWh_year'].min()),
                        'max': float(df['f5_environmental_impact_kWh_year'].max()),
                        'mean': float(df['f5_environmental_impact_kWh_year'].mean()),
                        'std': float(df['f5_environmental_impact_kWh_year'].std())
                    },
                    'reliability_mtbf': {
                        'min': float(df['system_MTBF_hours'].min()),
                        'max': float(df['system_MTBF_hours'].max()),
                        'mean': float(df['system_MTBF_hours'].mean()),
                        'std': float(df['system_MTBF_hours'].std())
                    }
                }
            }
        }
        
        with open(self.output_dir / 'optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    def generate_report(self, df: pd.DataFrame):
        """Generate comprehensive analysis report"""
        report = f"""
═══════════════════════════════════════════════════════════════════════════════
    ENHANCED RMTWIN MULTI-OBJECTIVE OPTIMIZATION REPORT
    Automation in Construction - Academic Edition
═══════════════════════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Framework Version: 8.0

1. EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

This report presents the results of a 6-objective optimization for Road 
Maintenance Digital Twin (RMTwin) configuration, incorporating sustainability 
and reliability dimensions alongside traditional performance metrics.

Key Findings:
• Total Pareto-optimal solutions: {len(df)}
• Feasible solutions: {df['is_feasible'].sum() if 'is_feasible' in df else len(df)} ({df['is_feasible'].sum()/len(df)*100:.1f}%)
• Objective space coverage: 6-dimensional
• Computational efficiency: {self.config.n_generations} generations with {self.config.population_size} population size

2. OBJECTIVE SPACE ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

2.1 Objective Ranges and Statistics
-----------------------------------
┌─────────────────────────┬────────────────┬────────────────┬────────────────┐
│ Objective               │ Minimum        │ Maximum        │ Mean ± Std     │
├─────────────────────────┼────────────────┼────────────────┼────────────────┤
│ Total Cost ($)          │ {df['f1_total_cost_USD'].min():>14,.0f} │ {df['f1_total_cost_USD'].max():>14,.0f} │ {df['f1_total_cost_USD'].mean():>7,.0f} ± {df['f1_total_cost_USD'].std():>6,.0f} │
│ Detection Recall        │ {df['detection_recall'].min():>14.3f} │ {df['detection_recall'].max():>14.3f} │ {df['detection_recall'].mean():>7.3f} ± {df['detection_recall'].std():>6.3f} │
│ Latency (seconds)       │ {df['f3_latency_seconds'].min():>14.1f} │ {df['f3_latency_seconds'].max():>14.1f} │ {df['f3_latency_seconds'].mean():>7.1f} ± {df['f3_latency_seconds'].std():>6.1f} │
│ Disruption (hours/year) │ {df['f4_traffic_disruption_hours'].min():>14.1f} │ {df['f4_traffic_disruption_hours'].max():>14.1f} │ {df['f4_traffic_disruption_hours'].mean():>7.1f} ± {df['f4_traffic_disruption_hours'].std():>6.1f} │
│ Energy (kWh/year)       │ {df['f5_environmental_impact_kWh_year'].min():>14,.0f} │ {df['f5_environmental_impact_kWh_year'].max():>14,.0f} │ {df['f5_environmental_impact_kWh_year'].mean():>7,.0f} ± {df['f5_environmental_impact_kWh_year'].std():>6,.0f} │
│ MTBF (hours)           │ {df['system_MTBF_hours'].min():>14,.0f} │ {df['system_MTBF_hours'].max():>14,.0f} │ {df['system_MTBF_hours'].mean():>7,.0f} ± {df['system_MTBF_hours'].std():>6,.0f} │
└─────────────────────────┴────────────────┴────────────────┴────────────────┘

2.2 Extreme Solutions
---------------------
"""
        
        # Add extreme solutions
        extremes = [
            ('Lowest Cost', df['f1_total_cost_USD'].idxmin()),
            ('Highest Recall', df['detection_recall'].idxmax()),
            ('Lowest Latency', df['f3_latency_seconds'].idxmin()),
            ('Most Sustainable', df['f5_environmental_impact_kWh_year'].idxmin()),
            ('Most Reliable', df['system_MTBF_hours'].idxmax())
        ]
        
        for name, idx in extremes:
            sol = df.iloc[idx]
            report += f"""
{name}:
  • Configuration: {sol['sensor']} + {sol['algorithm']}
  • Cost: ${sol['f1_total_cost_USD']:,.0f} | Recall: {sol['detection_recall']:.3f} | Energy: {sol['f5_environmental_impact_kWh_year']:,.0f} kWh/year
"""
        
        report += f"""

3. SUSTAINABILITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

3.1 Environmental Impact Summary
--------------------------------
• Average Annual Energy Consumption: {df['f5_environmental_impact_kWh_year'].mean():,.0f} kWh
• Equivalent CO₂ Emissions: {df['carbon_footprint_tons_CO2_year'].mean():.1f} tons/year
• Energy Efficiency Range: {df['f5_environmental_impact_kWh_year'].min():,.0f} - {df['f5_environmental_impact_kWh_year'].max():,.0f} kWh/year
• Low-Carbon Solutions (<10 tons CO₂/year): {(df['carbon_footprint_tons_CO2_year'] < 10).sum()} ({(df['carbon_footprint_tons_CO2_year'] < 10).sum()/len(df)*100:.1f}%)

3.2 Technology Sustainability Ranking
-------------------------------------
"""
        
        # Technology rankings
        tech_energy = df.groupby('sensor')['f5_environmental_impact_kWh_year'].agg(['mean', 'count'])
        tech_energy = tech_energy.sort_values('mean')
        
        for i, (sensor, row) in enumerate(tech_energy.iterrows(), 1):
            report += f"{i}. {sensor}: {row['mean']:,.0f} kWh/year (n={row['count']})\n"
            
        report += f"""

4. RELIABILITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

4.1 System Reliability Overview
-------------------------------
• Average System MTBF: {df['system_MTBF_hours'].mean():,.0f} hours ({df['system_MTBF_hours'].mean()/8760:.1f} years)
• Reliability Range: {df['system_MTBF_hours'].min():,.0f} - {df['system_MTBF_hours'].max():,.0f} hours
• High-Reliability Solutions (>3 years MTBF): {(df['system_MTBF_hours'] > 26280).sum()} ({(df['system_MTBF_hours'] > 26280).sum()/len(df)*100:.1f}%)

4.2 Component Reliability Impact
--------------------------------
"""
        
        # Component reliability analysis
        components = ['sensor', 'storage', 'communication', 'deployment']
        for comp in components:
            comp_reliability = df.groupby(comp)['system_MTBF_hours'].mean().sort_values(ascending=False)
            report += f"\n{comp.capitalize()} Systems (by average MTBF):\n"
            for system, mtbf in comp_reliability.head(3).items():
                report += f"  • {system}: {mtbf:,.0f} hours\n"
                
        report += f"""

5. MULTI-OBJECTIVE TRADE-OFF INSIGHTS
═══════════════════════════════════════════════════════════════════════════════

5.1 Correlation Analysis
------------------------
Key relationships between objectives:
"""
        
        # Calculate correlations
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                   'f4_traffic_disruption_hours', 'f5_environmental_impact_kWh_year',
                   'system_MTBF_hours']
        
        corr_matrix = df[obj_cols].corr()
        
        # Find strong correlations
        for i in range(len(obj_cols)):
            for j in range(i+1, len(obj_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    obj1 = obj_cols[i].replace('_', ' ').replace('f1 ', '').replace('f3 ', '')
                    obj2 = obj_cols[j].replace('_', ' ').replace('f1 ', '').replace('f3 ', '')
                    direction = "positively" if corr > 0 else "negatively"
                    report += f"• {obj1} and {obj2} are {direction} correlated (r={corr:.2f})\n"
                    
        report += f"""

5.2 Configuration Recommendations
---------------------------------

Based on the 6-dimensional optimization results:

1. **For Cost-Conscious Deployments:**
   - Focus on configurations with total cost < ${df['f1_total_cost_USD'].quantile(0.25):,.0f}
   - Recommended sensor: {df[df['f1_total_cost_USD'] < df['f1_total_cost_USD'].quantile(0.25)]['sensor'].mode().values[0] if len(df[df['f1_total_cost_USD'] < df['f1_total_cost_USD'].quantile(0.25)]) > 0 else 'N/A'}
   - Trade-off: Lower detection performance (avg recall: {df[df['f1_total_cost_USD'] < df['f1_total_cost_USD'].quantile(0.25)]['detection_recall'].mean():.3f})

2. **For Sustainability-Focused Deployments:**
   - Target configurations with < {df['f5_environmental_impact_kWh_year'].quantile(0.25):,.0f} kWh/year
   - Emphasize edge computing and efficient sensors
   - Achievable carbon footprint: < {df['carbon_footprint_tons_CO2_year'].quantile(0.25):.1f} tons CO₂/year

3. **For High-Reliability Requirements:**
   - Select configurations with MTBF > {df['system_MTBF_hours'].quantile(0.75):,.0f} hours
   - Prioritize proven technologies with industrial-grade components
   - Expected availability: > {(1 - 100/df['system_MTBF_hours'].quantile(0.75)):.1%}

4. **Balanced Solutions:**
   - Consider the centroid region of the Pareto front
   - These solutions offer reasonable trade-offs across all objectives
   - Typical profile: ${df['f1_total_cost_USD'].median():,.0f} cost, {df['detection_recall'].median():.3f} recall, {df['f5_environmental_impact_kWh_year'].median():,.0f} kWh/year

6. CONCLUSIONS
═══════════════════════════════════════════════════════════════════════════════

This enhanced multi-objective optimization framework successfully identified
{len(df)} Pareto-optimal configurations for Road Maintenance Digital Twins,
revealing complex trade-offs between cost, performance, efficiency, 
sustainability, and reliability.

Key Contributions:
• Demonstrated the value of incorporating sustainability metrics in DT design
• Revealed non-intuitive relationships between system reliability and cost
• Provided quantitative decision support for stakeholders with diverse priorities
• Established a replicable framework for infrastructure DT optimization

Future Work:
• Extend to dynamic optimization considering temporal variations
• Incorporate uncertainty quantification in objective evaluations
• Develop preference-based solution selection mechanisms

═══════════════════════════════════════════════════════════════════════════════
END OF REPORT
"""
        
        # Save report
        with open(self.output_dir / 'optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
            
        # Also print to console
        print(report)
        logger.info("Generated comprehensive optimization report")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Initialize configuration
    config = OptimizationConfig()
    
    # Log startup
    logger.info("="*80)
    logger.info("Enhanced RMTwin Multi-Objective Optimization Framework v8.0")
    logger.info("For Automation in Construction")
    logger.info("="*80)
    logger.info(f"Configuration: {config.n_objectives} objectives, "
               f"{config.n_generations} generations, "
               f"{config.population_size} population size")
    
    # Save configuration
    config.save(Path(config.output_dir) / 'optimization_config.json')
    
    try:
        # Step 1: Load and populate ontology
        logger.info("\nStep 1: Loading enhanced ontology...")
        populator = EnhancedOntologyPopulator()
        ontology_graph = populator.populate_from_csvs(
            config.sensor_csv,
            config.algorithm_csv,
            config.infrastructure_csv,
            config.cost_benefit_csv
        )
        
        # Save populated ontology
        ontology_graph.serialize(
            destination=Path(config.output_dir) / 'populated_ontology.ttl',
            format='turtle'
        )
        
        # Step 2: Create optimization problem
        logger.info("\nStep 2: Setting up optimization problem...")
        problem = EnhancedRMTwinProblem(ontology_graph, config)
        
        # Step 3: Configure algorithm
        logger.info("\nStep 3: Configuring NSGA-II algorithm...")
        algorithm = NSGA2(
            pop_size=config.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=config.crossover_eta, prob=config.crossover_prob),
            mutation=PM(eta=config.mutation_eta, prob=1.0/problem.n_var),
            eliminate_duplicates=True
        )
        
        # Step 4: Run optimization
        logger.info("\nStep 4: Running optimization...")
        start_time = time.time()
        
        termination = get_termination("n_gen", config.n_generations)
        
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            save_history=True,
            verbose=True
        )
        
        optimization_time = time.time() - start_time
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        
        # Step 5: Save and analyze results
        logger.info("\nStep 5: Saving results...")
        results_manager = ResultsManager(config)
        df = results_manager.save_results(res, problem.evaluator)
        
        if df.empty:
            logger.error("No solutions found!")
            return
            
        # Step 6: Generate visualizations
        logger.info("\nStep 6: Creating visualizations...")
        visualizer = EnhancedVisualizer(config)
        visualizer.create_all_visualizations(df, res)
        
        # Step 7: Generate report
        logger.info("\nStep 7: Generating analysis report...")
        results_manager.generate_report(df)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Total solutions found: {len(df)}")
        logger.info(f"Feasible solutions: {df['is_feasible'].sum() if 'is_feasible' in df else len(df)}")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

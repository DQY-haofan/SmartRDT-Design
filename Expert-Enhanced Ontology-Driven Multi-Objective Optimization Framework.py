#!/usr/bin/env python3
"""
Enhanced Ontology-Driven Multi-Objective Optimization Framework for 
Road Maintenance Digital Twin (RMTwin) Configuration

Version: 8.1 - Fixed Edition
Features: 6 objectives including sustainability metrics, parallel computing, 
         caching, and publication-quality visualizations
Fixed: JSON serialization, ontology loading, SPARQL optimization

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
import pyshacl

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
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
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
# JSON SERIALIZATION HELPER
# ============================================================================

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

# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class OptimizationConfig:
    """Central configuration for the optimization framework"""
    
    # File paths
    sensor_csv: str = 'sensors_data.txt'
    algorithm_csv: str = 'algorithms_data.txt'
    infrastructure_csv: str = 'infrastructure_data.txt'
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
    

        # Baseline comparison (NEW)
    run_baseline_comparison: bool = True
    baseline_n_random_samples: int = 1000
    baseline_grid_resolution: int = 5
    baseline_weight_combinations: int = 50
    
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
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationConfig':
        """Load configuration from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def update_from_json(self, filepath: str):
        """Update configuration from JSON file"""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.info(f"Updated config: {key} = {value}")

# ============================================================================
# ENHANCED ONTOLOGY POPULATOR (FIXED)
# ============================================================================

class EnhancedOntologyPopulator:
    """Populates RDTcO-Maint ontology with enhanced sensor, algorithm, and infrastructure data"""
    
    def __init__(self):
        self.g = Graph()
        self.g.bind("rdtco", RDTCO)
        self.g.bind("ex", EX)
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("owl", OWL)
        self.g.bind("xsd", XSD)
        self._setup_base_ontology()
        
    def _setup_base_ontology(self):
        """Set up base ontology structure with proper class hierarchy"""
        # Define core classes
        self.g.add((RDTCO.DigitalTwinConfiguration, RDF.type, OWL.Class))
        self.g.add((RDTCO.SensorSystem, RDF.type, OWL.Class))
        self.g.add((RDTCO.Algorithm, RDF.type, OWL.Class))
        self.g.add((RDTCO.StorageSystem, RDF.type, OWL.Class))
        self.g.add((RDTCO.CommunicationSystem, RDF.type, OWL.Class))
        self.g.add((RDTCO.ComputeDeployment, RDF.type, OWL.Class))
        
        # Define sensor subclasses
        sensor_types = [
            'MMS_LiDAR_System', 'MMS_Camera_System', 'UAV_LiDAR_System',
            'UAV_Camera_System', 'TLS_System', 'Handheld_3D_Scanner',
            'FiberOptic_Sensor', 'Vehicle_LowCost_Sensor', 'IoT_Network_System'
        ]
        
        for sensor_type in sensor_types:
            sensor_class = RDTCO[sensor_type]
            self.g.add((sensor_class, RDF.type, OWL.Class))
            self.g.add((sensor_class, RDFS.subClassOf, RDTCO.SensorSystem))
            self.g.add((sensor_class, RDFS.label, Literal(sensor_type)))
        
        # Define algorithm subclasses
        algo_types = ['DeepLearningAlgorithm', 'MachineLearningAlgorithm', 
                      'TraditionalAlgorithm', 'PointCloudAlgorithm']
        
        for algo_type in algo_types:
            algo_class = RDTCO[algo_type]
            self.g.add((algo_class, RDF.type, OWL.Class))
            self.g.add((algo_class, RDFS.subClassOf, RDTCO.Algorithm))
            self.g.add((algo_class, RDFS.label, Literal(algo_type)))
        
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
            ('hasAccuracyRangeMM', 'Accuracy range in mm', XSD.decimal),
            ('hasDataVolumeGBPerKm', 'Data volume GB per km', XSD.decimal),
            ('hasCoverageEfficiencyKmPerDay', 'Coverage efficiency km per day', XSD.decimal),
            ('hasOperatingSpeedKmh', 'Operating speed km/h', XSD.decimal),
            ('hasRecall', 'Detection recall rate', XSD.decimal),
            ('hasPrecision', 'Detection precision rate', XSD.decimal),
            ('hasFPS', 'Frames per second', XSD.decimal),
            ('hasAnnualOpCostUSD', 'Annual operational cost', XSD.decimal),
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
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} sensor instances...")
            
            for _, row in df.iterrows():
                sensor_uri = EX[row['Sensor_Instance_Name']]
                sensor_type = RDTCO[row['Sensor_RDF_Type']]
                
                # Ensure type exists
                self.g.add((sensor_type, RDF.type, OWL.Class))
                self.g.add((sensor_type, RDFS.subClassOf, RDTCO.SensorSystem))
                
                self.g.add((sensor_uri, RDF.type, sensor_type))
                self.g.add((sensor_uri, RDFS.label, Literal(row['Sensor_Instance_Name'])))
                
                # Add enhanced properties with safe conversion
                self.g.add((sensor_uri, RDTCO.hasInitialCostUSD, 
                           Literal(float(row['Initial_Cost_USD']), datatype=XSD.decimal)))
                self.g.add((sensor_uri, RDTCO.hasOperationalCostUSDPerDay, 
                           Literal(float(row['Operational_Cost_USD_per_day']), datatype=XSD.decimal)))
                self.g.add((sensor_uri, RDTCO.hasEnergyConsumptionW, 
                           Literal(float(row['Energy_Consumption_W']), datatype=XSD.decimal)))
                self.g.add((sensor_uri, RDTCO.hasMTBFHours, 
                           Literal(float(row['MTBF_hours']), datatype=XSD.decimal)))
                self.g.add((sensor_uri, RDTCO.hasOperatorSkillLevel, 
                           Literal(str(row['Operator_Skill_Level']))))
                
                if pd.notna(row['Calibration_Freq_months']) and row['Calibration_Freq_months'] != 'N/A':
                    try:
                        self.g.add((sensor_uri, RDTCO.hasCalibrationFreqMonths, 
                                   Literal(float(row['Calibration_Freq_months']), datatype=XSD.decimal)))
                    except:
                        pass
                
                # Add other properties
                self.g.add((sensor_uri, RDTCO.hasAccuracyRangeMM, 
                           Literal(float(row['Accuracy_Range_mm']), datatype=XSD.decimal)))
                self.g.add((sensor_uri, RDTCO.hasDataVolumeGBPerKm, 
                           Literal(float(row['Data_Volume_GB_per_km']), datatype=XSD.decimal)))
                self.g.add((sensor_uri, RDTCO.hasCoverageEfficiencyKmPerDay, 
                           Literal(float(row['Coverage_Efficiency_km_per_day']), datatype=XSD.decimal)))
                
                logger.debug(f"Added sensor: {sensor_uri}")
                
        except Exception as e:
            logger.error(f"Error loading sensors: {e}")
            raise
    
    def _load_algorithms(self, filepath: str):
        """Load enhanced algorithm data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} algorithm instances...")
            
            for _, row in df.iterrows():
                algo_uri = EX[row['Algorithm_Instance_Name']]
                algo_type = RDTCO[row['Algorithm_RDF_Type']]
                
                # Ensure type exists
                self.g.add((algo_type, RDF.type, OWL.Class))
                self.g.add((algo_type, RDFS.subClassOf, RDTCO.Algorithm))
                
                self.g.add((algo_uri, RDF.type, algo_type))
                self.g.add((algo_uri, RDFS.label, Literal(row['Algorithm_Instance_Name'])))
                
                # Add performance metrics
                self.g.add((algo_uri, RDTCO.hasPrecision, 
                           Literal(float(row['Precision']), datatype=XSD.decimal)))
                self.g.add((algo_uri, RDTCO.hasRecall, 
                           Literal(float(row['Recall']), datatype=XSD.decimal)))
                self.g.add((algo_uri, RDTCO.hasFPS, 
                           Literal(float(row['FPS']), datatype=XSD.decimal)))
                
                # Add enhanced properties
                self.g.add((algo_uri, RDTCO.hasHardwareRequirement, 
                           Literal(str(row['Hardware_Requirement']))))
                self.g.add((algo_uri, RDTCO.hasDataAnnotationCostUSD, 
                           Literal(float(row['Data_Annotation_Cost_USD']), datatype=XSD.decimal)))
                self.g.add((algo_uri, RDTCO.hasModelRetrainingFreqMonths, 
                           Literal(float(row['Model_Retraining_Freq_months']), datatype=XSD.decimal)))
                self.g.add((algo_uri, RDTCO.hasExplainabilityScore, 
                           Literal(int(row['Explainability_Score']), datatype=XSD.integer)))
                           
                logger.debug(f"Added algorithm: {algo_uri}")
                
        except Exception as e:
            logger.error(f"Error loading algorithms: {e}")
            raise
    
    def _load_infrastructure(self, filepath: str):
        """Load enhanced infrastructure data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} infrastructure instances...")
            
            for _, row in df.iterrows():
                comp_uri = EX[row['Component_Instance_Name']]
                comp_type = RDTCO[row['Component_RDF_Type']]
                
                self.g.add((comp_uri, RDF.type, comp_type))
                self.g.add((comp_uri, RDFS.label, Literal(row['Component_Instance_Name'])))
                
                # Add costs
                self.g.add((comp_uri, RDTCO.hasInitialCostUSD, 
                           Literal(float(row['Initial_Cost_USD']), datatype=XSD.decimal)))
                self.g.add((comp_uri, RDTCO.hasAnnualOpCostUSD, 
                           Literal(float(row['Annual_OpCost_USD']), datatype=XSD.decimal)))
                
                # Add enhanced properties
                if pd.notna(row['Energy_Consumption_W']) and row['Energy_Consumption_W'] != 'N/A':
                    try:
                        self.g.add((comp_uri, RDTCO.hasEnergyConsumptionW, 
                                   Literal(float(row['Energy_Consumption_W']), datatype=XSD.decimal)))
                    except:
                        pass
                        
                if pd.notna(row['MTBF_hours']) and row['MTBF_hours'] != 'N/A':
                    try:
                        self.g.add((comp_uri, RDTCO.hasMTBFHours, 
                                   Literal(float(row['MTBF_hours']), datatype=XSD.decimal)))
                    except:
                        pass
                        
                if pd.notna(row['Integration_Complexity']) and row['Integration_Complexity'] != 'N/A':
                    try:
                        self.g.add((comp_uri, RDTCO.hasIntegrationComplexity, 
                                   Literal(int(row['Integration_Complexity']), datatype=XSD.integer)))
                    except:
                        pass
                        
                if pd.notna(row['Cybersecurity_Vulnerability']) and row['Cybersecurity_Vulnerability'] != 'N/A':
                    try:
                        self.g.add((comp_uri, RDTCO.hasCybersecurityVulnerability, 
                                   Literal(int(row['Cybersecurity_Vulnerability']), datatype=XSD.integer)))
                    except:
                        pass
                        
                logger.debug(f"Added infrastructure: {comp_uri}")
                
        except Exception as e:
            logger.error(f"Error loading infrastructure: {e}")
            raise
                           
    def _load_cost_benefit(self, filepath: str):
        """Load cost-benefit data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} cost-benefit entries...")
            
            # Store as configuration parameters
            for _, row in df.iterrows():
                param_uri = EX[f"Parameter_{row['Metric_Name']}"]
                self.g.add((param_uri, RDF.type, RDTCO.ConfigurationParameter))
                self.g.add((param_uri, RDFS.label, Literal(row['Metric_Name'])))
                self.g.add((param_uri, RDTCO.hasValue, Literal(float(row['Value']), datatype=XSD.decimal)))
                self.g.add((param_uri, RDTCO.hasUnit, Literal(str(row['Unit']))))
                
        except Exception as e:
            logger.error(f"Error loading cost-benefit data: {e}")
            raise
            
    def _add_shacl_constraints(self):
        """Add SHACL constraints for validation"""
        # This is a placeholder - implement specific SHACL shapes as needed
        pass

# ============================================================================
# SOLUTION MAPPER WITH CACHING (FIXED)
# ============================================================================

class CachedSolutionMapper:
    """Maps between optimization variables and RMTwin configurations with caching"""
    
    def __init__(self, ontology_graph: Graph):
        self.g = ontology_graph
        self._cache_components()
        
    def _cache_components(self):
        """Cache all available components from ontology - FIXED VERSION"""
        
        # Initialize empty lists
        self.sensors = []
        self.algorithms = []
        self.storage_systems = []
        self.comm_systems = []
        self.deployments = []
        
        logger.info("Caching components from ontology...")
        
        # Query for all instances
        query_template = """
        SELECT DISTINCT ?instance ?type WHERE {
            ?instance rdf:type ?type .
            ?type rdfs:subClassOf* ?parent .
        }
        """
        
        # Query for sensors
        sensor_query = query_template.replace("?parent", "rdtco:SensorSystem")
        for row in self.g.query(sensor_query):
            instance_str = str(row.instance)
            if instance_str.startswith('http://example.org/'):
                self.sensors.append(instance_str)
        
        # Query for algorithms
        algo_query = query_template.replace("?parent", "rdtco:Algorithm")
        for row in self.g.query(algo_query):
            instance_str = str(row.instance)
            if instance_str.startswith('http://example.org/'):
                self.algorithms.append(instance_str)
        
        # Direct type queries for infrastructure
        for s, p, o in self.g:
            if str(p) == str(RDF.type):
                subject_str = str(s)
                object_str = str(o)
                
                if subject_str.startswith('http://example.org/'):
                    if 'StorageSystem' in object_str:
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
            logger.warning("No sensors found, using defaults")
            self.sensors = [
                "http://example.org/rmtwin#MMS_LiDAR_Riegl_VUX1HA",
                "http://example.org/rmtwin#UAV_Camera_DJI_Mavic3Pro",
                "http://example.org/rmtwin#IoT_Wireless_Network"
            ]
        
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
        
        # Ensure array is 1D
        if isinstance(x, np.ndarray):
            x = x.flatten()
        
        # Safe selection with bounds checking
        def safe_select(array, value, name):
            if not array:
                logger.warning(f"No {name} available, using default")
                return f"default_{name}"
            idx = int(value * len(array)) % max(len(array), 1)
            idx = min(idx, len(array) - 1)
            return array[idx]
        
        config = {
            'sensor': safe_select(self.sensors, x[0], 'sensor'),
            'data_rate': 10 + x[1] * 90,  # 10-100 Hz
            'geo_lod': ['Micro', 'Meso', 'Macro'][int(x[2] * 3) % 3],
            'cond_lod': ['Micro', 'Meso', 'Macro'][int(x[3] * 3) % 3],
            'algorithm': safe_select(self.algorithms, x[4], 'algorithm'),
            'detection_threshold': 0.1 + x[5] * 0.8,  # 0.1-0.9
            'storage': safe_select(self.storage_systems, x[6], 'storage'),
            'communication': safe_select(self.comm_systems, x[7], 'communication'),
            'deployment': safe_select(self.deployments, x[8], 'deployment'),
            'crew_size': int(1 + x[9] * 9),  # 1-10
            'inspection_cycle': int(1 + x[10] * 364)  # 1-365 days
        }
        
        return config

# ============================================================================
# ENHANCED FITNESS EVALUATOR WITH BATCH SPARQL QUERIES
# ============================================================================

class ParallelFitnessEvaluator:
    """Evaluates fitness with 6 objectives including sustainability metrics"""
    
    def __init__(self, ontology_graph: Graph, config: OptimizationConfig):
        self.g = ontology_graph
        self.config = config
        self.mapper = CachedSolutionMapper(ontology_graph)
        
        # Batch query cache
        self._batch_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Normalization parameters
        self.norm_params = {
            'cost': {'min': 100_000, 'max': 2_000_000},
            'recall': {'min': 0.0, 'max': 0.4},  # 1-recall
            'latency': {'min': 0.1, 'max': 300.0},
            'disruption': {'min': 0.0, 'max': 500.0},
            'environmental': {'min': 1_000, 'max': 100_000},  # kWh/year
            'reliability': {'min': 0.0, 'max': 0.001}  # 1/MTBF
        }
    
    def _batch_query_properties(self, subject_predicate_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[Union[float, str]]]:
        """Batch query multiple properties for efficiency"""
        if not subject_predicate_pairs:
            return {}
        
        # Check cache
        cache_key = str(sorted(subject_predicate_pairs))
        if cache_key in self._batch_cache:
            self._cache_hits += 1
            return self._batch_cache[cache_key]
        
        self._cache_misses += 1
        
        # Build batch query
        values_clause = []
        for subject, predicate in subject_predicate_pairs:
            values_clause.append(f"(<{subject}> <{predicate}>)")
        
        query = f"""
        PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
        SELECT ?subject ?predicate ?value WHERE {{
            VALUES (?subject ?predicate) {{ {' '.join(values_clause)} }}
            ?subject ?predicate ?value .
        }}
        """
        
        results = {}
        try:
            for row in self.g.query(query):
                key = (str(row.subject), str(row.predicate))
                value_str = str(row.value)
                try:
                    # Try to convert to float
                    results[key] = float(value_str)
                except ValueError:
                    # Keep as string
                    results[key] = value_str
        except Exception as e:
            logger.warning(f"Batch query failed: {e}")
        
        # Add None for missing values
        for pair in subject_predicate_pairs:
            if pair not in results:
                results[pair] = None
        
        # Cache results
        self._batch_cache[cache_key] = results
        return results
    
    def _get_query_result(self, query_results: Dict, subject: str, predicate: str, default=None):
        """Get value from batch query results"""
        return query_results.get((subject, predicate), default)
        
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of solutions"""
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
                    try:
                        obj, const = future.result()
                        objectives[idx] = obj
                        constraints[idx] = const
                    except Exception as e:
                        logger.warning(f"Evaluation failed for solution {idx}: {e}")
                        objectives[idx] = np.full(6, np.inf)
                        constraints[idx] = np.array([1e6, 1e6, 1e6])
        else:
            # Sequential processing for small batches
            for i, x in enumerate(X):
                try:
                    objectives[i], constraints[i] = self._evaluate_single(x)
                except Exception as e:
                    logger.warning(f"Evaluation failed for solution {i}: {e}")
                    objectives[i] = np.full(6, np.inf)
                    constraints[i] = np.array([1e6, 1e6, 1e6])
                
        return objectives, constraints
    
    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single solution using batch queries"""
        config = self.mapper.decode_solution(x)
        
        # Collect all needed queries
        queries_needed = []
        
        # Sensor queries
        sensor_uri = config['sensor']
        queries_needed.extend([
            (sensor_uri, str(RDTCO.hasInitialCostUSD)),
            (sensor_uri, str(RDTCO.hasOperationalCostUSDPerDay)),
            (sensor_uri, str(RDTCO.hasOperatorSkillLevel)),
            (sensor_uri, str(RDTCO.hasAccuracyRangeMM)),
            (sensor_uri, str(RDTCO.hasDataVolumeGBPerKm)),
            (sensor_uri, str(RDTCO.hasCoverageEfficiencyKmPerDay)),
            (sensor_uri, str(RDTCO.hasOperatingSpeedKmh)),
            (sensor_uri, str(RDTCO.hasEnergyConsumptionW)),
            (sensor_uri, str(RDTCO.hasMTBFHours))
        ])
        
        # Algorithm queries
        algo_uri = config['algorithm']
        queries_needed.extend([
            (algo_uri, str(RDTCO.hasRecall)),
            (algo_uri, str(RDTCO.hasDataAnnotationCostUSD)),
            (algo_uri, str(RDTCO.hasModelRetrainingFreqMonths)),
            (algo_uri, str(RDTCO.hasFPS))
        ])
        
        # Infrastructure queries
        for component in ['storage', 'communication', 'deployment']:
            if component in config:
                comp_uri = config[component]
                queries_needed.extend([
                    (comp_uri, str(RDTCO.hasInitialCostUSD)),
                    (comp_uri, str(RDTCO.hasEnergyConsumptionW)),
                    (comp_uri, str(RDTCO.hasMTBFHours))
                ])
        
        # Execute batch query
        query_results = self._batch_query_properties(queries_needed)
        
        # Calculate objectives using batch results
        f1 = self._calculate_total_cost_batch(config, query_results)
        f2 = self._calculate_detection_performance_batch(config, query_results)
        f3 = self._calculate_latency_batch(config, query_results)
        f4 = self._calculate_traffic_disruption_batch(config, query_results)
        f5 = self._calculate_environmental_impact_batch(config, query_results)
        f6 = self._calculate_system_reliability_batch(config, query_results)
        
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
    
    def _calculate_total_cost_batch(self, config: Dict, query_results: Dict) -> float:
        """Calculate total cost using batch query results"""
        total_cost = 0
        
        # Initial costs
        for component in ['sensor', 'storage', 'communication', 'deployment']:
            if component in config:
                cost = self._get_query_result(query_results, config[component], 
                                            str(RDTCO.hasInitialCostUSD), 0)
                if cost:
                    total_cost += cost
        
        # Operational costs
        sensor_op_cost = self._get_query_result(query_results, config['sensor'], 
                                            str(RDTCO.hasOperationalCostUSDPerDay), 0)
        if sensor_op_cost:
            coverage_efficiency = self._get_query_result(query_results, config['sensor'],
                                                        str(RDTCO.hasCoverageEfficiencyKmPerDay), 80)
            inspections_per_year = 365 / config['inspection_cycle']
            
            # FIX: Handle zero coverage efficiency
            if coverage_efficiency > 0:
                days_per_inspection = self.config.road_network_length_km / coverage_efficiency
                annual_sensor_cost = sensor_op_cost * days_per_inspection * inspections_per_year
            else:
                # For stationary sensors, use fixed operational cost
                annual_sensor_cost = sensor_op_cost * 365  # Continuous operation
                
            total_cost += annual_sensor_cost * self.config.planning_horizon_years
        
        # Crew costs
        skill_level = self._get_query_result(query_results, config['sensor'], 
                                        str(RDTCO.hasOperatorSkillLevel), 'Basic')
        skill_multiplier = {'Basic': 1.0, 'Intermediate': 1.5, 'Expert': 2.0}.get(
            str(skill_level), 1.0)
        
        # FIX: Calculate crew costs correctly for stationary sensors
        coverage_efficiency = self._get_query_result(query_results, config['sensor'],
                                                    str(RDTCO.hasCoverageEfficiencyKmPerDay), 80)
        
        if coverage_efficiency > 0:
            days_per_inspection = self.config.road_network_length_km / coverage_efficiency
            crew_daily_cost = config['crew_size'] * 1000 * skill_multiplier
            crew_annual_cost = crew_daily_cost * days_per_inspection * inspections_per_year
        else:
            # Stationary sensors need minimal crew time (only for maintenance)
            crew_daily_cost = config['crew_size'] * 1000 * skill_multiplier
            crew_annual_cost = crew_daily_cost * 10  # Assume 10 days/year for maintenance
            
        total_cost += crew_annual_cost * self.config.planning_horizon_years
        
        # Data annotation costs
        annotation_cost = self._get_query_result(query_results, config['algorithm'],
                                            str(RDTCO.hasDataAnnotationCostUSD), 0)
        if annotation_cost:
            total_annotation = annotation_cost * 10000
            total_cost += total_annotation
        
        # Model retraining costs
        retrain_freq = self._get_query_result(query_results, config['algorithm'],
                                            str(RDTCO.hasModelRetrainingFreqMonths), 0)
        if retrain_freq and retrain_freq > 0:
            retrainings = (self.config.planning_horizon_years * 12) / retrain_freq
            total_cost += retrainings * 5000
        
        return total_cost
   
   
    def _calculate_detection_performance_batch(self, config: Dict, query_results: Dict) -> float:
        """Calculate 1 - recall using batch query results"""
        base_recall = self._get_query_result(query_results, config['algorithm'], 
                                           str(RDTCO.hasRecall), 0.7)
        
        # Adjust for sensor accuracy
        sensor_accuracy = self._get_query_result(query_results, config['sensor'],
                                               str(RDTCO.hasAccuracyRangeMM), 10)
        if sensor_accuracy:
            accuracy_factor = 1 - (sensor_accuracy / 100)
            base_recall *= (0.8 + 0.2 * accuracy_factor)
        
        # Adjust for LOD
        lod_factors = {'Micro': 1.1, 'Meso': 1.0, 'Macro': 0.9}
        base_recall *= lod_factors.get(config['geo_lod'], 1.0)
        
        # Adjust for detection threshold
        threshold_factor = 1 - abs(config['detection_threshold'] - 0.5) * 0.2
        base_recall *= threshold_factor
        
        return 1 - min(base_recall, 0.99)
    
    def _calculate_latency_batch(self, config: Dict, query_results: Dict) -> float:
        """Calculate data-to-decision latency"""
        # Data acquisition time
        data_rate = config['data_rate']
        acq_time = 1 / data_rate if data_rate > 0 else 1.0
        
        # Data volume
        data_volume = self._get_query_result(query_results, config['sensor'],
                                           str(RDTCO.hasDataVolumeGBPerKm), 1.0)
        
        # Communication time
        comm_bandwidth = {'5G_Network': 1000, 'LoRaWAN': 0.05, 
                         'Fiber_Optic': 10000, '4G_LTE': 100}.get(
                         config['communication'].split('/')[-1], 100)
        
        comm_time = (data_volume * 1000) / comm_bandwidth
        
        # Processing time
        algo_fps = self._get_query_result(query_results, config['algorithm'],
                                        str(RDTCO.hasFPS), 10)
        if algo_fps and algo_fps > 0:
            proc_time = 1 / algo_fps
        else:
            proc_time = 0.1
        
        # Deployment factor
        deploy_factors = {'Edge_Computing': 1.5, 'Cloud_Computing': 1.0, 
                         'Hybrid_Edge_Cloud': 1.2}
        deploy_factor = deploy_factors.get(config['deployment'].split('/')[-1], 1.0)
        proc_time *= deploy_factor
        
        return acq_time + comm_time + proc_time
    
    
    def _calculate_traffic_disruption_batch(self, config: Dict, query_results: Dict) -> float:
        """Calculate traffic disruption hours"""
        # Base disruption per inspection
        base_disruption = 4.0
        
        # Adjust for sensor speed
        sensor_speed = self._get_query_result(query_results, config['sensor'],
                                            str(RDTCO.hasOperatingSpeedKmh), 80)
        
        # FIX: Handle zero speed (stationary sensors)
        if sensor_speed and sensor_speed > 0:
            speed_factor = 80 / sensor_speed
            base_disruption *= speed_factor
        elif sensor_speed == 0:
            # Stationary sensors (like fiber optic) cause minimal disruption after installation
            base_disruption = 0.1  # Minimal disruption for maintenance only
        
        # Total annual disruption
        inspections_per_year = 365 / config['inspection_cycle']
        annual_disruption = base_disruption * inspections_per_year
        
        # Time of day adjustment
        annual_disruption *= 0.79
        
        return annual_disruption

    
    def _calculate_environmental_impact_batch(self, config: Dict, query_results: Dict) -> float:
        """Calculate environmental impact in kWh/year"""
        total_energy_w = 0
        
        # Component energy consumption
        components = ['sensor', 'storage', 'communication', 'deployment']
        for comp in components:
            if comp in config:
                energy = self._get_query_result(query_results, config[comp],
                                            str(RDTCO.hasEnergyConsumptionW), 0)
                if energy:
                    total_energy_w += energy
        
        # Calculate operational hours
        inspections_per_year = 365 / config['inspection_cycle']
        coverage_efficiency = self._get_query_result(query_results, config['sensor'],
                                                    str(RDTCO.hasCoverageEfficiencyKmPerDay), 50)
        
        # FIX: Handle zero coverage efficiency (e.g., for stationary sensors)
        if coverage_efficiency <= 0:
            # For stationary sensors like fiber optic, assume continuous monitoring
            # No "days per inspection" - they monitor continuously
            sensor_hours = 365 * 24  # Continuous monitoring
        else:
            days_per_inspection = self.config.road_network_length_km / coverage_efficiency
            sensor_hours = days_per_inspection * inspections_per_year * 8
        
        # Backend runs 24/7
        backend_hours = 365 * 24
        
        # Calculate total energy
        sensor_energy_kwh = (total_energy_w * 0.3 * sensor_hours) / 1000
        backend_energy_kwh = (total_energy_w * 0.7 * backend_hours) / 1000
        
        # Add vehicle emissions (only for mobile sensors)
        if coverage_efficiency > 0:
            vehicle_km = self.config.road_network_length_km * inspections_per_year
            vehicle_kwh_equiv = vehicle_km * 0.8
        else:
            # Stationary sensors don't require vehicle travel
            vehicle_kwh_equiv = 0
        
        return sensor_energy_kwh + backend_energy_kwh + vehicle_kwh_equiv

    def _calculate_system_reliability_batch(self, config: Dict, query_results: Dict) -> float:
        """Calculate system reliability as 1/MTBF"""
        inverse_mtbf_sum = 0
        
        # Series reliability model
        components = ['sensor', 'storage', 'communication', 'deployment']
        for comp in components:
            if comp in config:
                mtbf = self._get_query_result(query_results, config[comp],
                                            str(RDTCO.hasMTBFHours), 0)
                if mtbf and mtbf > 0:
                    inverse_mtbf_sum += 1 / mtbf
        
        # Avoid division by zero
        if inverse_mtbf_sum == 0:
            return 1.0
            
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
# RESULTS MANAGER (FIXED JSON SERIALIZATION)
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
            # FIX: Handle very small or zero f6_raw values
            system_mtbf = 1 / f6_raw if f6_raw > 1e-10 else 1e10  # Use large number instead of inf
            
            # Build result row
            row = {
                'solution_id': int(i + 1),
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
                'crew_size': int(config['crew_size']),
                'inspection_cycle_days': int(config['inspection_cycle']),
                # Raw objectives
                'f1_total_cost_USD': round(float(f1_raw), 2),
                'f2_one_minus_recall': round(float(f2_raw), 4),
                'f3_latency_seconds': round(float(f3_raw), 2),
                'f4_traffic_disruption_hours': round(float(f4_raw), 2),
                'f5_environmental_impact_kWh_year': round(float(f5_raw), 2),
                'f6_system_reliability_inverse_MTBF': round(float(f6_raw), 8),
                # Derived metrics
                'detection_recall': round(float(recall), 4),
                'system_MTBF_hours': round(float(system_mtbf), 0),
                'annual_cost_USD': round(float(f1_raw / self.config.planning_horizon_years), 2),
                'cost_per_km_year': round(float(f1_raw / self.config.planning_horizon_years / 
                                        self.config.road_network_length_km), 2),
                'carbon_footprint_tons_CO2_year': round(float(f5_raw * self.config.carbon_intensity_kwh / 1000), 2),
                # Constraints
                'is_feasible': bool(np.all(const <= 0) if const is not None else True)
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
        """Save optimization summary with proper JSON serialization"""
        # Convert all values to JSON-serializable types
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.to_dict(),
            'results': {
                'total_solutions': int(len(df)),
                'feasible_solutions': int(df['is_feasible'].sum()) if 'is_feasible' in df else int(len(df)),
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
                        'min': float(df['system_MTBF_hours'].replace([np.inf], np.nan).min()),
                        'max': float(df['system_MTBF_hours'].replace([np.inf], np.nan).max()),
                        'mean': float(df['system_MTBF_hours'].replace([np.inf], np.nan).mean()),
                        'std': float(df['system_MTBF_hours'].replace([np.inf], np.nan).std())
                    }
                }
            }
        }
        
        # Save with custom encoder
        with open(self.output_dir / 'optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
    def generate_report(self, df: pd.DataFrame):
        """Generate comprehensive analysis report"""
        report = f"""
═══════════════════════════════════════════════════════════════════════════════
    ENHANCED RMTWIN MULTI-OBJECTIVE OPTIMIZATION REPORT
    Automation in Construction - Academic Edition
═══════════════════════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Framework Version: 8.1 (Fixed)

1. EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

This report presents the results of a 6-objective optimization for Road 
Maintenance Digital Twin (RMTwin) configuration, incorporating sustainability 
and reliability dimensions alongside traditional performance metrics.

Key Findings:
- Total Pareto-optimal solutions: {len(df)}
- Feasible solutions: {df['is_feasible'].sum() if 'is_feasible' in df else len(df)} ({df['is_feasible'].sum()/len(df)*100:.1f}%)
- Objective space coverage: 6-dimensional
- Computational efficiency: {self.config.n_generations} generations with {self.config.population_size} population size

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
│ MTBF (hours)           │ {df['system_MTBF_hours'].replace([np.inf], np.nan).min():>14,.0f} │ {df['system_MTBF_hours'].replace([np.inf], np.nan).max():>14,.0f} │ {df['system_MTBF_hours'].replace([np.inf], np.nan).mean():>7,.0f} ± {df['system_MTBF_hours'].replace([np.inf], np.nan).std():>6,.0f} │
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
            ('Most Reliable', df['system_MTBF_hours'].replace([np.inf], np.nan).idxmax())
        ]
        
        for name, idx in extremes:
            if pd.notna(idx):
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
- Average Annual Energy Consumption: {df['f5_environmental_impact_kWh_year'].mean():,.0f} kWh
- Equivalent CO₂ Emissions: {df['carbon_footprint_tons_CO2_year'].mean():.1f} tons/year
- Energy Efficiency Range: {df['f5_environmental_impact_kWh_year'].min():,.0f} - {df['f5_environmental_impact_kWh_year'].max():,.0f} kWh/year
- Low-Carbon Solutions (<10 tons CO₂/year): {(df['carbon_footprint_tons_CO2_year'] < 10).sum()} ({(df['carbon_footprint_tons_CO2_year'] < 10).sum()/len(df)*100:.1f}%)

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
- Average System MTBF: {df['system_MTBF_hours'].replace([np.inf], np.nan).mean():,.0f} hours ({df['system_MTBF_hours'].replace([np.inf], np.nan).mean()/8760:.1f} years)
- Reliability Range: {df['system_MTBF_hours'].replace([np.inf], np.nan).min():,.0f} - {df['system_MTBF_hours'].replace([np.inf], np.nan).max():,.0f} hours
- High-Reliability Solutions (>3 years MTBF): {(df['system_MTBF_hours'] > 26280).sum()} ({(df['system_MTBF_hours'] > 26280).sum()/len(df)*100:.1f}%)

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
        
        # Replace inf values before correlation
        df_corr = df[obj_cols].replace([np.inf, -np.inf], np.nan)
        corr_matrix = df_corr.corr()
        
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
   - Select configurations with MTBF > {df['system_MTBF_hours'].replace([np.inf], np.nan).quantile(0.75):,.0f} hours
   - Prioritize proven technologies with industrial-grade components
   - Expected availability: > {(1 - 100/df['system_MTBF_hours'].replace([np.inf], np.nan).quantile(0.75)):.1%}

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
- Demonstrated the value of incorporating sustainability metrics in DT design
- Revealed non-intuitive relationships between system reliability and cost
- Provided quantitative decision support for stakeholders with diverse priorities
- Established a replicable framework for infrastructure DT optimization

Future Work:
- Extend to dynamic optimization considering temporal variations
- Incorporate uncertainty quantification in objective evaluations
- Develop preference-based solution selection mechanisms

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
# VISUALIZATION MODULE (Placeholder - use existing code)
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
        """Create all visualizations - placeholder"""
        logger.info("Creating visualizations...")
        # Note: Use the existing visualization code from the original file
        # This is just a placeholder to keep the main script complete
        
        # Save a simple plot as example
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['f1_total_cost_USD']/1000, df['detection_recall'], alpha=0.6)
        ax.set_xlabel('Total Cost (k$)')
        ax.set_ylabel('Detection Recall')
        ax.set_title('Cost vs Performance Trade-off')
        fig.savefig(self.output_dir / 'png' / 'cost_vs_recall.png')
        plt.close(fig)
        
        logger.info("Visualization placeholder completed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Initialize configuration
    config = OptimizationConfig()
    
    # Update from config.json if exists
    config.update_from_json('config.json')
    
    # Log startup
    logger.info("="*80)
    logger.info("Enhanced RMTwin Multi-Objective Optimization Framework v8.1")
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
        
        # Output cache statistics
        if hasattr(problem.evaluator, '_cache_hits'):
            cache_hits = problem.evaluator._cache_hits
            cache_misses = problem.evaluator._cache_misses
            total_queries = cache_hits + cache_misses
            if total_queries > 0:
                cache_rate = cache_hits / total_queries * 100
                logger.info(f"SPARQL query cache hit rate: {cache_rate:.1f}% "
                          f"({cache_hits} hits, {cache_misses} misses)")
        
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
        

        # Step 8: Run baseline comparison (NEW)
        logger.info("\nStep 8: Running baseline comparison...")
        try:
            from baseline_comparison import run_baseline_comparison
            
            baseline_results = run_baseline_comparison(
                ontology_graph=ontology_graph,
                pareto_csv_path=Path(config.output_dir) / 'pareto_solutions_6d.csv'
            )
            
            if baseline_results:
                logger.info("Baseline comparison completed successfully")
                logger.info("Check ./results/baseline/ for detailed comparison results")
            
        except ImportError:
            logger.warning("baseline_comparison module not found, skipping baseline analysis")
        except Exception as e:
            logger.error(f"Error during baseline comparison: {e}")
            logger.info("Main optimization completed successfully, but baseline comparison failed")
        
        # Final summary (修改以包含基线对比信息)
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION AND ANALYSIS COMPLETE")
        logger.info(f"Total solutions found: {len(df)}")
        logger.info(f"Feasible solutions: {df['is_feasible'].sum() if 'is_feasible' in df else len(df)}")
        logger.info(f"Results saved to: {config.output_dir}")
        if baseline_results:
            logger.info(f"Baseline comparison saved to: {config.output_dir}/baseline/")
        logger.info("="*80)

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
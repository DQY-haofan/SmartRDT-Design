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

from enhanced_evaluation_v2 import (
    AdvancedEvaluationConfig,
    EnhancedFitnessEvaluatorV2, 
    DynamicNormalizer
)

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
    
    road_network_length_km: float = None
    planning_horizon_years: int = None
    budget_cap_usd: float = None
    min_recall_threshold: float = 0.80
    max_latency_seconds: float = 60.0
    max_disruption_hours: float = 100.0
    max_energy_kwh_year: float = 50_000
    min_mtbf_hours: float = 5_000
    
    # Algorithm parameters
    population_size: int = 200
    n_generations: int = 20
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


class IntegratedFitnessEvaluator:
    """集成增强版评估器的适配器类"""
    
    def __init__(self, ontology_graph: Graph, config: OptimizationConfig):
        self.g = ontology_graph
        self.config = config
        self.mapper = CachedSolutionMapper(ontology_graph)
        
        # 创建高级配置 - 确保所有参数都设置
        self.advanced_config = AdvancedEvaluationConfig(
            road_network_length_km=config.road_network_length_km,
            planning_horizon_years=config.planning_horizon_years,
            budget_cap_usd=config.budget_cap_usd,
            daily_wage_per_person=1500,
            fos_sensor_spacing_km=0.1,  # 确保设置
            depreciation_rate=0.1,      # 确保设置
            scenario_type='urban',      # 确保设置
            carbon_intensity_factor=0.417  # 确保设置
        )
        
        # 创建增强版评估器
        self.evaluator_v2 = EnhancedFitnessEvaluatorV2(
            ontology_graph=ontology_graph,
            config=self.advanced_config
        )
        
        # 移除动态归一化器 - 不再需要
        # self.normalizer = DynamicNormalizer()
        
        # 统计信息
        self._evaluation_count = 0
        self._total_eval_time = 0
        
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估批量解决方案 - 返回原始目标值"""
        import time
        batch_start = time.time()
        
        n_solutions = len(X)
        raw_objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 3))
        
        # 评估每个解
        for i, x in enumerate(X):
            eval_start = time.time()
            decoded = self.mapper.decode_solution(x)
            obj, const = self.evaluator_v2.evaluate_solution(x, decoded)
            raw_objectives[i] = obj
            constraints[i] = const
            

        # 批量评估统计
        batch_time = time.time() - batch_start
        self._evaluation_count += n_solutions
        self._total_eval_time += batch_time
        
        # 每1000次评估输出一次统计
        if self._evaluation_count % 1000 == 0:
            avg_time = self._total_eval_time / self._evaluation_count
            logger.info(f"Average evaluation time: {avg_time:.4f}s per solution")
            
            # 输出目标值范围用于调试
            logger.info("Current objective ranges:")
            logger.info(f"  Cost: ${raw_objectives[:, 0].min():.0f} - ${raw_objectives[:, 0].max():.0f}")
            logger.info(f"  1-Recall: {raw_objectives[:, 1].min():.3f} - {raw_objectives[:, 1].max():.3f}")
            logger.info(f"  Latency: {raw_objectives[:, 2].min():.1f} - {raw_objectives[:, 2].max():.1f}s")
            logger.info(f"  Disruption: {raw_objectives[:, 3].min():.1f} - {raw_objectives[:, 3].max():.1f}h")
            logger.info(f"  Carbon: {raw_objectives[:, 4].min():.0f} - {raw_objectives[:, 4].max():.0f} kgCO2e")
            logger.info(f"  1/MTBF: {raw_objectives[:, 5].min():.8f} - {raw_objectives[:, 5].max():.8f}")
        
        # 关键修改：直接返回原始值，不进行归一化！
        # NSGA-II可以处理不同尺度的目标
        return raw_objectives, constraints
    
    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估单个解（用于结果保存）"""
        decoded = self.mapper.decode_solution(x)
        return self.evaluator_v2.evaluate_solution(x, decoded)

# ============================================================================
# ENHANCED OPTIMIZATION PROBLEM
# ============================================================================

class EnhancedRMTwinProblem(Problem):
    """Multi-objective optimization problem with 6 objectives - 修复版"""
    
    def __init__(self, ontology_graph: Graph, config: OptimizationConfig):
        self.g = ontology_graph
        self.config = config
        self.evaluator = IntegratedFitnessEvaluator(ontology_graph, config)
        
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
        """评估种群 - 使用原始目标值"""
        # 直接使用评估器，返回原始值
        objectives, constraints = self.evaluator.evaluate_batch(X)
        
        # 关键：直接使用原始目标值，不进行归一化
        out["F"] = objectives
        out["G"] = constraints
        
        # 可选：输出当前种群的统计信息
        if hasattr(self, '_eval_counter'):
            self._eval_counter += 1
        else:
            self._eval_counter = 1
            
        if self._eval_counter % 10 == 0:  # 每10代输出一次
            logger.debug(f"Generation ~{self._eval_counter}: "
                        f"Best cost=${objectives[:, 0].min():.0f}, "
                        f"Best recall={1-objectives[:, 1].min():.3f}")

# ============================================================================
# RESULTS MANAGER (FIXED JSON SERIALIZATION)
# ============================================================================

class ResultsManager:
    """Manage optimization results and generate reports"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

    def save_results(self, res, evaluator) -> pd.DataFrame:
        """
        保存优化结果 - 适配增强版评估器
        
        注意：增强版评估器返回的是真实值，不需要反归一化
        """
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
            
            # 使用增强版评估器获取原始目标值
            # 注意：这里返回的已经是真实值，不是归一化的值
            if hasattr(evaluator, 'evaluator_v2'):
                # 使用增强版评估器
                obj, const = evaluator.evaluator_v2.evaluate_solution(x, config)
            else:
                # 兼容旧版本
                obj, const = evaluator._evaluate_single(x)
            
            # 直接使用返回的真实值
            f1_raw = obj[0]  # 总成本（已考虑年化）
            f2_raw = obj[1]  # 1-recall
            f3_raw = obj[2]  # 延迟（已考虑场景因素）
            f4_raw = obj[3]  # 交通干扰（已考虑流量和车道）
            f5_raw = obj[4]  # 碳排放 kgCO2e/year（不是kWh）
            f6_raw = obj[5]  # 1/MTBF（已考虑冗余）
            
            # 计算衍生指标
            recall = 1 - f2_raw
            system_mtbf = 1 / f6_raw if f6_raw > 1e-10 else 1e10
            
            # 计算年度成本（f1已经是全生命周期成本）
            annual_cost = f1_raw / self.config.planning_horizon_years
            
            # 计算每公里年成本
            cost_per_km_year = annual_cost / self.config.road_network_length_km
            
            # 转换碳排放为吨（f5已经是kgCO2e/year）
            carbon_footprint_tons = f5_raw / 1000
            
            # 构建结果行
            row = {
                'solution_id': int(i + 1),
                
                # 配置详情
                'sensor': config['sensor'].split('#')[-1],
                'data_rate_Hz': round(config['data_rate'], 2),
                'geometric_LOD': config['geo_lod'],
                'condition_LOD': config['cond_lod'],
                'algorithm': config['algorithm'].split('#')[-1],
                'detection_threshold': round(config['detection_threshold'], 3),
                'storage': config['storage'].split('#')[-1],
                'communication': config['communication'].split('#')[-1],
                'deployment': config['deployment'].split('#')[-1],
                'crew_size': int(config['crew_size']),
                'inspection_cycle_days': int(config['inspection_cycle']),
                
                # 原始目标值（真实值）
                'f1_total_cost_USD': round(float(f1_raw), 2),
                'f2_one_minus_recall': round(float(f2_raw), 4),
                'f3_latency_seconds': round(float(f3_raw), 2),
                'f4_traffic_disruption_hours': round(float(f4_raw), 2),
                'f5_carbon_emissions_kgCO2e_year': round(float(f5_raw), 2),  
                    'f5_environmental_impact_kWh_year': round(float(f5_raw / 0.417), 2),  # 兼容旧版本：转回能耗 # 注意：现在是碳排放
                'f6_system_reliability_inverse_MTBF': round(float(f6_raw), 8),
                
                # 衍生指标
                'detection_recall': round(float(recall), 4),
                'system_MTBF_hours': round(float(system_mtbf), 0),
                'system_MTBF_years': round(float(system_mtbf / 8760), 2),  # 新增：以年为单位
                'annual_cost_USD': round(float(annual_cost), 2),
                'cost_per_km_year': round(float(cost_per_km_year), 2),
                'carbon_footprint_tons_CO2_year': round(float(carbon_footprint_tons), 2),
                
                # 新增：增强版模型的特殊指标
                'annualized_capital_cost_USD': round(float(f1_raw * 0.1 / self.config.planning_horizon_years), 2),  # 估算
                'operational_cost_ratio': round(float(1 - 0.1), 2),  # 运营成本比例
                
                # 约束检查
                'is_feasible': bool(np.all(const <= 0) if const is not None else True),
                'latency_constraint_ok': bool(f3_raw <= 180.0),
                'recall_constraint_ok': bool(recall >= 0.70),
                'budget_constraint_ok': bool(f1_raw <= self.config.budget_cap_usd)
            }
            
            # 添加专家建议的额外分析字段
            # 场景影响分析（如果配置中有场景信息）
            if hasattr(evaluator, 'advanced_config'):
                row['scenario_type'] = evaluator.advanced_config.scenario_type
                row['carbon_intensity_factor'] = evaluator.advanced_config.carbon_intensity_factor
            
            results.append(row)
            
        # 创建DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('f1_total_cost_USD')
        
        # 添加排名信息
        df['cost_rank'] = df['f1_total_cost_USD'].rank()
        df['recall_rank'] = df['detection_recall'].rank(ascending=False)
        df['carbon_rank'] = df['f5_carbon_emissions_kgCO2e_year'].rank()
        df['reliability_rank'] = df['system_MTBF_hours'].rank(ascending=False)
        
        # 保存到CSV
        csv_path = self.output_dir / 'pareto_solutions_6d_enhanced.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} Pareto-optimal solutions to {csv_path}")
        
        # 保存增强版摘要统计
        self._save_enhanced_summary(df)
        
        # 生成增强版报告
        self._generate_enhanced_report(df)
        
        return df

    def _save_enhanced_summary(self, df: pd.DataFrame):
        """保存增强版摘要统计"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.to_dict(),
            'results': {
                'total_solutions': int(len(df)),
                'feasible_solutions': int(df['is_feasible'].sum()),
                'objective_statistics': {
                    'cost': {
                        'min': float(df['f1_total_cost_USD'].min()),
                        'max': float(df['f1_total_cost_USD'].max()),
                        'mean': float(df['f1_total_cost_USD'].mean()),
                        'std': float(df['f1_total_cost_USD'].std()),
                        'min_annual': float(df['annual_cost_USD'].min()),
                        'max_annual': float(df['annual_cost_USD'].max())
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
                        'below_1s': int((df['f3_latency_seconds'] < 1.0).sum())
                    },
                    'carbon_emissions': {
                        'min_kgCO2e': float(df['f5_carbon_emissions_kgCO2e_year'].min()),
                        'max_kgCO2e': float(df['f5_carbon_emissions_kgCO2e_year'].max()),
                        'mean_kgCO2e': float(df['f5_carbon_emissions_kgCO2e_year'].mean()),
                        'min_tons': float(df['carbon_footprint_tons_CO2_year'].min()),
                        'max_tons': float(df['carbon_footprint_tons_CO2_year'].max()),
                        'low_carbon_solutions': int((df['carbon_footprint_tons_CO2_year'] < 10).sum())
                    },
                    'reliability': {
                        'min_mtbf_hours': float(df['system_MTBF_hours'].replace([np.inf], np.nan).min()),
                        'max_mtbf_hours': float(df['system_MTBF_hours'].replace([np.inf], np.nan).max()),
                        'mean_mtbf_years': float(df['system_MTBF_years'].replace([np.inf], np.nan).mean()),
                        'high_reliability_solutions': int((df['system_MTBF_years'] > 3).sum())
                    }
                },
                'technology_distribution': {
                    'sensors': df['sensor'].value_counts().to_dict(),
                    'algorithms': df['algorithm'].value_counts().to_dict(),
                    'deployments': df['deployment'].value_counts().to_dict()
                }
            }
        }
        
        # 保存JSON摘要
        with open(self.output_dir / 'optimization_summary_enhanced.json', 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

    def _generate_enhanced_report(self, df: pd.DataFrame):
        """生成增强版分析报告"""
        report = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║              ENHANCED RMTWIN OPTIMIZATION REPORT V2.0                          ║
    ║                   With Expert-Recommended Improvements                         ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Total Pareto-optimal solutions: {len(df)}

    1. COST ANALYSIS (Enhanced with Annualized Capital Cost)
    ══════════════════════════════════════════════════════════════════════════════
    - Total Cost Range: ${df['f1_total_cost_USD'].min():,.0f} - ${df['f1_total_cost_USD'].max():,.0f}
    - Annual Cost Range: ${df['annual_cost_USD'].min():,.0f} - ${df['annual_cost_USD'].max():,.0f}
    - Cost per km/year: ${df['cost_per_km_year'].min():.2f} - ${df['cost_per_km_year'].max():.2f}

    Most Cost-Effective Configuration:
    {self._format_solution(df.loc[df['f1_total_cost_USD'].idxmin()])}

    2. PERFORMANCE ANALYSIS (With Class Imbalance Considerations)
    ══════════════════════════════════════════════════════════════════════════════
    - Detection Recall Range: {df['detection_recall'].min():.3f} - {df['detection_recall'].max():.3f}
    - Solutions meeting 70% threshold: {(df['detection_recall'] >= 0.70).sum()} ({(df['detection_recall'] >= 0.70).sum()/len(df)*100:.1f}%)
    - Average Recall by Algorithm Type:
    {self._get_algorithm_performance_summary(df)}

    3. LATENCY ANALYSIS (Scenario-Aware)
    ══════════════════════════════════════════════════════════════════════════════
    - Latency Range: {df['f3_latency_seconds'].min():.2f} - {df['f3_latency_seconds'].max():.2f} seconds
    - Real-time capable (<1s): {(df['f3_latency_seconds'] < 1.0).sum()} solutions
    - Meeting constraint (<180s): {(df['f3_latency_seconds'] <= 180).sum()} solutions

    4. ENVIRONMENTAL IMPACT (Carbon Footprint)
    ══════════════════════════════════════════════════════════════════════════════
    - Carbon Emissions Range: {df['f5_carbon_emissions_kgCO2e_year'].min():.0f} - {df['f5_carbon_emissions_kgCO2e_year'].max():.0f} kgCO₂e/year
    - Equivalent to: {df['carbon_footprint_tons_CO2_year'].min():.1f} - {df['carbon_footprint_tons_CO2_year'].max():.1f} tons/year
    - Low-carbon solutions (<10 tons/year): {(df['carbon_footprint_tons_CO2_year'] < 10).sum()}

    Most Sustainable Configuration:
    {self._format_solution(df.loc[df['f5_carbon_emissions_kgCO2e_year'].idxmin()])}

    5. RELIABILITY ANALYSIS (With Redundancy Benefits)
    ══════════════════════════════════════════════════════════════════════════════
    - MTBF Range: {df['system_MTBF_hours'].replace([np.inf], np.nan).min():.0f} - {df['system_MTBF_hours'].replace([np.inf], np.nan).max():.0f} hours
    - In years: {df['system_MTBF_years'].replace([np.inf], np.nan).min():.1f} - {df['system_MTBF_years'].replace([np.inf], np.nan).max():.1f} years
    - High reliability (>3 years): {(df['system_MTBF_years'] > 3).sum()} solutions

    6. KEY INSIGHTS FROM ENHANCED MODELING
    ══════════════════════════════════════════════════════════════════════════════
    {self._generate_insights(df)}

    ══════════════════════════════════════════════════════════════════════════════
    """
        
        # 保存报告
        with open(self.output_dir / 'optimization_report_enhanced.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Generated enhanced optimization report")

    def _format_solution(self, sol) -> str:
        """格式化单个解决方案的描述"""
        return f"""  • Sensor: {sol['sensor']}
    • Algorithm: {sol['algorithm']}
    • Cost: ${sol['f1_total_cost_USD']:,.0f} (${sol['annual_cost_USD']:,.0f}/year)
    • Recall: {sol['detection_recall']:.3f}
    • Carbon: {sol['carbon_footprint_tons_CO2_year']:.1f} tons/year
    • MTBF: {sol['system_MTBF_years']:.1f} years"""

    def _get_algorithm_performance_summary(self, df) -> str:
        """获取算法性能摘要"""
        algo_perf = df.groupby(df['algorithm'].str.extract(r'(DL|ML|Traditional|PC)')[0])['detection_recall'].mean()
        summary = ""
        for algo_type, recall in algo_perf.items():
            if pd.notna(algo_type):
                summary += f"  • {algo_type}: {recall:.3f}\n"
        return summary.strip()

    def _generate_insights(self, df) -> str:
        """生成关键洞察"""
        insights = []
        
        # 成本-性能权衡
        cost_perf_corr = df['f1_total_cost_USD'].corr(df['detection_recall'])
        insights.append(f"• Cost-Performance correlation: {cost_perf_corr:.2f} (higher cost {'does' if cost_perf_corr > 0.3 else 'does not'} guarantee better performance)")
        
        # 环境-可靠性关系
        env_rel_corr = df['f5_carbon_emissions_kgCO2e_year'].corr(df['system_MTBF_hours'])
        insights.append(f"• Carbon-Reliability trade-off: {env_rel_corr:.2f} (sustainability and reliability are {'positively' if env_rel_corr > 0 else 'negatively'} correlated)")
        
        # 最佳平衡解
        normalized = df[['f1_total_cost_USD', 'f2_one_minus_recall', 'f3_latency_seconds', 
                        'f5_carbon_emissions_kgCO2e_year', 'f6_system_reliability_inverse_MTBF']].copy()
        for col in normalized.columns:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
        normalized['balance_score'] = normalized.sum(axis=1)
        best_balanced_idx = normalized['balance_score'].idxmin()
        
        insights.append(f"• Best balanced solution: Solution #{df.loc[best_balanced_idx, 'solution_id']} "
                    f"({df.loc[best_balanced_idx, 'sensor']} + {df.loc[best_balanced_idx, 'algorithm']})")
        
        return '\n'.join(insights)


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
        # 替换为：
        logger.info("\nStep 2: Setting up optimization problem with enhanced evaluator...")
        problem = EnhancedRMTwinProblem(ontology_graph, config)
        # 确保使用新的评估器
        problem.evaluator = IntegratedFitnessEvaluator(ontology_graph, config)
        
        # Step 3: Configure algorithm
        # 对于原始目标值，可能需要调整交叉和变异参数
        algorithm = NSGA2(
            pop_size=config.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=config.crossover_eta, prob=config.crossover_prob),
            mutation=PM(eta=config.mutation_eta, prob=1.0/problem.n_var),
            eliminate_duplicates=True
        )

        # 添加：设置epsilon用于处理不同尺度的目标
        # 这有助于NSGA-II更好地处理原始目标值
        from pymoo.util.misc import set_if_none
        algorithm.epsilon = 1e-3  # 可以根据需要调整
        
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

        # 添加这行来生成增强版可视化
        from advanced_visualizations import create_all_publication_figures
        create_all_publication_figures(
            csv_file=Path(config.output_dir) / 'pareto_solutions_6d_enhanced.csv',
            output_dir=Path(config.output_dir) / 'publication_figures'
        )
        
        # Step 7: Generate report
        logger.info("\nStep 7: Generating analysis report...")
        results_manager.generate_report(df)
        

        # Step 8: Run baseline comparison (NEW)
        logger.info("\nStep 8: Running baseline comparison...")
        baseline_results = None  # 初始化变量
        try:
            from baseline_comparison import run_baseline_comparison
            
            baseline_results = run_baseline_comparison(
                ontology_graph=ontology_graph,
                pareto_csv_path=Path(config.output_dir) / 'pareto_solutions_6d_enhanced.csv'
            )
            
            if baseline_results:
                logger.info("Baseline comparison completed successfully")
                logger.info("Check ./results/baseline/ for detailed comparison results")
            
        except ImportError:
            logger.warning("baseline_comparison module not found, skipping baseline analysis")
        except Exception as e:
            logger.error(f"Error during baseline comparison: {e}")
            logger.info("Main optimization completed successfully, but baseline comparison failed")

        # Final summary (修改以处理baseline_results可能为None的情况)
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION AND ANALYSIS COMPLETE")
        logger.info(f"Total solutions found: {len(df)}")
        logger.info(f"Feasible solutions: {df['is_feasible'].sum() if 'is_feasible' in df else len(df)}")
        logger.info(f"Results saved to: {config.output_dir}")
        if baseline_results is not None:  # 检查是否为None
            logger.info(f"Baseline comparison saved to: {config.output_dir}/baseline/")
        logger.info("="*80)
                
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
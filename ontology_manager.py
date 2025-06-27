#!/usr/bin/env python3
"""
Ontology Manager for RMTwin Optimization
Handles ontology loading and population
"""

import logging
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)

# Namespaces
RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class OntologyManager:
    """Manages RDTcO-Maint ontology operations"""
    
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
        """Set up base ontology structure"""
        logger.info("Setting up base ontology structure...")
        
        # Define core classes
        core_classes = [
            'DigitalTwinConfiguration',
            'SensorSystem',
            'Algorithm',
            'StorageSystem',
            'CommunicationSystem',
            'ComputeDeployment',
            'ConfigurationParameter'
        ]
        
        for class_name in core_classes:
            class_uri = RDTCO[class_name]
            self.g.add((class_uri, RDF.type, OWL.Class))
            self.g.add((class_uri, RDFS.label, Literal(class_name)))
        
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
        algo_types = [
            'DeepLearningAlgorithm', 'MachineLearningAlgorithm',
            'TraditionalAlgorithm', 'PointCloudAlgorithm'
        ]
        
        for algo_type in algo_types:
            algo_class = RDTCO[algo_type]
            self.g.add((algo_class, RDF.type, OWL.Class))
            self.g.add((algo_class, RDFS.subClassOf, RDTCO.Algorithm))
            self.g.add((algo_class, RDFS.label, Literal(algo_type)))
        
        # Define properties
        self._define_properties()
        
        logger.info(f"Base ontology created with {len(self.g)} triples")
    
    def _define_properties(self):
        """Define ontology properties"""
        properties = [
            # Cost properties
            ('hasInitialCostUSD', 'Initial cost in USD', XSD.decimal),
            ('hasOperationalCostUSDPerDay', 'Daily operational cost', XSD.decimal),
            ('hasAnnualOpCostUSD', 'Annual operational cost', XSD.decimal),
            
            # Performance properties
            ('hasRecall', 'Detection recall rate', XSD.decimal),
            ('hasPrecision', 'Detection precision rate', XSD.decimal),
            ('hasFPS', 'Frames per second', XSD.decimal),
            ('hasAccuracyRangeMM', 'Accuracy range in mm', XSD.decimal),
            
            # Technical properties
            ('hasEnergyConsumptionW', 'Energy consumption in watts', XSD.decimal),
            ('hasMTBFHours', 'Mean time between failures in hours', XSD.decimal),
            ('hasDataVolumeGBPerKm', 'Data volume GB per km', XSD.decimal),
            ('hasCoverageEfficiencyKmPerDay', 'Coverage efficiency km per day', XSD.decimal),
            ('hasOperatingSpeedKmh', 'Operating speed km/h', XSD.decimal),
            
            # Operational properties
            ('hasOperatorSkillLevel', 'Required operator skill level', XSD.string),
            ('hasCalibrationFreqMonths', 'Calibration frequency in months', XSD.decimal),
            ('hasDataAnnotationCostUSD', 'Data annotation cost per image', XSD.decimal),
            ('hasModelRetrainingFreqMonths', 'Model retraining frequency', XSD.decimal),
            
            # Quality properties
            ('hasExplainabilityScore', 'Model explainability score 1-5', XSD.integer),
            ('hasIntegrationComplexity', 'Integration complexity 1-5', XSD.integer),
            ('hasCybersecurityVulnerability', 'Security vulnerability 1-5', XSD.integer),
            
            # Configuration properties
            ('isDecisionVariable', 'Marks a decision variable', XSD.boolean),
            ('hasMinValue', 'Minimum value constraint', XSD.decimal),
            ('hasMaxValue', 'Maximum value constraint', XSD.decimal),
            ('hasMutuallyExclusiveWith', 'Mutually exclusive component', XSD.anyURI),
            ('hasSynergyWith', 'Synergistic component', XSD.anyURI)
        ]
        
        for prop_name, comment, range_type in properties:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))
            self.g.add((prop_uri, RDFS.range, range_type))
    
    def populate_from_csv_files(self, sensor_csv: str, algorithm_csv: str,
                               infrastructure_csv: str, cost_benefit_csv: str) -> Graph:
        """Populate ontology from CSV files"""
        logger.info("Populating ontology from CSV files...")
        
        # Load each type of data
        self._load_sensors(sensor_csv)
        self._load_algorithms(algorithm_csv)
        self._load_infrastructure(infrastructure_csv)
        self._load_cost_benefit(cost_benefit_csv)
        
        # Add SHACL constraints
        self._add_shacl_constraints()
        
        logger.info(f"Ontology populated with {len(self.g)} triples")
        return self.g
    
    def _load_sensors(self, filepath: str):
        """Load sensor data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} sensor instances...")
            
            for _, row in df.iterrows():
                sensor_uri = EX[row['Sensor_Instance_Name']]
                sensor_type = RDTCO[row['Sensor_RDF_Type']]
                
                # Add instance
                self.g.add((sensor_uri, RDF.type, sensor_type))
                self.g.add((sensor_uri, RDFS.label, Literal(row['Sensor_Instance_Name'])))
                
                # Add properties with safe conversion
                self._add_property(sensor_uri, RDTCO.hasInitialCostUSD, 
                                 row['Initial_Cost_USD'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasOperationalCostUSDPerDay,
                                 row['Operational_Cost_USD_per_day'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasEnergyConsumptionW,
                                 row['Energy_Consumption_W'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasMTBFHours,
                                 row['MTBF_hours'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasOperatorSkillLevel,
                                 row['Operator_Skill_Level'], XSD.string)
                
                # Optional properties
                if pd.notna(row.get('Calibration_Freq_months')) and row['Calibration_Freq_months'] != 'N/A':
                    self._add_property(sensor_uri, RDTCO.hasCalibrationFreqMonths,
                                     row['Calibration_Freq_months'], XSD.decimal)
                
                # Technical properties
                self._add_property(sensor_uri, RDTCO.hasAccuracyRangeMM,
                                 row['Accuracy_Range_mm'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasDataVolumeGBPerKm,
                                 row['Data_Volume_GB_per_km'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasCoverageEfficiencyKmPerDay,
                                 row['Coverage_Efficiency_km_per_day'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasOperatingSpeedKmh,
                                 row['Operating_Speed_kmh'], XSD.decimal)
                
        except Exception as e:
            logger.error(f"Error loading sensors: {e}")
            raise
    
    def _load_algorithms(self, filepath: str):
        """Load algorithm data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} algorithm instances...")
            
            for _, row in df.iterrows():
                algo_uri = EX[row['Algorithm_Instance_Name']]
                algo_type = RDTCO[row['Algorithm_RDF_Type']]
                
                # Add instance
                self.g.add((algo_uri, RDF.type, algo_type))
                self.g.add((algo_uri, RDFS.label, Literal(row['Algorithm_Instance_Name'])))
                
                # Performance metrics
                self._add_property(algo_uri, RDTCO.hasPrecision,
                                 row['Precision'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasRecall,
                                 row['Recall'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasFPS,
                                 row['FPS'], XSD.decimal)
                
                # Cost and maintenance
                self._add_property(algo_uri, RDTCO.hasDataAnnotationCostUSD,
                                 row['Data_Annotation_Cost_USD'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasModelRetrainingFreqMonths,
                                 row['Model_Retraining_Freq_months'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasExplainabilityScore,
                                 row['Explainability_Score'], XSD.integer)
                
                # Hardware requirements
                self.g.add((algo_uri, RDTCO.hasHardwareRequirement,
                          Literal(str(row['Hardware_Requirement']))))
                
        except Exception as e:
            logger.error(f"Error loading algorithms: {e}")
            raise
    
    def _load_infrastructure(self, filepath: str):
        """Load infrastructure component data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} infrastructure instances...")
            
            for _, row in df.iterrows():
                comp_uri = EX[row['Component_Instance_Name']]
                comp_type = RDTCO[row['Component_RDF_Type']]
                
                # Add instance
                self.g.add((comp_uri, RDF.type, comp_type))
                self.g.add((comp_uri, RDFS.label, Literal(row['Component_Instance_Name'])))
                
                # Costs
                self._add_property(comp_uri, RDTCO.hasInitialCostUSD,
                                 row['Initial_Cost_USD'], XSD.decimal)
                self._add_property(comp_uri, RDTCO.hasAnnualOpCostUSD,
                                 row['Annual_OpCost_USD'], XSD.decimal)
                
                # Technical properties
                if pd.notna(row.get('Energy_Consumption_W')) and row['Energy_Consumption_W'] != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasEnergyConsumptionW,
                                     row['Energy_Consumption_W'], XSD.decimal)
                
                if pd.notna(row.get('MTBF_hours')) and row['MTBF_hours'] != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasMTBFHours,
                                     row['MTBF_hours'], XSD.decimal)
                
                if pd.notna(row.get('Integration_Complexity')) and row['Integration_Complexity'] != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasIntegrationComplexity,
                                     row['Integration_Complexity'], XSD.integer)
                
                if pd.notna(row.get('Cybersecurity_Vulnerability')) and row['Cybersecurity_Vulnerability'] != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasCybersecurityVulnerability,
                                     row['Cybersecurity_Vulnerability'], XSD.integer)
                
        except Exception as e:
            logger.error(f"Error loading infrastructure: {e}")
            raise
    
    def _load_cost_benefit(self, filepath: str):
        """Load cost-benefit data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} cost-benefit entries...")
            
            for _, row in df.iterrows():
                param_uri = EX[f"Parameter_{row['Metric_Name'].replace(' ', '_')}"]
                
                self.g.add((param_uri, RDF.type, RDTCO.ConfigurationParameter))
                self.g.add((param_uri, RDFS.label, Literal(row['Metric_Name'])))
                self.g.add((param_uri, RDTCO.hasValue, 
                          Literal(float(row['Value']), datatype=XSD.decimal)))
                self.g.add((param_uri, RDTCO.hasUnit, Literal(str(row['Unit']))))
                
                if pd.notna(row.get('Notes')):
                    self.g.add((param_uri, RDFS.comment, Literal(str(row['Notes']))))
                
        except Exception as e:
            logger.error(f"Error loading cost-benefit data: {e}")
            raise
    
    def _add_property(self, subject: URIRef, predicate: URIRef, 
                     value: any, datatype: URIRef):
        """Add property with safe type conversion"""
        if pd.notna(value) and str(value) != 'N/A':
            try:
                if datatype == XSD.decimal:
                    value = float(value)
                elif datatype == XSD.integer:
                    value = int(float(value))
                
                self.g.add((subject, predicate, Literal(value, datatype=datatype)))
            except:
                # Fallback to string
                self.g.add((subject, predicate, Literal(str(value))))
    
    def _add_shacl_constraints(self):
        """Add SHACL constraints for validation"""
        logger.info("Adding SHACL constraints...")
        
        # This is a placeholder for SHACL shapes
        # In a complete implementation, you would define shapes like:
        # - Budget constraint shape
        # - Sensor compatibility shape
        # - Algorithm requirement shape
        # etc.
        
        # Example shape for budget constraint
        budget_shape = BNode()
        self.g.add((budget_shape, RDF.type, URIRef("http://www.w3.org/ns/shacl#PropertyShape")))
        self.g.add((budget_shape, URIRef("http://www.w3.org/ns/shacl#path"), RDTCO.hasTotalCost))
        self.g.add((budget_shape, URIRef("http://www.w3.org/ns/shacl#maxInclusive"), 
                   Literal(10000000, datatype=XSD.decimal)))
        
    def validate_configuration(self, config_uri: URIRef) -> bool:
        """Validate a configuration against constraints"""
        # Placeholder for validation logic
        return True
    
    def save_ontology(self, filepath: str, format: str = 'turtle'):
        """Save ontology to file"""
        self.g.serialize(destination=filepath, format=format)
        logger.info(f"Saved ontology to {filepath}")
#!/usr/bin/env python3
"""
本体管理器 - 修复版
改进了组件提取逻辑以正确识别所有传感器
"""

import logging
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)

# 命名空间
RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class OntologyManager:
    """管理RDTcO-Maint本体操作"""
    
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
        """设置基础本体结构"""
        logger.info("设置基础本体结构...")
        
        # 定义核心类
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
        
        # 定义传感器子类
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
        
        # 定义算法子类
        algo_types = [
            'DeepLearningAlgorithm', 'MachineLearningAlgorithm',
            'TraditionalAlgorithm', 'PointCloudAlgorithm'
        ]
        
        for algo_type in algo_types:
            algo_class = RDTCO[algo_type]
            self.g.add((algo_class, RDF.type, OWL.Class))
            self.g.add((algo_class, RDFS.subClassOf, RDTCO.Algorithm))
            self.g.add((algo_class, RDFS.label, Literal(algo_type)))
        
        # 定义属性
        self._define_properties()
        
        logger.info(f"基础本体创建完成，包含 {len(self.g)} 个三元组")
    
    def _define_properties(self):
        """定义本体属性"""
        properties = [
            # 成本属性
            ('hasInitialCostUSD', '初始成本（美元）', XSD.decimal),
            ('hasOperationalCostUSDPerDay', '日运营成本', XSD.decimal),
            ('hasAnnualOpCostUSD', '年运营成本', XSD.decimal),
            
            # 性能属性
            ('hasRecall', '检测召回率', XSD.decimal),
            ('hasPrecision', '检测精确率', XSD.decimal),
            ('hasFPS', '每秒帧数', XSD.decimal),
            ('hasAccuracyRangeMM', '精度范围（毫米）', XSD.decimal),
            
            # 技术属性
            ('hasEnergyConsumptionW', '能耗（瓦）', XSD.decimal),
            ('hasMTBFHours', '平均故障间隔时间（小时）', XSD.decimal),
            ('hasDataVolumeGBPerKm', '每公里数据量（GB）', XSD.decimal),
            ('hasCoverageEfficiencyKmPerDay', '每天覆盖效率（公里）', XSD.decimal),
            ('hasOperatingSpeedKmh', '运行速度（公里/小时）', XSD.decimal),
            
            # 运营属性
            ('hasOperatorSkillLevel', '操作员技能等级', XSD.string),
            ('hasCalibrationFreqMonths', '校准频率（月）', XSD.decimal),
            ('hasDataAnnotationCostUSD', '数据标注成本（美元/张）', XSD.decimal),
            ('hasModelRetrainingFreqMonths', '模型重训练频率', XSD.decimal),
            
            # 质量属性
            ('hasExplainabilityScore', '模型可解释性评分 1-5', XSD.integer),
            ('hasIntegrationComplexity', '集成复杂度 1-5', XSD.integer),
            ('hasCybersecurityVulnerability', '安全漏洞 1-5', XSD.integer),
            
            # 配置属性
            ('isDecisionVariable', '标记决策变量', XSD.boolean),
            ('hasMinValue', '最小值约束', XSD.decimal),
            ('hasMaxValue', '最大值约束', XSD.decimal),
            ('hasMutuallyExclusiveWith', '互斥组件', XSD.anyURI),
            ('hasSynergyWith', '协同组件', XSD.anyURI)
        ]
        
        for prop_name, comment, range_type in properties:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))
            self.g.add((prop_uri, RDFS.range, range_type))
    
    def populate_from_csv_files(self, sensor_csv: str, algorithm_csv: str,
                               infrastructure_csv: str, cost_benefit_csv: str) -> Graph:
        """从CSV文件填充本体"""
        logger.info("从CSV文件填充本体...")
        
        # 加载每种类型的数据
        self._load_sensors(sensor_csv)
        self._load_algorithms(algorithm_csv)
        self._load_infrastructure(infrastructure_csv)
        self._load_cost_benefit(cost_benefit_csv)
        
        # 添加SHACL约束
        self._add_shacl_constraints()
        
        logger.info(f"本体填充完成，包含 {len(self.g)} 个三元组")
        
        # 验证加载的组件
        self._verify_loaded_components()
        
        return self.g
    
    def _load_sensors(self, filepath: str):
        """加载传感器数据"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"加载 {len(df)} 个传感器实例...")
            
            for _, row in df.iterrows():
                sensor_uri = EX[row['Sensor_Instance_Name']]
                sensor_type = RDTCO[row['Sensor_RDF_Type']]
                
                # 添加实例
                self.g.add((sensor_uri, RDF.type, sensor_type))
                self.g.add((sensor_uri, RDFS.label, Literal(row['Sensor_Instance_Name'])))
                
                # 添加属性（安全转换）
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
                
                # 可选属性
                if pd.notna(row.get('Calibration_Freq_months')) and str(row['Calibration_Freq_months']) != 'N/A':
                    self._add_property(sensor_uri, RDTCO.hasCalibrationFreqMonths,
                                     row['Calibration_Freq_months'], XSD.decimal)
                
                # 技术属性
                self._add_property(sensor_uri, RDTCO.hasAccuracyRangeMM,
                                 row['Accuracy_Range_mm'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasDataVolumeGBPerKm,
                                 row['Data_Volume_GB_per_km'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasCoverageEfficiencyKmPerDay,
                                 row['Coverage_Efficiency_km_per_day'], XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasOperatingSpeedKmh,
                                 row['Operating_Speed_kmh'], XSD.decimal)
                
        except Exception as e:
            logger.error(f"加载传感器时出错: {e}")
            raise
    
    def _load_algorithms(self, filepath: str):
        """加载算法数据"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"加载 {len(df)} 个算法实例...")
            
            for _, row in df.iterrows():
                algo_uri = EX[row['Algorithm_Instance_Name']]
                algo_type = RDTCO[row['Algorithm_RDF_Type']]
                
                # 添加实例
                self.g.add((algo_uri, RDF.type, algo_type))
                self.g.add((algo_uri, RDFS.label, Literal(row['Algorithm_Instance_Name'])))
                
                # 性能指标
                self._add_property(algo_uri, RDTCO.hasPrecision,
                                 row['Precision'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasRecall,
                                 row['Recall'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasFPS,
                                 row['FPS'], XSD.decimal)
                
                # 成本和维护
                self._add_property(algo_uri, RDTCO.hasDataAnnotationCostUSD,
                                 row['Data_Annotation_Cost_USD'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasModelRetrainingFreqMonths,
                                 row['Model_Retraining_Freq_months'], XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasExplainabilityScore,
                                 row['Explainability_Score'], XSD.integer)
                
                # 硬件需求
                self.g.add((algo_uri, RDTCO.hasHardwareRequirement,
                          Literal(str(row['Hardware_Requirement']))))
                
        except Exception as e:
            logger.error(f"加载算法时出错: {e}")
            raise
    
    def _load_infrastructure(self, filepath: str):
        """加载基础设施组件数据"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"加载 {len(df)} 个基础设施实例...")
            
            for _, row in df.iterrows():
                comp_uri = EX[row['Component_Instance_Name']]
                comp_type = RDTCO[row['Component_RDF_Type']]
                
                # 添加实例
                self.g.add((comp_uri, RDF.type, comp_type))
                self.g.add((comp_uri, RDFS.label, Literal(row['Component_Instance_Name'])))
                
                # 成本
                self._add_property(comp_uri, RDTCO.hasInitialCostUSD,
                                 row['Initial_Cost_USD'], XSD.decimal)
                self._add_property(comp_uri, RDTCO.hasAnnualOpCostUSD,
                                 row['Annual_OpCost_USD'], XSD.decimal)
                
                # 技术属性
                if pd.notna(row.get('Energy_Consumption_W')) and str(row['Energy_Consumption_W']) != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasEnergyConsumptionW,
                                     row['Energy_Consumption_W'], XSD.decimal)
                
                if pd.notna(row.get('MTBF_hours')) and str(row['MTBF_hours']) != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasMTBFHours,
                                     row['MTBF_hours'], XSD.decimal)
                
                if pd.notna(row.get('Integration_Complexity')) and str(row['Integration_Complexity']) != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasIntegrationComplexity,
                                     row['Integration_Complexity'], XSD.integer)
                
                if pd.notna(row.get('Cybersecurity_Vulnerability')) and str(row['Cybersecurity_Vulnerability']) != 'N/A':
                    self._add_property(comp_uri, RDTCO.hasCybersecurityVulnerability,
                                     row['Cybersecurity_Vulnerability'], XSD.integer)
                
        except Exception as e:
            logger.error(f"加载基础设施时出错: {e}")
            raise
    
    def _load_cost_benefit(self, filepath: str):
        """加载成本效益数据"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"加载 {len(df)} 个成本效益条目...")
            
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
            logger.error(f"加载成本效益数据时出错: {e}")
            raise
    
    def _add_property(self, subject: URIRef, predicate: URIRef, 
                     value: any, datatype: URIRef):
        """添加属性（安全类型转换）"""
        if pd.notna(value) and str(value) != 'N/A':
            try:
                if datatype == XSD.decimal:
                    value = float(value)
                elif datatype == XSD.integer:
                    value = int(float(value))
                
                self.g.add((subject, predicate, Literal(value, datatype=datatype)))
            except:
                # 失败时回退到字符串
                self.g.add((subject, predicate, Literal(str(value))))
    
    def _add_shacl_constraints(self):
        """添加SHACL约束以进行验证"""
        logger.info("添加SHACL约束...")
        
        # 预算约束形状
        budget_shape = BNode()
        self.g.add((budget_shape, RDF.type, URIRef("http://www.w3.org/ns/shacl#PropertyShape")))
        self.g.add((budget_shape, URIRef("http://www.w3.org/ns/shacl#path"), RDTCO.hasTotalCost))
        self.g.add((budget_shape, URIRef("http://www.w3.org/ns/shacl#maxInclusive"), 
                   Literal(20000000, datatype=XSD.decimal)))
        
        # 最小召回率约束
        recall_shape = BNode()
        self.g.add((recall_shape, RDF.type, URIRef("http://www.w3.org/ns/shacl#PropertyShape")))
        self.g.add((recall_shape, URIRef("http://www.w3.org/ns/shacl#path"), RDTCO.hasRecall))
        self.g.add((recall_shape, URIRef("http://www.w3.org/ns/shacl#minInclusive"), 
                   Literal(0.65, datatype=XSD.decimal)))
    
    def _verify_loaded_components(self):
        """验证加载的组件"""
        # 查询所有传感器实例
        sensor_query = """
        SELECT DISTINCT ?sensor WHERE {
            ?sensor rdf:type ?type .
            FILTER(CONTAINS(STR(?type), "Sensor") || 
                   ?sensor rdf:type rdtco:MMS_LiDAR_System ||
                   ?sensor rdf:type rdtco:MMS_Camera_System ||
                   ?sensor rdf:type rdtco:UAV_LiDAR_System ||
                   ?sensor rdf:type rdtco:UAV_Camera_System ||
                   ?sensor rdf:type rdtco:TLS_System ||
                   ?sensor rdf:type rdtco:Handheld_3D_Scanner ||
                   ?sensor rdf:type rdtco:FiberOptic_Sensor ||
                   ?sensor rdf:type rdtco:Vehicle_LowCost_Sensor ||
                   ?sensor rdf:type rdtco:IoT_Network_System)
        }
        """
        
        sensors = list(self.g.query(sensor_query))
        logger.info(f"验证：找到 {len(sensors)} 个传感器实例")
        
        # 查询算法
        algo_query = """
        SELECT DISTINCT ?algo WHERE {
            ?algo rdf:type ?type .
            FILTER(CONTAINS(STR(?type), "Algorithm"))
        }
        """
        
        algorithms = list(self.g.query(algo_query))
        logger.info(f"验证：找到 {len(algorithms)} 个算法实例")
    
    def validate_configuration(self, config_uri: URIRef) -> bool:
        """验证配置是否符合约束"""
        # 占位符，用于验证逻辑
        return True
    
    def save_ontology(self, filepath: str, format: str = 'turtle'):
        """保存本体到文件"""
        self.g.serialize(destination=filepath, format=format)
        logger.info(f"本体保存到 {filepath}")
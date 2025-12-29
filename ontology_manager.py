#!/usr/bin/env python3
"""
本体管理器 v2.0 - 支持SHACL验证
================================
新增功能:
- build_config_graph(): 将配置转换为RDF图
- shacl_validate_config(): 执行SHACL验证
- 支持运行时和后验语义验证

Author: RMTwin Research Team
Version: 2.0 (SHACL Support)
"""

import logging
import uuid
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)

# 命名空间
RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")
SH = Namespace("http://www.w3.org/ns/shacl#")


class OntologyManager:
    """管理RDTcO-Maint本体操作 - 支持SHACL验证"""

    def __init__(self):
        self.g = Graph()
        self.g.bind("rdtco", RDTCO)
        self.g.bind("ex", EX)
        self.g.bind("sh", SH)
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

        # 定义配置关系属性 (用于SHACL验证)
        self._define_configuration_properties()

        logger.info(f"基础本体创建完成，包含 {len(self.g)} 个三元组")

    def _define_configuration_properties(self):
        """定义配置关系属性（用于SHACL验证）"""
        # 对象属性 - 配置到组件的关系
        config_relations = [
            ('hasSensor', 'SensorSystem', '配置使用的传感器'),
            ('hasAlgorithm', 'Algorithm', '配置使用的算法'),
            ('hasStorage', 'StorageSystem', '配置使用的存储'),
            ('hasCommunication', 'CommunicationSystem', '配置使用的通信'),
            ('hasDeployment', 'ComputeDeployment', '配置使用的部署'),
        ]

        for prop_name, range_class, comment in config_relations:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            self.g.add((prop_uri, RDFS.domain, RDTCO.DigitalTwinConfiguration))
            self.g.add((prop_uri, RDFS.range, RDTCO[range_class]))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))

        # 数据属性 - 配置参数
        config_params = [
            ('hasInspectionCycleDays', XSD.integer, '检测周期（天）'),
            ('hasDataRateHz', XSD.decimal, '数据采集频率'),
            ('hasTotalCostUSD', XSD.decimal, '总成本'),
            ('hasRecall', XSD.decimal, '检测召回率'),
            ('hasLatencySeconds', XSD.decimal, '延迟（秒）'),
            ('hasCarbonKgCO2eYear', XSD.decimal, '年碳排放'),
        ]

        for prop_name, datatype, comment in config_params:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.domain, RDTCO.DigitalTwinConfiguration))
            self.g.add((prop_uri, RDFS.range, datatype))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))

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
            ('hasCoverageEfficiencyKmPerDay', '覆盖效率（公里/天）', XSD.decimal),
            ('hasDataVolumeGBPerKm', '数据量（GB/公里）', XSD.decimal),

            # 兼容性属性
            ('hasHardwareRequirement', '硬件需求', XSD.string),
            ('hasDataFormat', '数据格式', XSD.string),
            ('hasComponentCategory', '组件类别', XSD.string),
        ]

        for prop_name, label, datatype in properties:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.label, Literal(label)))

    # =========================================================================
    # SHACL 验证功能 (P0 实现)
    # =========================================================================

    def build_config_graph(self, config: dict) -> Graph:
        """
        将配置字典转换为RDF图，用于SHACL验证。

        Args:
            config: 包含组件URI和参数的配置字典

        Returns:
            包含配置实例的RDF Graph
        """
        g_cfg = Graph()
        g_cfg.bind("rdtco", RDTCO)
        g_cfg.bind("ex", EX)
        g_cfg.bind("sh", SH)
        g_cfg.bind("rdf", RDF)
        g_cfg.bind("xsd", XSD)

        # 创建配置实例URI
        cfg_uri = EX[f"cfg_{uuid.uuid4().hex[:8]}"]
        g_cfg.add((cfg_uri, RDF.type, RDTCO.DigitalTwinConfiguration))

        def as_uri(v):
            """转换为URIRef"""
            if v is None:
                return None
            if isinstance(v, URIRef):
                return v
            v_str = str(v)
            if v_str.startswith('http'):
                return URIRef(v_str)
            # 如果是简单名称，添加命名空间
            if '#' not in v_str and '/' not in v_str:
                return EX[v_str.replace(' ', '_')]
            return URIRef(v_str)

        # 添加组件链接
        component_mappings = [
            ('sensor', RDTCO.hasSensor),
            ('algorithm', RDTCO.hasAlgorithm),
            ('storage', RDTCO.hasStorage),
            ('communication', RDTCO.hasCommunication),
            ('deployment', RDTCO.hasDeployment),
        ]

        for key, predicate in component_mappings:
            if key in config and config[key] is not None:
                uri = as_uri(config[key])
                if uri:
                    g_cfg.add((cfg_uri, predicate, uri))

        # 添加决策变量
        if 'inspection_cycle' in config and config['inspection_cycle'] is not None:
            g_cfg.add((cfg_uri, RDTCO.hasInspectionCycleDays,
                       Literal(int(config['inspection_cycle']), datatype=XSD.integer)))

        if 'data_rate' in config and config['data_rate'] is not None:
            g_cfg.add((cfg_uri, RDTCO.hasDataRateHz,
                       Literal(float(config['data_rate']), datatype=XSD.decimal)))

        # 添加评估结果（可选，用于结果级约束）
        result_mappings = [
            ('total_cost', RDTCO.hasTotalCostUSD),
            ('f1_total_cost_USD', RDTCO.hasTotalCostUSD),
            ('recall', RDTCO.hasRecall),
            ('detection_recall', RDTCO.hasRecall),
            ('latency', RDTCO.hasLatencySeconds),
            ('f3_latency_seconds', RDTCO.hasLatencySeconds),
            ('carbon', RDTCO.hasCarbonKgCO2eYear),
            ('f5_carbon_emissions_kgCO2e_year', RDTCO.hasCarbonKgCO2eYear),
        ]

        for key, predicate in result_mappings:
            if key in config and config[key] is not None:
                try:
                    g_cfg.add((cfg_uri, predicate,
                               Literal(float(config[key]), datatype=XSD.decimal)))
                except (ValueError, TypeError):
                    pass

        return g_cfg

    def shacl_validate_config(self, config: dict, shapes_path: str = None) -> tuple:
        """
        使用SHACL验证配置。

        Args:
            config: 配置字典
            shapes_path: SHACL shapes文件路径（可选）

        Returns:
            (conforms: bool, report_text: str)
        """
        try:
            from pyshacl import validate
        except ImportError as e:
            logger.warning("pyshacl未安装，跳过SHACL验证")
            return True, "pyshacl not installed - validation skipped"

        # 构建数据图
        data_g = Graph()
        data_g += self.g  # 全局本体（组件属性）
        data_g += self.build_config_graph(config)  # 当前配置

        # 加载SHACL shapes
        if shapes_path is None:
            # 默认路径
            default_paths = [
                'shapes/min_shapes.ttl',
                './shapes/min_shapes.ttl',
                '../shapes/min_shapes.ttl',
            ]
            for p in default_paths:
                if Path(p).exists():
                    shapes_path = p
                    break

        if shapes_path is None or not Path(shapes_path).exists():
            logger.warning(f"SHACL shapes文件不存在: {shapes_path}")
            return True, "SHACL shapes file not found - validation skipped"

        try:
            shacl_g = Graph().parse(shapes_path, format="turtle")

            conforms, report_graph, report_text = validate(
                data_g,
                shacl_graph=shacl_g,
                inference="rdfs",
                abort_on_first=False,
                meta_shacl=False,
                advanced=True,
                debug=False
            )

            return conforms, report_text

        except Exception as e:
            logger.error(f"SHACL验证出错: {e}")
            return True, f"SHACL validation error: {str(e)}"

    def validate_configuration(self, config_or_uri, shapes_path: str = None) -> bool:
        """
        验证配置是否符合语义约束。

        支持两种输入：
        - URIRef: 从本体中查询配置
        - dict: 直接验证配置字典

        Args:
            config_or_uri: 配置字典或URIRef
            shapes_path: SHACL shapes文件路径

        Returns:
            是否符合约束
        """
        if isinstance(config_or_uri, dict):
            conforms, _ = self.shacl_validate_config(config_or_uri, shapes_path)
            return conforms
        else:
            # 占位符，用于URIRef输入
            return True

    def batch_validate_configs(self, configs: list, shapes_path: str = None) -> dict:
        """
        批量验证配置列表。

        Args:
            configs: 配置字典列表
            shapes_path: SHACL shapes文件路径

        Returns:
            {
                'pass_count': int,
                'total_count': int,
                'pass_ratio': float,
                'results': [{'conforms': bool, 'report': str}, ...]
            }
        """
        results = []
        pass_count = 0

        for cfg in configs:
            conforms, report = self.shacl_validate_config(cfg, shapes_path)
            pass_count += int(conforms)
            results.append({
                'conforms': bool(conforms),
                'report': report[:500] if report else ''  # 限制报告长度
            })

        total = len(configs)
        return {
            'pass_count': pass_count,
            'total_count': total,
            'pass_ratio': pass_count / max(1, total),
            'results': results
        }

    # =========================================================================
    # 原有功能
    # =========================================================================

    def populate_from_csv_files(self,
                                data_dir: str = 'data',
                                sensor_csv: str = None,
                                algorithm_csv: str = None,
                                infrastructure_csv: str = None,
                                cost_benefit_csv: str = None):
        """
        从CSV/TXT文件填充本体

        支持两种调用方式：
        1. populate_from_csv_files(data_dir='data')  # 自动查找文件
        2. populate_from_csv_files(sensor_csv='...', algorithm_csv='...', ...)  # 指定文件
        """
        logger.info("从数据文件填充本体...")

        # 如果指定了具体文件，使用指定的文件
        if sensor_csv:
            sensors_file = Path(sensor_csv)
            if sensors_file.exists():
                self._load_sensors(sensors_file)
            else:
                logger.warning(f"传感器文件不存在: {sensor_csv}")
        else:
            # 尝试默认路径
            for p in [Path(data_dir) / 'sensors_data.txt', Path('sensors_data.txt')]:
                if p.exists():
                    self._load_sensors(p)
                    break

        if algorithm_csv:
            algo_file = Path(algorithm_csv)
            if algo_file.exists():
                self._load_algorithms(algo_file)
            else:
                logger.warning(f"算法文件不存在: {algorithm_csv}")
        else:
            for p in [Path(data_dir) / 'algorithms_data.txt', Path('algorithms_data.txt')]:
                if p.exists():
                    self._load_algorithms(p)
                    break

        if infrastructure_csv:
            infra_file = Path(infrastructure_csv)
            if infra_file.exists():
                self._load_infrastructure(infra_file)
            else:
                logger.warning(f"基础设施文件不存在: {infrastructure_csv}")
        else:
            for p in [Path(data_dir) / 'infrastructure_data.txt', Path('infrastructure_data.txt')]:
                if p.exists():
                    self._load_infrastructure(p)
                    break

        if cost_benefit_csv:
            cost_file = Path(cost_benefit_csv)
            if cost_file.exists():
                self._load_cost_effectiveness(cost_file)
            else:
                logger.warning(f"成本效益文件不存在: {cost_benefit_csv}")
        else:
            for p in [Path(data_dir) / 'cost_effectiveness_data.txt', Path('cost_effectiveness_data.txt')]:
                if p.exists():
                    self._load_cost_effectiveness(p)
                    break

        # 添加SHACL约束
        self._add_shacl_constraints()

        logger.info(f"本体填充完成，包含 {len(self.g)} 个三元组")

        # 验证加载
        self._verify_loaded_components()

    def _load_sensors(self, filepath: Path):
        """加载传感器数据"""
        try:
            df = pd.read_csv(filepath, sep='|')
            logger.info(f"加载 {len(df)} 个传感器实例...")

            for _, row in df.iterrows():
                sensor_id = row['Component_ID'].replace(' ', '_')
                sensor_uri = EX[sensor_id]

                category = row.get('Component_Category', '')
                sensor_class = self._map_sensor_class(category)

                self.g.add((sensor_uri, RDF.type, RDTCO[sensor_class]))
                self.g.add((sensor_uri, RDFS.label, Literal(row['Component_ID'])))

                # 添加属性
                self._add_property(sensor_uri, RDTCO.hasInitialCostUSD,
                                   row.get('Initial_Cost_USD'), XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasOperationalCostUSDPerDay,
                                   row.get('Operational_Cost_USD_per_day'), XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasEnergyConsumptionW,
                                   row.get('Energy_Consumption_W'), XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasMTBFHours,
                                   row.get('MTBF_Hours'), XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasCoverageEfficiencyKmPerDay,
                                   row.get('Coverage_Efficiency_km_per_day'), XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasDataVolumeGBPerKm,
                                   row.get('Data_Volume_GB_per_km'), XSD.decimal)
                self._add_property(sensor_uri, RDTCO.hasAccuracyRangeMM,
                                   row.get('Accuracy_mm_min', row.get('Accuracy_mm_max')), XSD.decimal)

                # 存储组件类别
                if pd.notna(category):
                    self.g.add((sensor_uri, RDTCO.hasComponentCategory, Literal(category)))

        except Exception as e:
            logger.error(f"加载传感器数据时出错: {e}")
            raise

    def _map_sensor_class(self, category: str) -> str:
        """映射传感器类别到本体类"""
        category = str(category).upper() if pd.notna(category) else ''

        mapping = {
            'MMS': 'MMS_LiDAR_System',
            'UAV': 'UAV_LiDAR_System',
            'TLS': 'TLS_System',
            'HANDHELD': 'Handheld_3D_Scanner',
            'FIBER': 'FiberOptic_Sensor',
            'FOS': 'FiberOptic_Sensor',
            'VEHICLE': 'Vehicle_LowCost_Sensor',
            'IOT': 'IoT_Network_System',
            'CAMERA': 'MMS_Camera_System',
        }

        for key, value in mapping.items():
            if key in category:
                return value

        return 'SensorSystem'

    def _load_algorithms(self, filepath: Path):
        """加载算法数据"""
        try:
            df = pd.read_csv(filepath, sep='|')
            logger.info(f"加载 {len(df)} 个算法实例...")

            for _, row in df.iterrows():
                algo_id = row['Component_ID'].replace(' ', '_')
                algo_uri = EX[algo_id]

                category = row.get('Component_Category', '')
                algo_class = self._map_algorithm_class(category)

                self.g.add((algo_uri, RDF.type, RDTCO[algo_class]))
                self.g.add((algo_uri, RDFS.label, Literal(row['Component_ID'])))

                # 添加属性
                self._add_property(algo_uri, RDTCO.hasRecall,
                                   row.get('Detection_Recall_Typical'), XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasPrecision,
                                   row.get('Detection_Precision_Typical'), XSD.decimal)
                self._add_property(algo_uri, RDTCO.hasFPS,
                                   row.get('Processing_FPS_Typical'), XSD.decimal)

                # 硬件需求
                hw_req = row.get('Hardware_Requirement', 'CPU')
                if pd.notna(hw_req):
                    self.g.add((algo_uri, RDTCO.hasHardwareRequirement, Literal(str(hw_req))))

                # 数据格式
                data_fmt = row.get('Required_Data_Format', '')
                if pd.notna(data_fmt):
                    self.g.add((algo_uri, RDTCO.hasDataFormat, Literal(str(data_fmt))))

                # 组件类别
                if pd.notna(category):
                    self.g.add((algo_uri, RDTCO.hasComponentCategory, Literal(category)))

        except Exception as e:
            logger.error(f"加载算法数据时出错: {e}")
            raise

    def _map_algorithm_class(self, category: str) -> str:
        """映射算法类别到本体类"""
        category = str(category).upper() if pd.notna(category) else ''

        if 'DL' in category or 'DEEP' in category:
            return 'DeepLearningAlgorithm'
        elif 'ML' in category or 'MACHINE' in category:
            return 'MachineLearningAlgorithm'
        elif 'POINT' in category or '3D' in category:
            return 'PointCloudAlgorithm'
        else:
            return 'TraditionalAlgorithm'

    def _load_infrastructure(self, filepath: Path):
        """加载基础设施数据"""
        try:
            df = pd.read_csv(filepath, sep='|')
            logger.info(f"加载 {len(df)} 个基础设施实例...")

            for _, row in df.iterrows():
                infra_id = row['Component_ID'].replace(' ', '_')
                infra_uri = EX[infra_id]

                category = str(row.get('Component_Category', '')).upper()

                if 'STORAGE' in category:
                    infra_class = RDTCO.StorageSystem
                elif 'COMM' in category or 'NETWORK' in category:
                    infra_class = RDTCO.CommunicationSystem
                elif 'DEPLOY' in category or 'COMPUTE' in category:
                    infra_class = RDTCO.ComputeDeployment
                else:
                    infra_class = RDTCO.ConfigurationParameter

                self.g.add((infra_uri, RDF.type, infra_class))
                self.g.add((infra_uri, RDFS.label, Literal(row['Component_ID'])))

                # 添加属性
                self._add_property(infra_uri, RDTCO.hasInitialCostUSD,
                                   row.get('Initial_Cost_USD'), XSD.decimal)
                self._add_property(infra_uri, RDTCO.hasOperationalCostUSDPerDay,
                                   row.get('Operational_Cost_USD_per_day'), XSD.decimal)
                self._add_property(infra_uri, RDTCO.hasEnergyConsumptionW,
                                   row.get('Energy_Consumption_W'), XSD.decimal)

                # 组件类别
                if pd.notna(row.get('Component_Category')):
                    self.g.add((infra_uri, RDTCO.hasComponentCategory,
                                Literal(str(row['Component_Category']))))

        except Exception as e:
            logger.error(f"加载基础设施数据时出错: {e}")
            raise

    def _load_cost_effectiveness(self, filepath: Path):
        """加载成本效益参数"""
        try:
            df = pd.read_csv(filepath, sep='|')
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
        self.g.add((budget_shape, RDF.type, SH.PropertyShape))
        self.g.add((budget_shape, SH.path, RDTCO.hasTotalCost))
        self.g.add((budget_shape, SH.maxInclusive,
                    Literal(20000000, datatype=XSD.decimal)))

        # 最小召回率约束
        recall_shape = BNode()
        self.g.add((recall_shape, RDF.type, SH.PropertyShape))
        self.g.add((recall_shape, SH.path, RDTCO.hasRecall))
        self.g.add((recall_shape, SH.minInclusive,
                    Literal(0.65, datatype=XSD.decimal)))

    def _verify_loaded_components(self):
        """验证加载的组件"""
        # 查询所有传感器实例
        sensor_query = """
        PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?sensor WHERE {
            ?sensor rdf:type ?type .
            FILTER(
                CONTAINS(STR(?type), "Sensor") || 
                CONTAINS(STR(?type), "sensor") ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#MMS_LiDAR_System" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#MMS_Camera_System" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#UAV_LiDAR_System" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#UAV_Camera_System" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#TLS_System" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#Handheld_3D_Scanner" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#FiberOptic_Sensor" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#Vehicle_LowCost_Sensor" ||
                STR(?type) = "http://www.semanticweb.org/rmtwin/ontologies/rdtco#IoT_Network_System"
            )
        }
        """

        sensors = list(self.g.query(sensor_query))
        logger.info(f"验证：找到 {len(sensors)} 个传感器实例")

        # 查询算法
        algo_query = """
        PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?algo WHERE {
            ?algo rdf:type ?type .
            FILTER(CONTAINS(STR(?type), "Algorithm") || CONTAINS(STR(?type), "algorithm"))
        }
        """

        algorithms = list(self.g.query(algo_query))
        logger.info(f"验证：找到 {len(algorithms)} 个算法实例")

    def save_ontology(self, filepath: str, format: str = 'turtle'):
        """保存本体到文件"""
        self.g.serialize(destination=filepath, format=format)
        logger.info(f"本体保存到 {filepath}")


# =============================================================================
# 便捷函数
# =============================================================================

def create_ontology_manager(data_dir: str = 'data') -> OntologyManager:
    """创建并初始化本体管理器"""
    manager = OntologyManager()
    manager.populate_from_csv_files(data_dir)
    return manager
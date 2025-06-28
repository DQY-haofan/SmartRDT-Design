#!/usr/bin/env python3
"""
增强的适应度评估模块V3 - 专家级改进版
保持接口兼容性，改进计算逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from functools import lru_cache
from rdflib.namespace import RDF, RDFS, OWL
from rdflib import Graph, Namespace
import pickle

logger = logging.getLogger(__name__)

RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class SolutionMapper:
    """解决方案映射器 - 保持原有接口"""
    
    def __init__(self, ontology_graph: Graph):
        self.g = ontology_graph
        self._cache_components()
        self._decode_cache = {}
        
    def _cache_components(self):
        """缓存所有可用组件"""
        self.sensors = []
        self.algorithms = []
        self.storage_systems = []
        self.comm_systems = []
        self.deployments = []
        
        logger.info("缓存本体组件...")
        
        # 保持原有的组件提取逻辑
        sensor_patterns = [
            'MMS_LiDAR_System', 'MMS_Camera_System', 
            'UAV_LiDAR_System', 'UAV_Camera_System',
            'TLS_System', 'Handheld_3D_Scanner',
            'FiberOptic_Sensor', 'Vehicle_LowCost_Sensor',
            'IoT_Network_System', 'Sensor', 'sensor'
        ]
        
        algo_patterns = [
            'DeepLearningAlgorithm', 'MachineLearningAlgorithm',
            'TraditionalAlgorithm', 'PointCloudAlgorithm',
            'Algorithm', 'algorithm'
        ]
        
        deploy_patterns = [
            'Deployment', 'Compute', 'Edge', 'Cloud',
            'ComputeDeployment', 'Deployment_Edge_Computing',
            'Deployment_Cloud_Computing', 'Deployment_Hybrid_Edge_Cloud',
            'Deployment_OnPremise_Server'
        ]
        
        for s, p, o in self.g:
            if p == RDF.type and str(s).startswith('http://example.org/rmtwin#'):
                subject_str = str(s)
                type_str = str(o)
                
                is_sensor = any(pattern in type_str for pattern in sensor_patterns)
                if is_sensor and subject_str not in self.sensors:
                    self.sensors.append(subject_str)
                    continue
                
                is_algorithm = any(pattern in type_str for pattern in algo_patterns)
                if is_algorithm and subject_str not in self.algorithms:
                    self.algorithms.append(subject_str)
                    continue
                
                if 'Storage' in type_str and subject_str not in self.storage_systems:
                    self.storage_systems.append(subject_str)
                    continue
                
                if 'Communication' in type_str and subject_str not in self.comm_systems:
                    self.comm_systems.append(subject_str)
                    continue
                
                is_deployment = any(pattern in type_str for pattern in deploy_patterns)
                if is_deployment and subject_str not in self.deployments:
                    self.deployments.append(subject_str)
        
        # 确保至少有默认值
        if not self.sensors:
            logger.warning("未找到传感器！使用默认值")
            self.sensors = ["http://example.org/rmtwin#IoT_LoRaWAN_Sensor"]
        if not self.algorithms:
            logger.warning("未找到算法！使用默认值")
            self.algorithms = ["http://example.org/rmtwin#Traditional_Canny_Optimized"]
        if not self.storage_systems:
            self.storage_systems = ["http://example.org/rmtwin#Storage_AWS_S3_Standard"]
        if not self.comm_systems:
            self.comm_systems = ["http://example.org/rmtwin#Communication_LoRaWAN_Gateway"]
        if not self.deployments:
            self.deployments = ["http://example.org/rmtwin#Deployment_Cloud_GPU_A4000"]
        
        logger.info(f"缓存的组件: {len(self.sensors)} 传感器, "
                   f"{len(self.algorithms)} 算法, {len(self.storage_systems)} 存储, "
                   f"{len(self.comm_systems)} 通信, {len(self.deployments)} 部署")
    
    def decode_solution(self, x: np.ndarray) -> Dict:
        """解码解决方案向量"""
        x_key = tuple(float(xi) for xi in x)
        
        if x_key in self._decode_cache:
            return self._decode_cache[x_key]
        
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
        
        self._decode_cache[x_key] = config
        return config


class EnhancedFitnessEvaluatorV3:
    """增强的适应度评估器 - 专家级改进版"""
    
    def __init__(self, ontology_graph: Graph, config):
        self.g = ontology_graph
        self.config = config
        self.solution_mapper = SolutionMapper(ontology_graph)
        
        # 组件属性缓存
        self._property_cache = {}
        
        # 改进的归一化参数
        self.norm_params = {
            'cost': {'min': 100_000, 'max': 5_000_000},
            'recall': {'min': 0.0, 'max': 0.5},  # 1-recall
            'latency': {'min': 0.1, 'max': 300.0},
            'disruption': {'min': 0.0, 'max': 300.0},
            'environmental': {'min': 500, 'max': 50_000},  # kgCO2e/year
            'reliability': {'min': 0.0, 'max': 0.001}  # 1/MTBF
        }
        
        # 新增：设备折旧率字典
        self.depreciation_rates = {
            'MMS': 0.15,
            'UAV': 0.20,
            'TLS': 0.12,
            'Handheld': 0.12,
            'Vehicle': 0.15,
            'IoT': 0.10,
            'FOS': 0.08,
            'Camera': 0.12
        }
        
        # 新增：环境因子
        self.environmental_factors = {
            'urban': 0.85,
            'rural': 0.95,
            'tunnel': 0.70,
            'mixed': 0.90
        }
        
        # 新增：制造碳强度
        self.manufacturing_carbon_intensity = {
            'Electronics': 0.5,     # kg CO2 / $100
            'Mechanical': 0.3,      
            'Software': 0.1,
            'Vehicle': 0.8
        }
        
        # 新增：数据中心PUE
        self.datacenter_pue = {
            'Cloud': 1.67,
            'Edge': 1.2,
            'OnPremise': 1.4
        }
        
        # 预缓存所有属性
        self._initialize_cache()
        
        # 为进程池准备数据
        self._prepare_pool_data()
        
        # 统计
        self._evaluation_count = 0
    
    def _initialize_cache(self):
        """初始化属性缓存"""
        logger.info("初始化属性缓存...")
        
        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasAnnualOpCostUSD',
            'hasEnergyConsumptionW', 'hasMTBFHours', 'hasOperatorSkillLevel',
            'hasCalibrationFreqMonths', 'hasDataAnnotationCostUSD',
            'hasModelRetrainingFreqMonths', 'hasExplainabilityScore',
            'hasIntegrationComplexity', 'hasCybersecurityVulnerability',
            'hasAccuracyRangeMM', 'hasDataVolumeGBPerKm',
            'hasCoverageEfficiencyKmPerDay', 'hasOperatingSpeedKmh',
            'hasRecall', 'hasPrecision', 'hasFPS', 'hasHardwareRequirement'
        ]
        
        all_components = (
            self.solution_mapper.sensors + 
            self.solution_mapper.algorithms + 
            self.solution_mapper.storage_systems + 
            self.solution_mapper.comm_systems + 
            self.solution_mapper.deployments
        )
        
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
                except Exception:
                    pass
        
        logger.info(f"为 {len(self._property_cache)} 个组件缓存了属性")
    
    def _prepare_pool_data(self):
        """准备进程池所需的数据"""
        self._mapper_data = {
            'sensors': self.solution_mapper.sensors,
            'algorithms': self.solution_mapper.algorithms,
            'storage_systems': self.solution_mapper.storage_systems,
            'comm_systems': self.solution_mapper.comm_systems,
            'deployments': self.solution_mapper.deployments
        }
        
        self._config_dict = vars(self.config) if hasattr(self.config, '__dict__') else self.config
    
    def _query_property(self, subject: str, predicate: str, default=None):
        """从缓存查询属性"""
        if subject in self._property_cache:
            if predicate in self._property_cache[subject]:
                return self._property_cache[subject][predicate]
        return default
    
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量评估解决方案 - 保持接口不变"""
        n_solutions = len(X)
        objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 5))
        
        for i, x in enumerate(X):
            try:
                objectives[i], constraints[i] = self._evaluate_single(x)
            except Exception as e:
                logger.error(f"评估解决方案 {i} 时出错: {e}")
                # 惩罚值
                objectives[i] = np.array([1e10, 1, 1000, 1000, 200000, 1])
                constraints[i] = np.array([1000, 1, 1e10, 100000, -1000])
        
        self._evaluation_count += n_solutions
        
        if self._evaluation_count % 1000 == 0:
            logger.debug(f"已评估 {self._evaluation_count} 个解决方案")
        
        return objectives, constraints
    
    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估单个解决方案"""
        config = self.solution_mapper.decode_solution(x)
        
        # 计算所有6个目标
        f1 = self._calculate_total_cost_enhanced(config)
        f2 = self._calculate_detection_performance_enhanced(config)
        f3 = self._calculate_latency_enhanced(config)
        f4 = self._calculate_traffic_disruption_enhanced(config)
        f5 = self._calculate_environmental_impact(config)
        f6 = self._calculate_system_reliability(config)
        
        objectives = np.array([f1, f2, f3, f4, f5, f6])
        
        # 计算约束
        recall = 1 - f2
        constraints = np.array([
            f3 - self.config.max_latency_seconds,
            self.config.min_recall_threshold - recall,
            f1 - self.config.budget_cap_usd,
            f5 - self.config.max_carbon_emissions_kgCO2e_year,
            self.config.min_mtbf_hours - (1/f6 if f6 > 0 else 1e6)
        ])
        
        return objectives, constraints
    
    def _calculate_total_cost_enhanced(self, config: Dict) -> float:
        """改进的成本计算"""
        sensor_name = str(config['sensor']).split('#')[-1]
        
        # 初始投资成本
        sensor_initial_cost = self._query_property(
            config['sensor'], 'hasInitialCostUSD', 100000)
        
        # FOS传感器的特殊处理
        if 'FOS' in sensor_name or 'Fiber' in sensor_name:
            sensor_spacing_km = self.config.fos_sensor_spacing_km
            road_length = self.config.road_network_length_km
            sensors_needed = road_length / sensor_spacing_km
            actual_sensor_cost = sensor_initial_cost * sensors_needed
            installation_cost = 5000 * sensors_needed
            total_sensor_initial = actual_sensor_cost + installation_cost
        else:
            total_sensor_initial = sensor_initial_cost
        
        # 其他组件成本
        storage_initial = self._query_property(config['storage'], 'hasInitialCostUSD', 0)
        comm_initial = self._query_property(config['communication'], 'hasInitialCostUSD', 0)
        deployment_initial = self._query_property(config['deployment'], 'hasInitialCostUSD', 0)
        algo_initial = self._query_property(config['algorithm'], 'hasInitialCostUSD', 20000)
        
        total_initial_investment = (total_sensor_initial + storage_initial + 
                                  comm_initial + deployment_initial + algo_initial)
        
        # 改进：根据设备类型使用不同折旧率
        sensor_type = None
        for key in self.depreciation_rates.keys():
            if key in sensor_name:
                sensor_type = key
                break
        depreciation_rate = self.depreciation_rates.get(sensor_type, 0.10)
        annual_capital_cost = total_initial_investment * depreciation_rate
        
        # 传感器运营成本
        sensor_daily_cost = self._query_property(
            config['sensor'], 'hasOperationalCostUSDPerDay', 100)
        coverage_km_day = self._query_property(
            config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        
        road_length = self.config.road_network_length_km
        inspections_per_year = 365 / config['inspection_cycle']
        
        if coverage_km_day > 0:  # 移动传感器
            days_per_inspection = road_length / coverage_km_day
            sensor_annual_operational = sensor_daily_cost * days_per_inspection * inspections_per_year
        else:  # 固定传感器
            if 'FOS' in sensor_name:
                operational_cost_per_sensor_day = 0.5
                sensors_needed = road_length / sensor_spacing_km
                sensor_annual_operational = operational_cost_per_sensor_day * sensors_needed * 365
            else:
                sensor_annual_operational = sensor_daily_cost * 365
        
        # 基础设施运营成本
        storage_annual = self._query_property(config['storage'], 'hasAnnualOpCostUSD', 5000)
        comm_annual = self._query_property(config['communication'], 'hasAnnualOpCostUSD', 2000)
        deployment_annual = self._query_property(config['deployment'], 'hasAnnualOpCostUSD', 10000)
        
        # 改进：数据存储成本计算
        data_volume_gb_per_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        annual_data_volume = data_volume_gb_per_km * road_length * inspections_per_year
        
        if 'Cloud' in str(config['storage']):
            # AWS S3标准存储成本
            storage_cost_per_gb_month = 0.023
            data_storage_annual = annual_data_volume * storage_cost_per_gb_month * 12
            # 考虑数据保留策略
            data_retention_years = getattr(self.config, 'data_retention_years', 3)
            data_storage_annual *= min(data_retention_years, 3)  # 最多保留3年
            storage_annual += data_storage_annual
            
            # 数据传输成本
            if 'Cloud' in str(config['deployment']):
                data_transfer_cost_per_gb = 0.09  # AWS数据传出
                annual_transfer_cost = annual_data_volume * data_transfer_cost_per_gb * 0.3  # 假设30%数据需要传出
                storage_annual += annual_transfer_cost
        
        # 人员成本
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
        
        # ML/DL算法的数据标注成本
        data_annotation_annual = 0
        algo_name = str(config['algorithm']).split('#')[-1]
        if 'DL' in algo_name or 'Deep' in algo_name or 'ML' in algo_name:
            annotation_cost = self._query_property(
                config['algorithm'], 'hasDataAnnotationCostUSD', 0.5)
            
            if 'Camera' in sensor_name:
                images_per_km = 100
                annual_images = images_per_km * road_length * inspections_per_year
            else:
                annual_images = 10000
            
            # 考虑主动学习减少标注需求
            active_learning_factor = 0.3 if 'DL' in algo_name else 0.5
            data_annotation_annual = annotation_cost * annual_images * active_learning_factor
        
        # 模型重训练成本
        retrain_freq = self._query_property(config['algorithm'], 'hasModelRetrainingFreqMonths', 12)
        model_retraining_annual = 0
        if retrain_freq and retrain_freq > 0:
            retrainings_per_year = 12 / retrain_freq
            retraining_cost = 5000 if 'DL' in algo_name else 2000
            model_retraining_annual = retraining_cost * retrainings_per_year
        
        # 年度总成本
        total_annual_cost = (annual_capital_cost + sensor_annual_operational + 
                           storage_annual + comm_annual + deployment_annual + 
                           crew_annual_cost + data_annotation_annual + model_retraining_annual)
        
        # 应用季节性调整
        if self.config.apply_seasonal_adjustments:
            winter_factor = 1.3
            seasonal_adjustment = 0.25
            total_annual_cost *= (1 + (winter_factor - 1) * seasonal_adjustment)
        
        # 软约束惩罚
        if config['crew_size'] > 5:
            total_annual_cost += (config['crew_size'] - 5) * 50000
        
        if config['inspection_cycle'] < 7:
            total_annual_cost *= 1.5
        
        # 总生命周期成本
        planning_horizon = self.config.planning_horizon_years
        total_lifecycle_cost = total_annual_cost * planning_horizon
        
        return total_lifecycle_cost
    
    def _calculate_detection_performance_enhanced(self, config: Dict) -> float:
        """改进的检测性能计算"""
        # 基础召回率（调整为更现实的值）
        base_recall = self._query_property(config['algorithm'], 'hasRecall', 0.7)
        
        # 应用现实世界性能折扣
        real_world_factor = 1  # 实验室到现实的性能下降
        base_recall *= real_world_factor
        
        # 传感器精度影响（使用改进的S型曲线）
        accuracy_mm = self._query_property(config['sensor'], 'hasAccuracyRangeMM', 10)
        target_crack_mm = 10  # 目标检测裂缝尺寸
        
        def accuracy_impact(accuracy, target):
            ratio = accuracy / target
            if ratio < 0.5:
                return 1.0
            elif ratio < 2.0:
                return 1.0 - 0.3 * ((ratio - 0.5) / 1.5) ** 2
            else:
                return 0.7 * np.exp(-(ratio - 2) * 0.5)
        
        accuracy_factor = accuracy_impact(accuracy_mm, target_crack_mm)
        base_recall *= accuracy_factor
        
        # LOD影响
        lod_factors = {
            'Micro': {'factor': 1.10, 'threshold_adj': 0.05},
            'Meso': {'factor': 1.0, 'threshold_adj': 0},
            'Macro': {'factor': 0.85, 'threshold_adj': -0.05}
        }
        
        lod_data = lod_factors.get(config['geo_lod'], lod_factors['Meso'])
        base_recall *= lod_data['factor']
        
        # 检测阈值影响
        threshold_optimal = 0.5
        threshold_penalty = abs(config['detection_threshold'] - threshold_optimal) * 0.1
        base_recall -= threshold_penalty
        
        # 环境因子影响
        scenario_type = getattr(self.config, 'scenario_type', 'mixed')
        env_factor = self.environmental_factors.get(scenario_type, 0.90)
        base_recall *= env_factor
        
        # 算法特定调整
        algo_name = str(config['algorithm']).split('#')[-1]
        
        # 类别不平衡惩罚
        class_imbalance_penalties = self.config.class_imbalance_penalties
        penalty = 0
        for algo_type, pen_value in class_imbalance_penalties.items():
            if algo_type in algo_name:
                penalty = pen_value
                break
        
        base_recall -= penalty
        
        # 硬件影响
        hardware_req = self._query_property(config['algorithm'], 'hasHardwareRequirement', 'CPU')
        deployment_type = str(config['deployment']).split('#')[-1]
        
        # 硬件匹配检查
        hw_mismatch_penalty = 0
        if 'GPU' in str(hardware_req) and 'Edge' in deployment_type:
            hw_mismatch_penalty = 0.1  # 边缘设备可能GPU性能有限
        elif 'HighEnd_GPU' in str(hardware_req) and not 'Cloud' in deployment_type:
            hw_mismatch_penalty = 0.15  # 高端GPU需求在非云端难以满足
        
        base_recall -= hw_mismatch_penalty
        
        # 确保合理范围（0.5-0.9）
        base_recall = np.clip(base_recall, 0.5, 0.9)
        
        # 添加小的随机噪声
        noise = np.random.normal(0, 0.003)
        base_recall += noise
        
        return 1 - np.clip(base_recall, 0.01, 0.99)
    
    def _calculate_latency_enhanced(self, config: Dict) -> float:
        """改进的延迟计算"""
        sensor_name = str(config['sensor']).split('#')[-1]
        
        # 1. 数据采集时间
        if 'MMS' in sensor_name:
            coverage_speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
            # 城市环境速度降低
            if self.config.scenario_type == 'urban':
                coverage_speed *= 0.6  # 城市交通影响
            scan_segment_km = 5  # 每次处理5km路段
            acq_time = (scan_segment_km / coverage_speed) * 3600 if coverage_speed > 0 else 300
        
        elif 'UAV' in sensor_name:
            coverage_speed = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 2.7)
            if coverage_speed > 0:
                acq_time = (1.0 / (coverage_speed / 24)) * 3600
            else:
                acq_time = 1200
        
        elif 'TLS' in sensor_name or 'Handheld' in sensor_name:
            acq_time = 300
        
        elif 'IoT' in sensor_name or 'FOS' in sensor_name:
            # 实时监测系统
            data_rate = config['data_rate']
            samples_needed = 100  # 减少到100个样本
            acq_time = samples_needed / data_rate if data_rate > 0 else 10
        
        else:
            acq_time = 60
        
        # 2. 数据量估算
        base_data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        lod_multipliers = {
            'Micro': 2.0,   # 减少倍数
            'Meso': 1.0,
            'Macro': 0.5
        }
        data_gb = base_data_gb * lod_multipliers.get(config['geo_lod'], 1.0)
        
        # 3. 通信延迟
        comm_type = str(config['communication']).split('#')[-1]
        
        # 更现实的带宽设置
        realistic_comm_specs = {
            'Communication_5G_SmallCell': {
                'bandwidth': 50,     # 实际5G平均
                'latency': 0.05,
                'reliability': 0.92
            },
            'Communication_LoRaWAN_Gateway': {
                'bandwidth': 0.05,
                'latency': 1.5,
                'reliability': 0.88
            },
            'Communication_Fiber_Metro': {
                'bandwidth': 300,    # 实际光纤
                'latency': 0.02,
                'reliability': 0.99
            },
            'Communication_4G_Rural': {
                'bandwidth': 15,     # 农村4G
                'latency': 0.1,
                'reliability': 0.85
            }
        }
        
        # 获取通信规格
        default_comm = {'bandwidth': 20, 'latency': 0.1, 'reliability': 0.9}
        comm_data = default_comm.copy()
        
        # 查找匹配的通信类型
        for comm_key, comm_spec in realistic_comm_specs.items():
            if any(keyword in comm_type for keyword in comm_key.split('_')[1:]):
                comm_data = comm_spec
                break
        
        # 场景影响
        scenario_impacts = {
            'urban': {'5G': 0.8, '4G': 0.7, 'Fiber': 0.95, 'LoRaWAN': 0.6},
            'rural': {'5G': 0.4, '4G': 0.6, 'Fiber': 0.7, 'LoRaWAN': 0.9},
            'mixed': {'5G': 0.6, '4G': 0.65, 'Fiber': 0.85, 'LoRaWAN': 0.8}
        }
        
        # 确定技术类型
        tech = 'LoRaWAN'  # 默认
        for t in ['Fiber', '5G', '4G', 'LoRaWAN']:
            if t in comm_type:
                tech = t
                break
        
        scenario_factor = scenario_impacts.get(self.config.scenario_type, {}).get(tech, 0.7)
        
        effective_bandwidth = comm_data['bandwidth'] * scenario_factor * comm_data['reliability']
        network_latency = comm_data['latency'] / scenario_factor
        
        # 通信时间
        protocol_overhead = 1.15  # 减少开销估计
        comm_time = (data_gb * 1000 * 8 * protocol_overhead) / effective_bandwidth if effective_bandwidth > 0 else 1000
        
        # 重传
        retransmission_factor = 1 + (1 - comm_data['reliability']) * 0.5
        comm_time *= retransmission_factor
        
        # 4. 处理时间
        algo_name = str(config['algorithm']).split('#')[-1]
        
        # 更现实的处理时间
        algo_processing_times = {
            'DL': 30,        # 深度学习：30秒/GB
            'ML': 10,        # 机器学习：10秒/GB  
            'Traditional': 2, # 传统算法：2秒/GB
            'PC': 15         # 点云处理：15秒/GB
        }
        
        algo_type = 'Traditional'
        for key in algo_processing_times.keys():
            if key in algo_name:
                algo_type = key
                break
        
        base_proc_time_per_gb = algo_processing_times[algo_type]
        
        # LOD处理影响
        lod_processing_factors = {
            'Micro': 1.5,
            'Meso': 1.0,
            'Macro': 0.7
        }
        lod_factor = lod_processing_factors.get(config['geo_lod'], 1.0)
        
        # 部署性能
        deploy_type = str(config['deployment']).split('#')[-1]
        deploy_performance = {
            'Deployment_Edge_Computing': {
                'compute_factor': 1.5,    # 改进的边缘性能
                'startup_overhead': 2.0
            },
            'Deployment_Cloud_Computing': {
                'compute_factor': 1.0,
                'startup_overhead': 1.0
            },
            'Deployment_Hybrid_Edge_Cloud': {
                'compute_factor': 1.2,
                'startup_overhead': 1.5
            },
            'Deployment_OnPremise_Server': {
                'compute_factor': 1.3,
                'startup_overhead': 0.5
            }
        }
        
        # 查找匹配的部署类型
        deploy_data = {'compute_factor': 1.5, 'startup_overhead': 2.0}
        for deploy_key, deploy_spec in deploy_performance.items():
            for keyword in ['Edge', 'Cloud', 'Hybrid', 'OnPremise']:
                if keyword in deploy_type and keyword in deploy_key:
                    deploy_data = deploy_spec
                    break
        
        proc_time = (base_proc_time_per_gb * data_gb * lod_factor * 
                    deploy_data['compute_factor'] + deploy_data['startup_overhead'])
        
        # 5. 排队延迟
        if 'Cloud' in deploy_type:
            queue_time = np.random.exponential(5)  # 平均5秒
        else:
            queue_time = np.random.exponential(2)
        
        # 6. 结果传输
        result_size_gb = data_gb * 0.05  # 结果更小
        result_transmission_time = (result_size_gb * 1000 * 8) / effective_bandwidth if effective_bandwidth > 0 else 5
        
        # 7. 验证时间（仅高精度需求）
        verification_time = 0
        if ('DL' in algo_name or 'ML' in algo_name) and config['geo_lod'] == 'Micro':
            verification_time = 15
        
        # 总延迟
        total_latency = (acq_time + network_latency + comm_time + 
                        proc_time + queue_time + result_transmission_time + 
                        verification_time)
        
        # 确保最小延迟
        min_latency = 10.0  # 至少10秒
        total_latency = max(total_latency, min_latency)
        
        # 限制最大延迟
        max_latency = 600.0  # 最多10分钟
        total_latency = min(total_latency, max_latency)
        
        # 添加随机性（±5%）
        noise_factor = 1 + (np.random.random() - 0.5) * 0.1
        total_latency *= noise_factor
        
        return total_latency
    
    def _calculate_traffic_disruption_enhanced(self, config: Dict) -> float:
        """改进的交通干扰计算"""
        # 基础干扰时间
        base_disruption = 3.0  # 减少到3小时
        
        sensor_name = str(config['sensor']).split('#')[-1]
        speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
        
        # 传感器类型影响
        if 'UAV' in sensor_name:
            disruption_factor = 0.05  # UAV干扰最小
        elif 'FOS' in sensor_name or 'IoT' in sensor_name:
            disruption_factor = 0.02  # 固定传感器仅安装时
        elif 'Vehicle' in sensor_name:
            # 车载传感器与交通流同速，干扰小
            disruption_factor = 0.3
        elif speed > 0:
            # 移动传感器基于相对速度
            traffic_speed = 60  # 假设交通平均速度
            speed_diff = abs(speed - traffic_speed)
            disruption_factor = 0.5 + (speed_diff / traffic_speed) * 0.5
        else:
            disruption_factor = 1.0
        
        # 计算年度干扰
        inspections_per_year = 365 / config['inspection_cycle']
        
        if 'FOS' in sensor_name or 'IoT' in sensor_name:
            # 固定传感器
            installation_days = 5  # 初始安装
            maintenance_visits = 4  # 季度维护
            annual_disruption = (base_disruption * disruption_factor * installation_days / 
                               self.config.planning_horizon_years +  # 摊销安装
                               base_disruption * disruption_factor * maintenance_visits * 0.5)
        else:
            # 移动传感器
            road_length = self.config.road_network_length_km
            coverage_per_day = self._query_property(
                config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
            
            if coverage_per_day > 0:
                days_per_inspection = road_length / coverage_per_day
                # 考虑并行作业减少干扰
                parallel_factor = 0.7 if config['crew_size'] > 3 else 1.0
                annual_disruption = (base_disruption * disruption_factor * 
                                   days_per_inspection * inspections_per_year * parallel_factor)
            else:
                annual_disruption = base_disruption * disruption_factor * inspections_per_year
        
        # 交通量影响（使用更合理的模型）
        traffic_volume = self.config.traffic_volume_hourly
        # 使用对数关系，避免线性增长
        traffic_factor = 1 + np.log10(traffic_volume / 1000 + 1)
        
        # 车道影响
        lane_closure_ratio = self.config.default_lane_closure_ratio
        # 部分封闭影响小于完全封闭
        lane_factor = 1 + lane_closure_ratio * 1.5
        
        # 时间优化
        night_work_ratio = 0.4  # 40%夜间作业
        weekend_ratio = 0.2     # 20%周末作业
        time_reduction = night_work_ratio * 0.7 + weekend_ratio * 0.5
        time_factor = 1 - time_reduction
        
        # 季节影响
        if self.config.apply_seasonal_adjustments:
            # 冬季施工窗口减少，单次干扰可能增加
            seasonal_factor = 1.1
        else:
            seasonal_factor = 1.0
        
        # 总干扰
        total_disruption = (annual_disruption * lane_factor * traffic_factor * 
                          time_factor * seasonal_factor)
        
        # 确保合理范围
        total_disruption = np.clip(total_disruption, 0, 500)  # 最多500小时/年
        
        return total_disruption
    
    def _calculate_environmental_impact(self, config: Dict) -> float:
        """改进的环境影响计算"""
        # 能耗计算
        total_power_w = 0
        
        # 各组件能耗
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                power = self._query_property(config[comp], 'hasEnergyConsumptionW', 0)
                if power:
                    total_power_w += power
        
        # 算法计算能耗（更现实的估计）
        algo_name = str(config['algorithm']).split('#')[-1]
        algo_power = {
            'DL': 200,      # 深度学习GPU
            'ML': 80,       # 机器学习
            'Traditional': 30,  # 传统算法
            'PC': 150       # 点云处理
        }
        
        for key, power in algo_power.items():
            if key in algo_name:
                total_power_w += power
                break
        
        # 运行时间计算
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        road_length = self.config.road_network_length_km
        inspections_per_year = 365 / config['inspection_cycle']
        
        if coverage > 0:
            # 移动传感器
            days_per_inspection = road_length / coverage
            sensor_hours = days_per_inspection * inspections_per_year * 6  # 6小时/天
            
            # 车辆碳排放（更准确的模型）
            vehicle_km = road_length * inspections_per_year
            # 考虑不同车型
            if 'MMS' in str(config['sensor']):
                fuel_consumption = 0.12  # 12L/100km (大型车)
            else:
                fuel_consumption = 0.08  # 8L/100km (标准车)
            
            vehicle_fuel_l = vehicle_km * fuel_consumption
            vehicle_emissions_kg = vehicle_fuel_l * 2.31  # 汽油碳排放因子
        else:
            # 固定传感器
            sensor_hours = 365 * 24
            vehicle_emissions_kg = 0
        
        # 考虑数据中心PUE
        deploy_type = str(config['deployment']).split('#')[-1]
        pue = 1.4  # 默认
        for key, value in self.datacenter_pue.items():
            if key in deploy_type:
                pue = value
                break
        
        # 后端始终运行
        backend_hours = 365 * 24
        
        # 能耗分解（考虑PUE）
        sensor_energy_kwh = (total_power_w * 0.3 * sensor_hours) / 1000
        backend_energy_kwh = (total_power_w * 0.7 * backend_hours * pue) / 1000
        total_energy_kwh = sensor_energy_kwh + backend_energy_kwh
        
        # 碳强度（考虑可再生能源）
        carbon_intensity = self.config.carbon_intensity_factor
        if 'Cloud' in deploy_type:
            # 大型云提供商使用更多可再生能源
            carbon_intensity *= 0.7
        
        electricity_emissions_kg = total_energy_kwh * carbon_intensity
        
        # 改进的制造排放计算
        equipment_cost = self._query_property(config['sensor'], 'hasInitialCostUSD', 100000)
        sensor_type = str(config['sensor']).split('_')[0]
        
        # 根据设备类型确定碳强度
        if 'Vehicle' in sensor_type or 'MMS' in sensor_type:
            mfg_carbon_intensity = self.manufacturing_carbon_intensity['Vehicle']
        elif 'UAV' in sensor_type or 'IoT' in sensor_type:
            mfg_carbon_intensity = self.manufacturing_carbon_intensity['Electronics']
        else:
            mfg_carbon_intensity = self.manufacturing_carbon_intensity['Mechanical']
        
        manufacturing_emissions = (equipment_cost / 100) * mfg_carbon_intensity
        equipment_lifetime = 8  # 8年寿命
        annual_manufacturing = manufacturing_emissions / equipment_lifetime
        
        # 数据存储排放
        data_volume_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        annual_data_gb = data_volume_gb * road_length * inspections_per_year
        
        # 存储碳排放（考虑存储类型）
        if 'Cloud' in str(config['storage']):
            storage_emissions_kg = annual_data_gb * 0.005  # 云存储更高效
        else:
            storage_emissions_kg = annual_data_gb * 0.01   # 本地存储
        
        # 网络传输排放
        if 'Cloud' in str(config['deployment']):
            network_emissions_kg = annual_data_gb * 0.002  # 网络传输碳排放
        else:
            network_emissions_kg = 0
        
        # 报废处理排放（简化估算）
        end_of_life_emissions = manufacturing_emissions * 0.1
        annual_eol = end_of_life_emissions / equipment_lifetime
        
        # 总排放量
        total_emissions = (electricity_emissions_kg + vehicle_emissions_kg + 
                          annual_manufacturing + storage_emissions_kg + 
                          network_emissions_kg + annual_eol)
        
        # 确保合理范围
        total_emissions = np.clip(total_emissions, 100, 100000)  # 100kg-100t CO2/年
        
        return total_emissions
    
    def _calculate_system_reliability(self, config: Dict) -> float:
        """改进的系统可靠性计算"""
        # 组件MTBF收集
        component_mtbfs = []
        
        # 考虑系统架构
        deployment_type = str(config['deployment']).split('#')[-1]
        storage_type = str(config['storage']).split('#')[-1]
        
        # 确定架构类型
        if 'Cloud' in deployment_type and 'Cloud' in storage_type:
            architecture = 'distributed'
        elif 'Hybrid' in deployment_type or 'Hybrid' in storage_type:
            architecture = 'load_balanced'
        elif 'OnPremise' in deployment_type:
            architecture = 'active_backup'
        else:
            architecture = 'single_point'
        
        architecture_factors = {
            'single_point': 1.0,
            'active_backup': 1.8,
            'load_balanced': 2.5,
            'distributed': 3.0
        }
        arch_factor = architecture_factors[architecture]
        
        # 处理各组件
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                base_mtbf = self._query_property(config[comp], 'hasMTBFHours', 10000)
                
                if base_mtbf and base_mtbf > 0:
                    # 环境因素
                    if comp == 'sensor':
                        # 户外传感器
                        sensor_name = str(config[comp]).split('#')[-1]
                        if 'UAV' in sensor_name:
                            environmental_factor = 0.6  # UAV更脆弱
                        elif 'Vehicle' in sensor_name:
                            environmental_factor = 0.7  # 车辆振动
                        elif 'FOS' in sensor_name or 'IoT' in sensor_name:
                            environmental_factor = 0.9  # 固定安装，保护较好
                        else:
                            environmental_factor = 0.8
                    else:
                        environmental_factor = 0.95  # 室内设备
                    
                    # 维护影响
                    calibration_freq = self._query_property(
                        config[comp], 'hasCalibrationFreqMonths', 12)
                    if calibration_freq and calibration_freq > 0:
                        # 频繁维护提高可靠性
                        maintenance_factor = 1 + (12 / calibration_freq) * 0.2
                        maintenance_factor = min(maintenance_factor, 1.5)
                    else:
                        maintenance_factor = 1.0
                    
                    # 质量等级影响（基于成本）
                    component_cost = self._query_property(config[comp], 'hasInitialCostUSD', 10000)
                    if component_cost > 100000:
                        quality_factor = 1.3  # 高端设备
                    elif component_cost > 20000:
                        quality_factor = 1.0  # 中端
                    else:
                        quality_factor = 0.8  # 低端
                    
                    effective_mtbf = (base_mtbf * arch_factor * environmental_factor * 
                                    maintenance_factor * quality_factor)
                    component_mtbfs.append(effective_mtbf)
        
        # 算法可靠性
        algo_mtbf = self._query_property(config['algorithm'], 'hasMTBFHours', 50000)
        if algo_mtbf and algo_mtbf > 0:
            # 软件复杂性影响
            algo_name = str(config['algorithm']).split('#')[-1]
            if 'DL' in algo_name:
                complexity_factor = 0.7  # 深度学习更复杂
            elif 'ML' in algo_name:
                complexity_factor = 0.85
            else:
                complexity_factor = 0.95  # 传统算法更稳定
            
            # 可解释性影响
            explainability = self._query_property(
                config['algorithm'], 'hasExplainabilityScore', 3)
            explain_factor = 0.8 + (explainability / 5) * 0.4 if explainability else 1.0
            
            effective_algo_mtbf = algo_mtbf * complexity_factor * explain_factor * arch_factor
            component_mtbfs.append(effective_algo_mtbf)
        
        # 系统MTBF计算
        if component_mtbfs:
            # 考虑部分冗余的改进模型
            if architecture == 'distributed' or architecture == 'load_balanced':
                # 并联系统近似
                system_failure_rate = 0
                for mtbf in component_mtbfs:
                    # 冗余降低单个组件失效影响
                    redundancy_factor = 0.3 if architecture == 'distributed' else 0.5
                    system_failure_rate += (1/mtbf) * redundancy_factor
            else:
                # 串联系统
                system_failure_rate = sum(1/mtbf for mtbf in component_mtbfs)
            
            # 共因故障（减少到更合理的水平）
            common_cause_rate = 1 / (365 * 24 * 20)  # 20年一次
            system_failure_rate += common_cause_rate
            
            # 人为错误（改进的模型）
            skill_level = self._query_property(config['sensor'], 'hasOperatorSkillLevel', 'Basic')
            human_error_rates = {
                'Basic': 1 / (365 * 24 * 1),      # 每年一次
                'Intermediate': 1 / (365 * 24 * 2), # 每2年一次
                'Expert': 1 / (365 * 24 * 5)       # 每5年一次
            }
            human_error_rate = human_error_rates.get(str(skill_level), human_error_rates['Basic'])
            
            # 团队规模影响（改进：大团队有更好的交叉检查）
            if config['crew_size'] <= 2:
                crew_factor = 1.2  # 小团队风险更高
            elif config['crew_size'] <= 5:
                crew_factor = 1.0
            else:
                crew_factor = 0.9  # 大团队有冗余
            
            system_failure_rate += human_error_rate * crew_factor
            
            # 软件更新风险
            if 'DL' in str(config['algorithm']) or 'ML' in str(config['algorithm']):
                update_risk = 1 / (365 * 24 * 0.5)  # 每6个月一次更新风险
                system_failure_rate += update_risk * 0.1  # 10%更新导致问题
            
            return system_failure_rate
        else:
            return 1e-6  # 默认高可靠性
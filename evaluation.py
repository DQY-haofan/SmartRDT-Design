#!/usr/bin/env python3
"""
增强的适应度评估模块V3 - 6目标
修复了缓存问题并保留了原始的严谨实现
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
    """修复版的解决方案映射器 - 解决缓存问题"""
    
    def __init__(self, ontology_graph: Graph):
        self.g = ontology_graph
        self._cache_components()
        # 使用字典而非lru_cache装饰器来避免unhashable type错误
        self._decode_cache = {}
        
    def _cache_components(self):
            """缓存所有可用组件 - 修复版本"""
            self.sensors = []
            self.algorithms = []
            self.storage_systems = []
            self.comm_systems = []
            self.deployments = []
            
            logger.info("缓存本体组件...")
            
            # 定义所有传感器类型模式
            sensor_patterns = [
                'MMS_LiDAR_System', 'MMS_Camera_System', 
                'UAV_LiDAR_System', 'UAV_Camera_System',
                'TLS_System', 'Handheld_3D_Scanner',
                'FiberOptic_Sensor', 'Vehicle_LowCost_Sensor',
                'IoT_Network_System', 'Sensor', 'sensor'
            ]
            
            # 算法类型模式
            algo_patterns = [
                'DeepLearningAlgorithm', 'MachineLearningAlgorithm',
                'TraditionalAlgorithm', 'PointCloudAlgorithm',
                'Algorithm', 'algorithm'
            ]
            
            # 部署类型模式
            deploy_patterns = [
                'Deployment', 'Compute', 'Edge', 'Cloud',
                'ComputeDeployment', 'Deployment_Edge_Computing',
                'Deployment_Cloud_Computing', 'Deployment_Hybrid_Edge_Cloud',
                'Deployment_OnPremise_Server'
            ]
            
            # 遍历所有三元组
            for s, p, o in self.g:
                if p == RDF.type and str(s).startswith('http://example.org/rmtwin#'):
                    subject_str = str(s)
                    type_str = str(o)
                    
                    # 检查是否是传感器
                    is_sensor = False
                    for pattern in sensor_patterns:
                        if pattern in type_str:
                            is_sensor = True
                            break
                    
                    if is_sensor and subject_str not in self.sensors:
                        self.sensors.append(subject_str)
                        continue
                    
                    # 检查是否是算法
                    is_algorithm = False
                    for pattern in algo_patterns:
                        if pattern in type_str:
                            is_algorithm = True
                            break
                    
                    if is_algorithm and subject_str not in self.algorithms:
                        self.algorithms.append(subject_str)
                        continue
                    
                    # 检查存储系统
                    if 'Storage' in type_str and subject_str not in self.storage_systems:
                        self.storage_systems.append(subject_str)
                        continue
                    
                    # 检查通信系统
                    if 'Communication' in type_str and subject_str not in self.comm_systems:
                        self.comm_systems.append(subject_str)
                        continue
                    
                    # 检查部署系统
                    is_deployment = False
                    for pattern in deploy_patterns:
                        if pattern in type_str:
                            is_deployment = True
                            break
                    
                    if is_deployment and subject_str not in self.deployments:
                        self.deployments.append(subject_str)
            
            # 确保至少有默认值
            if not self.sensors:
                logger.warning("未找到传感器！使用默认值")
                self.sensors = ["http://example.org/rmtwin#MMS_LiDAR_Riegl_VUX1HA"]
            if not self.algorithms:
                logger.warning("未找到算法！使用默认值")
                self.algorithms = ["http://example.org/rmtwin#DL_YOLOv5s_Enhanced"]
            if not self.storage_systems:
                self.storage_systems = ["http://example.org/rmtwin#Storage_Cloud_AWS_S3"]
            if not self.comm_systems:
                self.comm_systems = ["http://example.org/rmtwin#Communication_5G_Network"]
            if not self.deployments:
                self.deployments = ["http://example.org/rmtwin#Deployment_Cloud_Computing"]
            
            logger.info(f"缓存的组件: {len(self.sensors)} 传感器, "
                    f"{len(self.algorithms)} 算法, {len(self.storage_systems)} 存储, "
                    f"{len(self.comm_systems)} 通信, {len(self.deployments)} 部署")
            
            # 调试输出 - 显示前几个传感器
            if len(self.sensors) <= 20:
                logger.debug("传感器列表:")
                for sensor in self.sensors[:5]:
                    logger.debug(f"  - {sensor.split('#')[-1]}")
                if len(self.sensors) > 5:
                    logger.debug(f"  ... 还有 {len(self.sensors)-5} 个传感器")
    
    def decode_solution(self, x: np.ndarray) -> Dict:
        """解码解决方案向量 - 修复缓存问题"""
        # 转换为元组以用作字典键
        x_key = tuple(float(xi) for xi in x)  # 确保都是float
        
        # 检查缓存
        if x_key in self._decode_cache:
            return self._decode_cache[x_key]
        
        # 解码配置
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
        
        # 存入缓存
        self._decode_cache[x_key] = config
        
        return config


class EnhancedFitnessEvaluatorV3:
    """增强的适应度评估器 - 6目标，保留原始严谨实现"""
    
    def __init__(self, ontology_graph: Graph, config):
        self.g = ontology_graph
        self.config = config
        self.solution_mapper = SolutionMapper(ontology_graph)
        
        # 组件属性缓存
        self._property_cache = {}
        
        # 归一化参数
        self.norm_params = {
            'cost': {'min': 100_000, 'max': 20_000_000},
            'recall': {'min': 0.0, 'max': 0.4},  # 1-recall
            'latency': {'min': 0.1, 'max': 500.0},
            'disruption': {'min': 0.0, 'max': 500.0},
            'environmental': {'min': 1_000, 'max': 200_000},  # kgCO2e/year
            'reliability': {'min': 0.0, 'max': 0.001}  # 1/MTBF
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
        
        # 定义要缓存的属性
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
        
        # 获取所有组件
        all_components = (
            self.solution_mapper.sensors + 
            self.solution_mapper.algorithms + 
            self.solution_mapper.storage_systems + 
            self.solution_mapper.comm_systems + 
            self.solution_mapper.deployments
        )
        
        # 缓存每个组件的属性
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
        
        # 转换配置为字典
        self._config_dict = vars(self.config) if hasattr(self.config, '__dict__') else self.config
    
    def _query_property(self, subject: str, predicate: str, default=None):
        """从缓存查询属性"""
        if subject in self._property_cache:
            if predicate in self._property_cache[subject]:
                return self._property_cache[subject][predicate]
        return default
    
    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量评估解决方案"""
        n_solutions = len(X)
        objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 5))
        
        # 使用简单的循环评估（避免进程池的复杂性）
        for i, x in enumerate(X):
            try:
                objectives[i], constraints[i] = self._evaluate_single(x)
            except Exception as e:
                logger.error(f"评估解决方案 {i} 时出错: {e}")
                # 设置惩罚值
                objectives[i] = np.array([1e10, 1, 1000, 1000, 200000, 1])
                constraints[i] = np.array([1000, 1, 1e10, 100000, -1000])
        
        # 更新统计
        self._evaluation_count += n_solutions
        
        # 定期记录进度
        if self._evaluation_count % 1000 == 0:
            # logger.info(f"已评估 {self._evaluation_count} 个解决方案。")
            
            # 记录目标范围
            logger.debug(f"目标范围:")
            logger.debug(f"  成本: ${objectives[:, 0].min():.0f} - ${objectives[:, 0].max():.0f}")
            logger.debug(f"  1-召回率: {objectives[:, 1].min():.3f} - {objectives[:, 1].max():.3f}")
            logger.debug(f"  延迟: {objectives[:, 2].min():.1f} - {objectives[:, 2].max():.1f}秒")
            logger.debug(f"  干扰: {objectives[:, 3].min():.1f} - {objectives[:, 3].max():.1f}小时")
            logger.debug(f"  碳排放: {objectives[:, 4].min():.0f} - {objectives[:, 4].max():.0f} kgCO2e")
            logger.debug(f"  1/MTBF: {objectives[:, 5].min():.6f} - {objectives[:, 5].max():.6f}")
        
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
            f3 - self.config.max_latency_seconds,  # 最大延迟
            self.config.min_recall_threshold - recall,  # 最小召回率
            f1 - self.config.budget_cap_usd,  # 预算
            f5 - self.config.max_carbon_emissions_kgCO2e_year,  # 最大碳排放
            self.config.min_mtbf_hours - (1/f6 if f6 > 0 else 1e6)  # 最小MTBF
        ])
        
        return objectives, constraints
    
    # 以下是所有详细的计算方法（保留原始实现）
    def _calculate_total_cost_enhanced(self, config: Dict) -> float:
        """增强的成本计算 - 保留原始实现的所有细节"""
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
        
        # 年度运营成本（含折旧）
        depreciation_rate = self.config.depreciation_rate
        annual_capital_cost = total_initial_investment * depreciation_rate
        
        # 传感器运营成本
        sensor_daily_cost = self._query_property(
            config['sensor'], 'hasOperationalCostUSDPerDay', 100)
        coverage_km_day = self._query_property(
            config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        
        road_length = self.config.road_network_length_km
        
        if coverage_km_day > 0:  # 移动传感器
            inspections_per_year = 365 / config['inspection_cycle']
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
        
        # 人员成本（含技能系数）
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
            
            data_annotation_annual = annotation_cost * annual_images
        
        # 模型重训练成本
        retrain_freq = self._query_property(config['algorithm'], 'hasModelRetrainingFreqMonths', 12)
        model_retraining_annual = 0
        if retrain_freq and retrain_freq > 0:
            retrainings_per_year = 12 / retrain_freq
            retraining_cost = 5000  # 基础重训练成本
            model_retraining_annual = retraining_cost * retrainings_per_year
        
        # 年度总成本
        total_annual_cost = (annual_capital_cost + sensor_annual_operational + 
                           storage_annual + comm_annual + deployment_annual + 
                           crew_annual_cost + data_annotation_annual + model_retraining_annual)
        
        # 应用季节性调整
        if self.config.apply_seasonal_adjustments:
            winter_factor = 1.3  # 冬季运营成本增加30%
            seasonal_adjustment = 0.25  # 25%的运营在冬季
            total_annual_cost *= (1 + (winter_factor - 1) * seasonal_adjustment)
        
        # 应用软约束惩罚
        if config['crew_size'] > 5:
            total_annual_cost += (config['crew_size'] - 5) * 50000
        
        if config['inspection_cycle'] < 7:  # 紧急调度
            total_annual_cost *= 1.5  # 加班系数
        
        # 总生命周期成本
        planning_horizon = self.config.planning_horizon_years
        total_lifecycle_cost = total_annual_cost * planning_horizon
        
        return total_lifecycle_cost
    
    def _calculate_detection_performance_enhanced(self, config: Dict) -> float:
        """增强的检测性能计算"""
        base_recall = self._query_property(config['algorithm'], 'hasRecall', 0.7)
        
        # 传感器精度影响
        accuracy_mm = self._query_property(config['sensor'], 'hasAccuracyRangeMM', 10)
        accuracy_factor = 1 - (accuracy_mm / 100)
        base_recall *= (0.8 + 0.2 * accuracy_factor)
        
        # LOD影响（细致的因子）
        lod_factors = {
            'Micro': {'factor': 1.15, 'threshold_adj': 0.05},
            'Meso': {'factor': 1.0, 'threshold_adj': 0},
            'Macro': {'factor': 0.85, 'threshold_adj': -0.05}
        }
        
        lod_data = lod_factors.get(config['geo_lod'], lod_factors['Meso'])
        base_recall *= lod_data['factor']
        
        # 检测阈值影响
        threshold_optimal = 0.5
        threshold_penalty = abs(config['detection_threshold'] - threshold_optimal) * 0.1
        base_recall -= threshold_penalty
        
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
        
        # 硬件需求影响
        hardware_req = self._query_property(config['algorithm'], 'hasHardwareRequirement', 'CPU')
        if 'HighEnd_GPU' in str(hardware_req):
            base_recall += 0.02  # 高端GPU能提升性能
        
        # 算法精度考虑
        precision = self._query_property(config['algorithm'], 'hasPrecision', 0.8)
        base_recall *= (0.9 + 0.1 * precision)
        
        # 添加小的随机噪声以增加真实性
        noise = np.random.normal(0, 0.005)
        base_recall += noise
        
        return 1 - np.clip(base_recall, 0.01, 0.99)  # 返回1-recall用于最小化
    
    def _calculate_latency_enhanced(self, config: Dict) -> float:
        """增强的延迟计算 - 更真实的估计"""
        # 数据采集时间（考虑实际扫描时间）
        if 'MMS' in str(config['sensor']):
            # 移动测绘系统需要覆盖整个路段
            coverage_speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
            road_length = self.config.road_network_length_km
            acq_time = (road_length / coverage_speed) * 3600  # 转换为秒
        else:
            data_rate = config['data_rate']
            acq_time = 60 / data_rate  # 假设每次采集需要处理60个样本
        
        # 数据传输时间（更真实的估计）
        base_data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        lod_multipliers = {'Micro': 2.0, 'Meso': 1.0, 'Macro': 0.5}
        data_gb = base_data_gb * lod_multipliers.get(config['geo_lod'], 1.0)
        
        # 通信时间（考虑实际网络条件）
        comm_type = str(config['communication']).split('#')[-1]
        if 'LoRaWAN' in comm_type:
            comm_time = data_gb * 1000 * 8 / 0.05  # 50kbps
        elif '4G' in comm_type:
            comm_time = data_gb * 1000 * 8 / 50  # 实际4G速度约50Mbps
        elif '5G' in comm_type:
            comm_time = data_gb * 1000 * 8 / 200  # 实际5G速度约200Mbps
        else:  # Fiber
            comm_time = data_gb * 1000 * 8 / 1000  # 1Gbps
        
        # 处理时间（基于算法复杂度）
        algo_name = str(config['algorithm']).split('#')[-1]
        if 'DL' in algo_name or 'Deep' in algo_name:
            base_proc_time = 30  # 深度学习需要更多时间
        elif 'ML' in algo_name:
            base_proc_time = 10
        else:
            base_proc_time = 5
        
        # 部署位置的影响
        deploy_factors = {
            'Deployment_Edge_Computing': 2.0,  # 边缘计算资源有限
            'Deployment_Cloud_Computing': 1.0,
            'Deployment_Hybrid_Edge_Cloud': 1.5,
            'Deployment_OnPremise_Server': 1.8
        }
        
        deploy_type = str(config['deployment']).split('#')[-1]
        deploy_factor = deploy_factors.get(deploy_type, 1.0)
        proc_time = base_proc_time * deploy_factor
        
        # 总延迟
        total_latency = acq_time + comm_time + proc_time
        
        return total_latency
    
    def _calculate_traffic_disruption_enhanced(self, config: Dict) -> float:
        """增强的交通干扰计算"""
        # 每次检查的基础干扰
        base_disruption = 4.0  # 小时
        
        # 传感器特定调整
        sensor_name = str(config['sensor']).split('#')[-1]
        speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
        
        if 'UAV' in sensor_name:
            # 无人机造成的交通干扰最小
            disruption_factor = 0.1
        elif 'FOS' in sensor_name or 'IoT' in sensor_name:
            # 固定传感器 - 仅在安装/维护时产生干扰
            disruption_factor = 0.05
        elif speed > 0:
            # 移动传感器 - 基于速度相对于交通流的干扰
            speed_factor = max(0.1, min(2.0, 80 / speed))
            disruption_factor = speed_factor
        else:
            disruption_factor = 1.0
        
        # 计算年度干扰
        inspections_per_year = 365 / config['inspection_cycle']
        
        if 'FOS' in sensor_name or 'IoT' in sensor_name:
            # 固定传感器 - 仅维护访问
            maintenance_visits = 4  # 季度维护
            annual_disruption = base_disruption * disruption_factor * maintenance_visits
        else:
            # 移动传感器
            road_length = self.config.road_network_length_km
            coverage_per_day = self._query_property(
                config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
            
            if coverage_per_day > 0:
                days_per_inspection = road_length / coverage_per_day
                annual_disruption = base_disruption * disruption_factor * days_per_inspection * inspections_per_year
            else:
                annual_disruption = base_disruption * disruption_factor * inspections_per_year
        
        # 交通量影响
        traffic_volume = self.config.traffic_volume_hourly
        traffic_factor = (traffic_volume / 1000) ** 0.5  # 影响递减的平方根
        
        # 车道封闭影响
        default_lane_closure = self.config.default_lane_closure_ratio
        lane_factor = 1 + default_lane_closure
        
        # 时间优化
        night_work_ratio = 0.3  # 30%的工作可以在夜间进行
        night_impact_reduction = 0.7  # 夜间工作影响减少70%
        time_factor = 1 - (night_work_ratio * night_impact_reduction)
        
        # 人员规模影响（人员越多工作越快但干扰越大）
        crew_factor = 1 + (config['crew_size'] - 3) * 0.1
        
        # 总干扰
        total_disruption = annual_disruption * lane_factor * traffic_factor * time_factor * crew_factor
        
        return total_disruption
    
    def _calculate_environmental_impact(self, config: Dict) -> float:
        """计算环境影响（kgCO2e/年）- f5"""
        # 组件能耗
        total_power_w = 0
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                power = self._query_property(config[comp], 'hasEnergyConsumptionW', 0)
                if power:
                    total_power_w += power
        
        # 算法计算需求
        algo_name = str(config['algorithm']).split('#')[-1]
        if 'DL' in algo_name or 'Deep' in algo_name:
            total_power_w += 250  # 深度学习需要更多计算
        elif 'ML' in algo_name:
            total_power_w += 100
        else:
            total_power_w += 50
        
        # 运行时间计算
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        road_length = self.config.road_network_length_km
        
        if coverage > 0:
            # 移动传感器运行
            inspections_per_year = 365 / config['inspection_cycle']
            days_per_inspection = road_length / coverage
            sensor_hours = days_per_inspection * inspections_per_year * 8  # 8小时/天
            
            # 车辆排放
            vehicle_km = road_length * inspections_per_year
            vehicle_fuel_l = vehicle_km * 0.08  # 8L/100km油耗
            vehicle_emissions_kg = vehicle_fuel_l * 2.31  # 2.31 kg CO2/L汽油
        else:
            # 固定传感器 - 连续运行
            sensor_hours = 365 * 24
            vehicle_emissions_kg = 0
        
        # 后端/云运行 - 始终运行
        backend_hours = 365 * 24
        
        # 能耗分解
        sensor_energy_kwh = (total_power_w * 0.3 * sensor_hours) / 1000
        backend_energy_kwh = (total_power_w * 0.7 * backend_hours) / 1000
        total_energy_kwh = sensor_energy_kwh + backend_energy_kwh
        
        # 碳强度
        carbon_intensity = self.config.carbon_intensity_factor  # kg CO2/kWh
        electricity_emissions_kg = total_energy_kwh * carbon_intensity
        
        # 制造排放（按寿命摊销）
        equipment_cost = self._query_property(config['sensor'], 'hasInitialCostUSD', 100000)
        manufacturing_emissions = equipment_cost * 0.001  # 粗略估计：1 kg CO2/$1000
        annual_manufacturing = manufacturing_emissions / 10  # 10年寿命
        
        # 数据存储排放
        data_volume_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        annual_data_gb = data_volume_gb * road_length * (365 / config['inspection_cycle'])
        storage_emissions_kg = annual_data_gb * 0.01  # 0.01 kg CO2/GB存储
        
        # 总排放量（kgCO2e/年）
        total_emissions = (electricity_emissions_kg + vehicle_emissions_kg + 
                          annual_manufacturing + storage_emissions_kg)
        
        return total_emissions
    
    def _calculate_system_reliability(self, config: Dict) -> float:
        """计算系统可靠性（1/MTBF）- f6"""
        # 组件可靠性（串联系统模型）
        component_mtbfs = []
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                base_mtbf = self._query_property(config[comp], 'hasMTBFHours', 10000)
                
                if base_mtbf and base_mtbf > 0:
                    # 基于部署类型应用冗余因子
                    comp_type = str(config[comp]).split('#')[-1]
                    
                    redundancy_multipliers = self.config.redundancy_multipliers
                    
                    redundancy = 1.0
                    for red_type, mult in redundancy_multipliers.items():
                        if red_type in comp_type:
                            redundancy = mult
                            break
                    
                    # 环境因素
                    if comp == 'sensor':
                        # 户外传感器可靠性降低
                        environmental_factor = 0.8
                    else:
                        environmental_factor = 1.0
                    
                    # 维护影响
                    calibration_freq = self._query_property(
                        config[comp], 'hasCalibrationFreqMonths', 12)
                    if calibration_freq and calibration_freq > 0:
                        maintenance_factor = min(1.2, 1 + (12 / calibration_freq) * 0.1)
                    else:
                        maintenance_factor = 1.0
                    
                    effective_mtbf = base_mtbf * redundancy * environmental_factor * maintenance_factor
                    component_mtbfs.append(effective_mtbf)
        
        # 算法可靠性（软件）
        algo_mtbf = self._query_property(config['algorithm'], 'hasMTBFHours', 50000)
        if algo_mtbf and algo_mtbf > 0:
            # 软件可靠性受复杂性影响
            explainability = self._query_property(
                config['algorithm'], 'hasExplainabilityScore', 3)
            complexity_factor = 0.7 + (explainability / 5) * 0.3 if explainability else 1.0
            component_mtbfs.append(algo_mtbf * complexity_factor)
        
        # 系统MTBF计算（串联系统）
        if component_mtbfs:
            # 对于串联系统：1/MTBF_system = sum(1/MTBF_i)
            inverse_mtbf_sum = sum(1/mtbf for mtbf in component_mtbfs)
            
            # 添加共因故障
            common_cause_rate = 1 / (365 * 24 * 10)  # 10年一次
            inverse_mtbf_sum += common_cause_rate
            
            # 基于人员规模和技能添加人为错误因子
            skill_level = self._query_property(config['sensor'], 'hasOperatorSkillLevel', 'Basic')
            human_error_rates = {
                'Basic': 1 / (365 * 24 * 0.5),  # 每6个月一次错误
                'Intermediate': 1 / (365 * 24 * 1),  # 每年一次错误
                'Expert': 1 / (365 * 24 * 2)  # 每2年一次错误
            }
            human_error_rate = human_error_rates.get(str(skill_level), human_error_rates['Basic'])
            
            # 人员规模影响（人越多出错可能性越大）
            crew_factor = 1 + (config['crew_size'] - 1) * 0.1
            inverse_mtbf_sum += human_error_rate * crew_factor
            
            return inverse_mtbf_sum
        else:
            return 1e-6  # 如果没有数据则默认为非常高的可靠性
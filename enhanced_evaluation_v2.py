#!/usr/bin/env python3
"""
增强版评估模块 V2.0 - 集成专家建议的高级建模
Enhanced Evaluation Module V2.0 - Integrating Expert Recommendations

完整实现六个目标函数的专家级改进
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS, RDF, OWL, XSD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")

@dataclass
class AdvancedEvaluationConfig:
    """高级评估配置 - 包含所有专家建议的参数"""
    
    # 基础参数
    road_network_length_km: float = 500.0
    planning_horizon_years: int = 10
    budget_cap_usd: float = 20_000_000
    
    # f1: 成本相关
    depreciation_rate: float = 0.1  
    daily_wage_per_person: float = 1500
    fos_sensor_spacing_km: float = 0.1  # 添加这个！
    
    # f2: 检测性能
    class_imbalance_penalties: Dict[str, float] = None
    
    # f3: 延迟相关
    scenario_type: str = 'urban'  # urban, rural, mixed
    network_quality_factors: Dict[str, Dict[str, float]] = None
    
    # f4: 交通干扰
    traffic_volume_hourly: int = 2000
    peak_hour_factor: float = 2.0
    night_hour_factor: float = 0.3
    default_lane_closure_ratio: float = 0.3
    
    # f5: 环境影响
    carbon_intensity_factor: float = 0.417  # kgCO2e/kWh (US average)
    
    # f6: 可靠性
    redundancy_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.class_imbalance_penalties is None:
            self.class_imbalance_penalties = {
                'Traditional': 0.05,
                'ML': 0.02,
                'DL': 0.01,
                'PC': 0.03
            }
        
        if self.network_quality_factors is None:
            self.network_quality_factors = {
                'rural': {'Fiber': 0.8, '5G': 0.7, '4G': 0.9, 'LoRaWAN': 1.0},
                'urban': {'Fiber': 1.0, '5G': 1.0, '4G': 1.0, 'LoRaWAN': 0.9}
            }
        
        if self.redundancy_multipliers is None:
            self.redundancy_multipliers = {
                'Cloud': 10.0,
                'OnPremise': 1.5,
                'Edge': 2.0,
                'Hybrid': 5.0
            }

class EnhancedFitnessEvaluatorV2:
    """增强版适应度评估器 V2"""
    
    def __init__(self, ontology_graph: Graph, config: AdvancedEvaluationConfig):
        self.g = ontology_graph
        self.config = config
        self._cache = {}
        
    def evaluate_solution(self, x: np.ndarray, decoded_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """评估单个解决方案"""
        
        # 计算六个目标
        f1 = self._calculate_total_cost_v2(decoded_config)
        f2 = self._calculate_detection_performance_v2(decoded_config)
        f3 = self._calculate_latency_v2(decoded_config)
        f4 = self._calculate_traffic_disruption_v2(decoded_config)
        f5 = self._calculate_environmental_impact_v2(decoded_config)
        f6 = self._calculate_system_reliability_v2(decoded_config)
        
        objectives = np.array([f1, f2, f3, f4, f5, f6])
        
        # 计算约束
        recall = 1 - f2
        constraints = np.array([
            f3 - 180.0,  # 最大延迟
            0.70 - recall,  # 最小召回率
            f1 - self.config.budget_cap_usd  # 预算
        ])
        
        return objectives, constraints
    
    def _query_property(self, subject: str, predicate: str, default=None):
        """查询属性值（带缓存）"""
        cache_key = f"{subject}#{predicate}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        query = f"""
        SELECT ?value WHERE {{
            <{subject}> <{predicate}> ?value .
        }}
        """
        
        for row in self.g.query(query):
            try:
                value = float(str(row.value))
            except:
                value = str(row.value)
            self._cache[cache_key] = value
            return value
            
        self._cache[cache_key] = default
        return default
    
    def _calculate_total_cost_v2(self, config: Dict) -> float:
        """
        增强版成本计算 - 包含FOS修复和年化资本成本
        
        成本组成：
        1. 初始投资（设备+安装）
        2. 年化资本成本（折旧）
        3. 运营成本（能源+维护）
        4. 人力成本
        """
        
        # 提取传感器名称
        sensor_name = str(config['sensor']).split('#')[-1]
        
        # ========== 1. 计算初始投资成本 ==========
            
        # 获取单个传感器成本
        sensor_initial_cost = self._query_property(
            config['sensor'], str(RDTCO.hasInitialCostUSD), 100000)
        
        # 关键修复：FOS需要多个传感器覆盖整个网络
        if 'FOS' in sensor_name or 'Fiber' in sensor_name:
            # 光纤传感器需要沿道路密集部署
            sensor_spacing_km = self.config.fos_sensor_spacing_km  # 使用配置中的值
            sensors_needed = self.config.road_network_length_km / sensor_spacing_km
            
            # 实际传感器成本 = 单价 × 数量
            actual_sensor_cost = sensor_initial_cost * sensors_needed
            
            # 安装成本（每个传感器点的安装费用）
            installation_cost_per_sensor = 5000  # 包括挖掘、布线、保护等
            total_installation_cost = installation_cost_per_sensor * sensors_needed
            
            # FOS的总初始成本
            total_sensor_initial = actual_sensor_cost + total_installation_cost
            
            # 添加详细日志
            logger.debug(f"FOS cost calculation:")
            logger.debug(f"  Network length: {self.config.road_network_length_km} km")
            logger.debug(f"  Sensor spacing: {sensor_spacing_km} km")
            logger.debug(f"  Sensors needed: {sensors_needed:.0f}")
            logger.debug(f"  Unit cost: ${sensor_initial_cost:.0f}")
            logger.debug(f"  Total sensor cost: ${actual_sensor_cost:.0f}")
            logger.debug(f"  Installation cost: ${total_installation_cost:.0f}")
            logger.debug(f"  Total initial: ${total_sensor_initial:.0f}")
        else:
            # 移动传感器只需要一套设备
            total_sensor_initial = sensor_initial_cost
            logger.debug(f"Mobile sensor {sensor_name}: ${total_sensor_initial:.0f}")
        
        # 其他组件的初始成本
        storage_initial = self._query_property(
            config['storage'], str(RDTCO.hasInitialCostUSD), 0)
        comm_initial = self._query_property(
            config['communication'], str(RDTCO.hasInitialCostUSD), 0)
        deployment_initial = self._query_property(
            config['deployment'], str(RDTCO.hasInitialCostUSD), 0)
        
        # 算法开发/许可成本
        algo_initial = self._query_property(
            config['algorithm'], str(RDTCO.hasInitialCostUSD), 20000)
        
        # 总初始投资
        total_initial_investment = (total_sensor_initial + storage_initial + 
                                comm_initial + deployment_initial + algo_initial)
        
        # ========== 2. 计算年化资本成本（折旧） ==========
        
        depreciation_rate = getattr(self.config, 'depreciation_rate', 0.1)  # 10年折旧
        annual_capital_cost = total_initial_investment * depreciation_rate
        
        # ========== 3. 计算年度运营成本 ==========
        
        # 传感器运营成本
        sensor_daily_cost = self._query_property(
            config['sensor'], str(RDTCO.hasOperationalCostUSDPerDay), 100)
        
        # 覆盖效率（km/day）
        coverage_km_day = self._query_property(
            config['sensor'], str(RDTCO.hasCoverageEfficiencyKmPerDay), 80)
        
        if coverage_km_day > 0:  # 移动传感器
            # 每年需要的巡检次数
            inspections_per_year = 365 / config['inspection_cycle']
            # 每次巡检需要的天数
            days_per_inspection = self.config.road_network_length_km / coverage_km_day
            # 年度传感器运营成本
            sensor_annual_operational = sensor_daily_cost * days_per_inspection * inspections_per_year
        else:  # 固定传感器（FOS）
            if 'FOS' in sensor_name:
                # FOS的运营成本包括：电力、数据传输、定期检查
                # 假设每个传感器点每天0.5美元的运营成本
                operational_cost_per_sensor_day = 0.5
                sensors_needed = self.config.road_network_length_km / sensor_spacing_km
                sensor_annual_operational = operational_cost_per_sensor_day * sensors_needed * 365
            else:
                # 其他固定传感器
                sensor_annual_operational = sensor_daily_cost * 365
        
        # 其他组件的年度运营成本
        storage_annual = self._query_property(
            config['storage'], str(RDTCO.hasAnnualOpCostUSD), 5000)
        comm_annual = self._query_property(
            config['communication'], str(RDTCO.hasAnnualOpCostUSD), 2000)
        deployment_annual = self._query_property(
            config['deployment'], str(RDTCO.hasAnnualOpCostUSD), 10000)
        
        # ========== 4. 计算人力成本 ==========
        
        # 获取操作员技能等级
        skill_level = self._query_property(
            config['sensor'], str(RDTCO.hasOperatorSkillLevel), 'Basic')
        
        # 技能等级工资乘数
        skill_multiplier = {
            'Basic': 1.0,
            'Intermediate': 1.5,
            'Expert': 2.0
        }.get(str(skill_level), 1.0)
        
        # 基础日工资
        daily_wage = self.config.daily_wage_per_person * skill_multiplier
        
        if coverage_km_day > 0:  # 移动传感器需要操作员
            # 年度人力成本
            crew_annual_cost = (config['crew_size'] * daily_wage * 
                            days_per_inspection * inspections_per_year)
        else:  # 固定传感器
            if 'FOS' in sensor_name:
                # FOS只需要偶尔的维护检查，假设每年10天
                maintenance_days_per_year = 10
                crew_annual_cost = config['crew_size'] * daily_wage * maintenance_days_per_year
            else:
                # 其他固定传感器
                crew_annual_cost = config['crew_size'] * daily_wage * 20
        
        # ========== 5. 计算数据标注成本（仅深度学习） ==========
        
        data_annotation_annual = 0
        if 'DL' in str(config['algorithm']) or 'Deep' in str(config['algorithm']):
            # 深度学习需要持续的数据标注
            annotation_cost_per_image = self._query_property(
                config['algorithm'], str(RDTCO.hasDataAnnotationCostUSD), 0.5)
            
            # 估算年度图像数量
            if 'Camera' in sensor_name:
                images_per_km = 100  # 每公里100张图像
                annual_images = images_per_km * self.config.road_network_length_km * inspections_per_year
            else:
                annual_images = 10000  # 其他传感器的默认值
            
            data_annotation_annual = annotation_cost_per_image * annual_images
        
        # ========== 6. 总成本计算 ==========
        
        # 年度总成本
        total_annual_cost = (annual_capital_cost + 
                            sensor_annual_operational + 
                            storage_annual + 
                            comm_annual + 
                            deployment_annual + 
                            crew_annual_cost + 
                            data_annotation_annual)
        
        # 全生命周期总成本
        total_lifecycle_cost = total_annual_cost * self.config.planning_horizon_years
        
        # 添加季节性调整（如果适用）
        if hasattr(self.config, 'seasonal_factor'):
            total_lifecycle_cost *= self.config.seasonal_factor
        
        logger.debug(f"Cost breakdown for {sensor_name}: "
                    f"Initial=${total_initial_investment:.0f}, "
                    f"Annual=${total_annual_cost:.0f}, "
                    f"Total=${total_lifecycle_cost:.0f}")
        
        return total_lifecycle_cost
    
    def _calculate_detection_performance_v2(self, config: Dict) -> float:
        """增强版检测性能 - 类别不平衡惩罚"""
        
        base_recall = self._query_property(config['algorithm'], str(RDTCO.hasRecall), 0.7)
        
        # 传感器精度影响
        accuracy_mm = self._query_property(config['sensor'], str(RDTCO.hasAccuracyRangeMM), 10)
        accuracy_factor = 1 - (accuracy_mm / 100)
        
        # LOD影响
        lod_factor = {'Micro': 1.1, 'Meso': 1.0, 'Macro': 0.9}.get(config['geo_lod'], 1.0)
        
        # 类别不平衡惩罚（专家建议）
        algo_name = str(config['algorithm']).split('#')[-1]
        penalty = 0
        for algo_type, pen_value in self.config.class_imbalance_penalties.items():
            if algo_type in algo_name:
                penalty = pen_value
                break
        
        final_recall = base_recall * accuracy_factor * lod_factor - penalty
        return 1 - np.clip(final_recall, 0.01, 0.99)
    
    def _calculate_latency_v2(self, config: Dict) -> float:
        """增强版延迟 - 场景依赖的带宽"""
        
        # 数据量
        data_gb = self._query_property(config['sensor'], str(RDTCO.hasDataVolumeGBPerKm), 1.0)
        
        # 基础带宽
        comm_type = str(config['communication']).split('#')[-1]
        base_bw = {
            'Communication_5G_Network': 1000,
            'Communication_LoRaWAN': 0.05,
            'Communication_Fiber_Optic': 10000,
            'Communication_4G_LTE': 100
        }.get(comm_type, 100)
        
        # 场景因子（专家建议）
        tech = None
        for t in ['Fiber', '5G', '4G', 'LoRaWAN']:
            if t in comm_type:
                tech = t
                break
        
        scenario_factor = 1.0
        if tech and self.config.scenario_type in self.config.network_quality_factors:
            scenario_factor = self.config.network_quality_factors[self.config.scenario_type].get(tech, 1.0)
        
        effective_bw = base_bw * scenario_factor
        
        # 通信时间
        comm_time = (data_gb * 1000) / effective_bw if effective_bw > 0 else 100
        
        # 处理时间
        deploy_factor = {
            'Edge': 1.5, 'Cloud': 1.0, 'Hybrid': 1.2, 'OnPremise': 1.3
        }.get(config['deployment'].split('_')[-1], 1.0)
        
        proc_time = 0.1 * deploy_factor
        
        return 1/config['data_rate'] + comm_time + proc_time
    
    def _calculate_traffic_disruption_v2(self, config: Dict) -> float:
        """增强版交通干扰 - 精细化模型"""
        
        base_hours = 4.0
        inspections_year = 365 / config['inspection_cycle']
        
        speed = self._query_property(config['sensor'], str(RDTCO.hasOperatingSpeedKmh), 80)
        
        if speed > 0:
            speed_factor = 80 / speed if speed < 80 else 1.0
            disruption = base_hours * speed_factor * inspections_year
        else:
            disruption = 0.1 * inspections_year  # 固定传感器
        
        # 交通影响（专家建议）
        lane_factor = 1 + self.config.default_lane_closure_ratio
        traffic_factor = self.config.traffic_volume_hourly / 1000
        
        return disruption * lane_factor * traffic_factor
    
    def _calculate_environmental_impact_v2(self, config: Dict) -> float:
        """增强版环境影响 - 碳足迹"""
        
        total_power_w = 0
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                power = self._query_property(config[comp], str(RDTCO.hasEnergyConsumptionW), 0)
                total_power_w += power if power else 0
        
        # 运行时间
        coverage = self._query_property(config['sensor'], str(RDTCO.hasCoverageEfficiencyKmPerDay), 80)
        
        if coverage > 0:
            sensor_hours = (self.config.road_network_length_km / coverage) * (365 / config['inspection_cycle']) * 8
            vehicle_km = self.config.road_network_length_km * (365 / config['inspection_cycle'])
            vehicle_kwh = vehicle_km * 0.8
        else:
            sensor_hours = 365 * 24
            vehicle_kwh = 0
        
        backend_hours = 365 * 24
        
        total_kwh = (total_power_w * 0.3 * sensor_hours + 
                    total_power_w * 0.7 * backend_hours) / 1000 + vehicle_kwh
        
        # 碳排放（专家建议）
        return total_kwh * self.config.carbon_intensity_factor
    
    def _calculate_system_reliability_v2(self, config: Dict) -> float:
        """增强版可靠性 - 考虑冗余"""
        
        inverse_mtbf = 0
        
        for comp in ['sensor', 'storage', 'communication', 'deployment']:
            if comp in config:
                base_mtbf = self._query_property(config[comp], str(RDTCO.hasMTBFHours), 10000)
                
                # 冗余因子（专家建议）
                comp_type = str(config[comp]).split('#')[-1]
                redundancy = 1.0
                for red_type, mult in self.config.redundancy_multipliers.items():
                    if red_type in comp_type:
                        redundancy = mult
                        break
                
                effective_mtbf = base_mtbf * redundancy
                if effective_mtbf > 0:
                    inverse_mtbf += 1 / effective_mtbf
        
        return inverse_mtbf if inverse_mtbf > 0 else 1e-6

class DynamicNormalizer:
    """动态归一化器"""
    
    def __init__(self):
        self.history = []
        self.window_size = 50
        
    def normalize_population(self, raw_objectives: np.ndarray) -> np.ndarray:
        """动态归一化 - 避免截断帕累托前沿"""
        
        self.history.append(raw_objectives)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # 使用历史数据确定范围
        all_data = np.vstack(self.history)
        
        normalized = np.zeros_like(raw_objectives)
        for i in range(raw_objectives.shape[1]):
            col_data = all_data[:, i]
            min_val = np.percentile(col_data, 1)
            max_val = np.percentile(col_data, 99)
            
            if max_val - min_val > 1e-10:
                normalized[:, i] = (raw_objectives[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5
                
        return np.clip(normalized, 0, 1)
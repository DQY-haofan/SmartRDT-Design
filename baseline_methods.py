#!/usr/bin/env python3
"""
Baseline Methods for RMTwin Configuration
Complete implementation of all baseline approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import time
from abc import ABC, abstractmethod

# Import will be done in the classes to avoid circular import

logger = logging.getLogger(__name__)


class BaselineMethod(ABC):
    """Abstract base class for baseline methods"""
    
    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config
        self.results = []
        self.execution_time = 0
        
    @abstractmethod
    def optimize(self) -> pd.DataFrame:
        """Run optimization and return results"""
        pass
    
    def _create_result_entry(self, x: np.ndarray, objectives: np.ndarray, 
                           constraints: np.ndarray, solution_id: int) -> Dict:
        """Create standardized result entry"""
        config = self.evaluator.solution_mapper.decode_solution(x)
        is_feasible = np.all(constraints <= 0)
        
        return {
            'solution_id': solution_id,
            'method': self.__class__.__name__,
            
            # Configuration
            'sensor': config['sensor'].split('#')[-1],
            'algorithm': config['algorithm'].split('#')[-1],
            'data_rate_Hz': config['data_rate'],
            'geometric_LOD': config['geo_lod'],
            'condition_LOD': config['cond_lod'],
            'detection_threshold': config['detection_threshold'],
            'storage': config['storage'].split('#')[-1],
            'communication': config['communication'].split('#')[-1],
            'deployment': config['deployment'].split('#')[-1],
            'crew_size': config['crew_size'],
            'inspection_cycle_days': config['inspection_cycle'],
            
            # Objectives
            'f1_total_cost_USD': float(objectives[0]),
            'f2_one_minus_recall': float(objectives[1]),
            'f3_latency_seconds': float(objectives[2]),
            'f4_traffic_disruption_hours': float(objectives[3]),
            'f5_carbon_emissions_kgCO2e_year': float(objectives[4]),
            'f6_system_reliability_inverse_MTBF': float(objectives[5]),
            
            # Derived metrics
            'detection_recall': float(1 - objectives[1]),
            'system_MTBF_hours': float(1/objectives[5] if objectives[5] > 0 else 1e6),
            'system_MTBF_years': float(1/objectives[5]/8760 if objectives[5] > 0 else 100),
            'annual_cost_USD': float(objectives[0] / self.config.planning_horizon_years),
            'cost_per_km_year': float(objectives[0] / self.config.planning_horizon_years / 
                                    self.config.road_network_length_km),
            'carbon_footprint_tons_CO2_year': float(objectives[4] / 1000),
            
            # Feasibility
            'is_feasible': bool(is_feasible),
            'time_seconds': self.execution_time
        }

class RandomSearchBaseline(BaselineMethod):
    """改进的随机搜索基线方法"""
    
    def optimize(self) -> pd.DataFrame:
        """生成随机解决方案，带有智能初始化和约束感知"""
        n_samples = self.config.n_random_samples
        logger.info(f"Running Random Search with {n_samples} samples...")
        
        start_time = time.time()
        
        # 首先学习一些参考可行配置
        reference_configs = self._generate_reference_configs()
        
        for i in range(n_samples):
            if i < len(reference_configs):
                # 使用参考配置
                x = reference_configs[i]
            elif np.random.random() < 0.7:
                # 70% 使用智能随机
                x = self._generate_smart_random_solution()
            else:
                # 30% 纯随机探索
                x = np.random.rand(11)
            
            # 如果已经有可行解，尝试从可行解变异
            if len(self.results) > 50 and np.random.random() < 0.3:
                feasible_results = [r for r in self.results if r['is_feasible']]
                if feasible_results:
                    base_idx = np.random.randint(0, len(feasible_results))
                    x = self._mutate_from_feasible(feasible_results[base_idx])
            
            # 评估
            objectives, constraints = self.evaluator._evaluate_single(x)
            
            # 存储结果
            self.results.append(
                self._create_result_entry(x, objectives, constraints, i+1)
            )
            
            # 进度日志
            if (i + 1) % 100 == 0:
                feasible_count = sum(1 for r in self.results if r['is_feasible'])
                logger.info(f"  Evaluated {i + 1} random solutions... ({feasible_count} feasible)")
        
        self.execution_time = time.time() - start_time
        
        # 更新时间
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Random Search completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Found {df['is_feasible'].sum()} feasible solutions")
        
        return df
    
    def _generate_reference_configs(self) -> List[np.ndarray]:
        """生成一些参考配置"""
        configs = []
        
        # 配置1：低成本IoT方案
        configs.append(np.array([
            1.0, 0.2, 0.5, 0.5, 0.8, 0.7, 0.0, 0.3, 1.0, 0.2, 0.15
        ]))
        
        # 配置2：平衡的车载方案
        configs.append(np.array([
            0.85, 0.4, 0.5, 0.5, 0.5, 0.6, 0.0, 0.5, 0.8, 0.3, 0.1
        ]))
        
        # 配置3：中等性能方案
        configs.append(np.array([
            0.5, 0.5, 0.5, 0.5, 0.4, 0.7, 0.0, 0.6, 0.5, 0.3, 0.12
        ]))
        
        # 生成更多变体
        while len(configs) < 10:
            base = configs[np.random.randint(0, min(3, len(configs)))]
            mutated = base + np.random.normal(0, 0.1, 11)
            mutated = np.clip(mutated, 0, 1)
            configs.append(mutated)
        
        return configs
    
    def _generate_smart_random_solution(self) -> np.ndarray:
        """生成智能随机解决方案"""
        x = np.zeros(11)
        
        # 传感器：偏向便宜的选项
        x[0] = np.random.beta(2, 1)  # 偏向高值（便宜的传感器）
        
        # 数据率：中等
        x[1] = np.random.uniform(0.3, 0.7)
        
        # LOD：偏好Meso
        x[2] = np.random.choice([0.33, 0.5, 0.67], p=[0.2, 0.6, 0.2])
        x[3] = x[2]  # 相同的LOD
        
        # 算法：任意
        x[4] = np.random.random()
        
        # 检测阈值：中等到高
        x[5] = np.random.uniform(0.5, 0.8)
        
        # 存储：偏好云（便宜）
        x[6] = np.random.choice([0.0, 0.5, 1.0], p=[0.6, 0.3, 0.1])
        
        # 通信：中等
        x[7] = np.random.uniform(0.3, 0.7)
        
        # 部署：偏好云
        x[8] = np.random.choice([0.0, 0.5, 1.0], p=[0.2, 0.3, 0.5])
        
        # 团队规模：小到中等
        x[9] = np.random.uniform(0.1, 0.5)  # 1-5人
        
        # 检查周期：月度到季度
        x[10] = np.random.uniform(0.08, 0.25)  # 30-90天
        
        return x
    
    def _mutate_from_feasible(self, feasible_result: Dict) -> np.ndarray:
        """从可行解变异"""
        # 简化的重构 - 实际应该反向解码
        x = np.random.rand(11)
        x[0] = np.random.uniform(0.8, 1.0)  # 倾向便宜的传感器
        x[2] = 0.5  # Meso LOD
        x[3] = 0.5
        x[5] = 0.7  # 合理的阈值
        x[6] = 0.0  # 云存储
        x[8] = 1.0  # 云部署
        
        # 添加小的扰动
        noise = np.random.normal(0, 0.05, 11)
        x = np.clip(x + noise, 0, 1)
        
        return x


class GridSearchBaseline(BaselineMethod):
    """改进的网格搜索，专注于可行区域"""
    
    def optimize(self) -> pd.DataFrame:
        """在关键变量上进行网格搜索"""
        resolution = self.config.grid_resolution
        logger.info(f"Running Grid Search with resolution {resolution}...")
        
        start_time = time.time()
        
        # 专注于可能可行的区域
        sensor_indices = np.linspace(0.2, 1.0, 5)  # 覆盖所有传感器
        algo_indices = np.linspace(0, 1, 5)        # 所有算法
        deployment_values = [0.0, 0.5, 1.0]        # 所有部署选项
        cycle_values = np.linspace(0.05, 0.3, 5)  # 更多周期选项
        
        solution_id = 0
        
        for sensor_idx in sensor_indices:
            for algo_idx in algo_indices:
                for deployment in deployment_values:
                    for cycle in cycle_values:
                        solution_id += 1
                        
                        # 创建解决方案向量
                        x = np.zeros(11)
                        x[0] = sensor_idx
                        x[1] = 0.5  # 中等数据率
                        x[2] = 0.5  # Meso LOD
                        x[3] = 0.5  # Meso LOD
                        x[4] = algo_idx
                        x[5] = 0.7  # 高阈值
                        x[6] = 0.0  # 云存储（最便宜）
                        x[7] = 0.5  # 中等通信
                        x[8] = deployment
                        x[9] = 0.3  # 3人团队
                        x[10] = cycle
                        
                        # 评估
                        objectives, constraints = self.evaluator._evaluate_single(x)
                        
                        # 存储结果
                        self.results.append(
                            self._create_result_entry(x, objectives, constraints, solution_id)
                        )
        
        self.execution_time = time.time() - start_time
        
        # 更新时间
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Grid Search completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Evaluated {len(df)} grid points")
        logger.info(f"  Found {df['is_feasible'].sum()} feasible solutions")
        
        return df


class WeightedSumBaseline(BaselineMethod):
    """改进的加权和优化"""
    
    def optimize(self) -> pd.DataFrame:
        """使用不同权重组合进行优化"""
        n_weights = self.config.weight_combinations
        logger.info(f"Running Weighted Sum with {n_weights} weight sets...")
        
        start_time = time.time()
        
        # 生成权重组合
        weight_sets = self._generate_weight_sets(n_weights)
        
        for weight_idx, weights in enumerate(weight_sets):
            # 为每个权重组合寻找最佳解决方案
            best_score = float('inf')
            best_solution = None
            best_objectives = None
            best_constraints = None
            
            # 使用多个随机起点
            for start in range(20):
                x = self._generate_smart_initial_point()
                
                # 局部搜索
                for iteration in range(50):
                    objectives, constraints = self.evaluator._evaluate_single(x)
                    
                    # 只考虑可行解
                    if np.all(constraints <= 0):
                        # 归一化并计算加权和
                        norm_obj = self._normalize_objectives(objectives)
                        score = np.dot(weights, norm_obj)
                        
                        if score < best_score:
                            best_score = score
                            best_solution = x.copy()
                            best_objectives = objectives
                            best_constraints = constraints
                    
                    # 梯度下降式的改进
                    x = self._local_search_step(x, weights)
                        # 即使没有找到可行解也要记录
            if best_solution is None:
                # 记录失败的尝试
                x = self._generate_smart_initial_point()
                objectives, constraints = self.evaluator._evaluate_single(x)
                result = self._create_result_entry(x, objectives, constraints, weight_idx + 1)
                result['weights'] = weights.tolist()
                result['optimization_failed'] = True
                self.results.append(result)
            # 存储这个权重组合的最佳解决方案
            if best_solution is not None:
                result = self._create_result_entry(
                    best_solution, best_objectives, best_constraints, weight_idx + 1
                )
                result['weights'] = weights.tolist()
                self.results.append(result)
            
            # 进度日志
            if (weight_idx + 1) % 10 == 0:
                logger.info(f"  Evaluated {weight_idx + 1} weight combinations...")
        
        self.execution_time = time.time() - start_time
        
        # 更新时间
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Weighted Sum completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Found {len(df)} solutions")
        
        return df
    
    def _generate_weight_sets(self, n_sets: int) -> np.ndarray:
        """生成多样化的权重组合"""
        weights = []
        
        # 均匀权重
        weights.append(np.ones(6) / 6)
        
        # 单目标焦点
        for i in range(6):
            w = np.zeros(6)
            w[i] = 1.0
            weights.append(w)
        
        # 平衡的配对
        for i in range(6):
            for j in range(i+1, 6):
                w = np.zeros(6)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
                if len(weights) >= n_sets:
                    break
            if len(weights) >= n_sets:
                break
        
        # 剩余的随机权重
        while len(weights) < n_sets:
            w = np.random.rand(6)
            w = w / w.sum()  # 归一化
            weights.append(w)
        
        return np.array(weights[:n_sets])
    
    def _generate_smart_initial_point(self) -> np.ndarray:
        """生成智能初始点"""
        x = np.zeros(11)
        x[0] = np.random.uniform(0.7, 1.0)  # 便宜的传感器
        x[1] = 0.5
        x[2] = 0.5
        x[3] = 0.5
        x[4] = np.random.random()
        x[5] = 0.7
        x[6] = 0.0
        x[7] = 0.5
        x[8] = np.random.choice([0.5, 1.0])
        x[9] = 0.3
        x[10] = np.random.uniform(0.08, 0.2)
        return x
    
    def _local_search_step(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """局部搜索步骤"""
        # 简单的随机扰动
        x_new = x.copy()
        for i in range(len(x)):
            if np.random.random() < 0.2:  # 20%的变量被修改
                if i in [2, 3]:  # LOD
                    x_new[i] = np.random.choice([0.33, 0.5, 0.67])
                else:
                    x_new[i] = np.clip(x[i] + np.random.normal(0, 0.1), 0, 1)
        return x_new
    
    def _normalize_objectives(self, obj: np.ndarray) -> np.ndarray:
        """归一化目标到[0,1]"""
        # 基于问题知识的近似范围
        ranges = [
            (1e5, 2e7),      # 成本
            (0, 0.3),        # 1-召回率
            (0.1, 200),      # 延迟
            (0, 200),        # 干扰
            (100, 50000),    # 碳排放
            (1e-6, 1e-3)     # 1/MTBF
        ]
        
        norm = np.zeros_like(obj)
        for i, (min_val, max_val) in enumerate(ranges):
            norm[i] = (obj[i] - min_val) / (max_val - min_val + 1e-10)
        
        return np.clip(norm, 0, 1)


class ExpertHeuristicBaseline(BaselineMethod):
    """基于专家知识的配置"""
    
    def optimize(self) -> pd.DataFrame:
        """生成专家推荐的配置"""
        logger.info("Running Expert Heuristic baseline...")
        
        start_time = time.time()
        
        # 定义改进的专家配置，着重可行性
        expert_configs = [
            ("LowCost", self._low_cost_config()),
            ("Balanced", self._balanced_config()),
            ("QuickDeploy", self._quick_deploy_config()),
            ("Urban", self._urban_config()),
            ("Rural", self._rural_config()),
            ("Sustainable", self._sustainable_config()),
            ("Reliable", self._reliable_config()),
            ("Emergency", self._emergency_config()),
            ("Research", self._research_config()),
            ("Practical", self._practical_config())
        ]
        
        # 为每个基础配置添加变体
        base_configs = expert_configs.copy()
        for name, config in base_configs:
            # 创建两个轻微变体
            for i in range(2):
                varied = self._add_variation(config, 0.1)
                expert_configs.append((f"{name}_Var{i+1}", varied))
        
        for idx, (name, x) in enumerate(expert_configs):
            # 评估
            objectives, constraints = self.evaluator._evaluate_single(x)
            
            # 存储结果
            result = self._create_result_entry(x, objectives, constraints, idx + 1)
            result['method'] = f"ExpertHeuristic-{name}"
            self.results.append(result)
        
        self.execution_time = time.time() - start_time
        
        # 更新时间
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Expert Heuristic completed in {self.execution_time:.2f} seconds")
        logger.info(f"  Generated {len(df)} expert configurations")
        logger.info(f"  {df['is_feasible'].sum()} are feasible")
        
        return df
    
    def _low_cost_config(self) -> np.ndarray:
        """最低成本配置 - 改进版"""
        return np.array([
            1.0,    # IoT传感器（最便宜）
            0.2,    # 低数据率
            0.5,    # Meso LOD（平衡）
            0.5,    # Meso LOD
            0.8,    # 传统算法（便宜）
            0.6,    # 中等阈值
            0.0,    # 云存储
            0.4,    # LoRaWAN
            1.0,    # 云部署
            0.2,    # 2人团队
            0.15    # 55天周期
        ])
    
    def _balanced_config(self) -> np.ndarray:
        """平衡配置 - 改进版"""
        return np.array([
            0.85,   # 车载传感器
            0.5,    # 中等设置
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.5,    # ML算法
            0.65,   # 较好的阈值
            0.0,    # 云存储
            0.5,    # 4G
            0.8,    # 混合部署
            0.3,    # 3人团队
            0.1     # 36天
        ])
    
    def _quick_deploy_config(self) -> np.ndarray:
        """快速部署配置"""
        return np.array([
            0.9,    # 车载传感器（易部署）
            0.4,    # 中等数据率
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.7,    # 简单ML算法
            0.6,    # 中等阈值
            0.0,    # 云存储
            0.6,    # 4G（现有基础设施）
            1.0,    # 云部署
            0.3,    # 3人团队
            0.12    # 45天
        ])
    
    def _urban_config(self) -> np.ndarray:
        """城市配置"""
        return np.array([
            0.88,   # 车载传感器
            0.5,    # 中等数据率
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.4,    # ML算法
            0.7,    # 高阈值
            0.0,    # 云存储
            0.7,    # 接近5G
            0.8,    # 混合部署
            0.35,   # 3-4人团队
            0.08    # 30天
        ])
    
    def _rural_config(self) -> np.ndarray:
        """农村配置"""
        return np.array([
            0.95,   # 接近IoT（覆盖范围）
            0.3,    # 较低数据率
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.6,    # 传统/ML混合
            0.6,    # 中等阈值
            0.0,    # 云存储
            0.4,    # LoRaWAN
            1.0,    # 云部署
            0.2,    # 2人团队
            0.2     # 73天
        ])
    
    def _sustainable_config(self) -> np.ndarray:
        """可持续配置"""
        return np.array([
            0.95,   # 低功耗IoT
            0.15,   # 很低的数据率
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.8,    # 高效传统算法
            0.6,    # 中等阈值
            0.0,    # 云（共享资源）
            0.3,    # LoRaWAN（低功耗）
            1.0,    # 云部署
            0.15,   # 小团队
            0.25    # 90天
        ])
    
    def _reliable_config(self) -> np.ndarray:
        """可靠配置 - 改进版"""
        return np.array([
            0.4,    # 更好的传感器
            0.4,    # 中等数据率
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.4,    # ML算法
            0.7,    # 高阈值
            0.5,    # 混合存储
            0.6,    # 4G
            0.5,    # 混合部署
            0.3,    # 3人团队
            0.1     # 36天
        ])
    
    def _emergency_config(self) -> np.ndarray:
        """应急配置"""
        return np.array([
            0.5,    # UAV快速部署
            0.6,    # 较高数据率
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.3,    # 快速ML算法
            0.5,    # 较低阈值（安全）
            0.0,    # 云存储
            0.7,    # 接近5G
            0.2,    # 边缘（低延迟）
            0.4,    # 4人团队
            0.01    # 每日检查
        ])
    
    def _research_config(self) -> np.ndarray:
        """研究级配置 - 调整为更实际"""
        return np.array([
            0.3,    # 较好的传感器
            0.7,    # 高数据率
            0.33,   # 接近Micro LOD
            0.33,   # 接近Micro LOD
            0.2,    # DL算法
            0.8,    # 高阈值
            0.5,    # 混合存储
            0.8,    # 接近光纤
            0.3,    # 边缘计算
            0.5,    # 5人团队
            0.06    # 21天
        ])
    
    def _practical_config(self) -> np.ndarray:
        """实用配置"""
        return np.array([
            0.88,   # 车载传感器
            0.45,   # 中等一切
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.5,    # ML算法
            0.65,   # 好的阈值
            0.0,    # 云存储
            0.6,    # 4G
            0.8,    # 主要是云
            0.35,   # 3-4人团队
            0.11    # 40天
        ])
    
    def _add_variation(self, config: np.ndarray, variation_level: float) -> np.ndarray:
        """添加小变化"""
        varied = config.copy()
        
        for i in range(len(varied)):
            if i in [2, 3]:  # LOD - 不变
                continue
            
            # 添加高斯噪声
            noise = np.random.normal(0, variation_level)
            varied[i] = np.clip(varied[i] + noise, 0, 1)
        
        return varied  

class BaselineRunner:
    """Orchestrates all baseline methods"""
    
    def __init__(self, ontology_graph, config):
        self.ontology_graph = ontology_graph
        self.config = config
        
        # Import here to avoid circular import
        from evaluation import EnhancedFitnessEvaluatorV3  # 改为 V3
        
        # Initialize shared evaluator
        self.evaluator = EnhancedFitnessEvaluatorV3(ontology_graph, config)  # 改为 V3
        
        # Initialize baseline methods
        self.methods = {
            'random': RandomSearchBaseline(self.evaluator, config),
            'grid': GridSearchBaseline(self.evaluator, config),
            'weighted': WeightedSumBaseline(self.evaluator, config),
            'expert': ExpertHeuristicBaseline(self.evaluator, config)
        }
    
    def run_all_methods(self) -> Dict[str, pd.DataFrame]:
        """Run all baseline methods"""
        results = {}
        
        for name, method in self.methods.items():
            logger.info(f"\nRunning baseline method: {name}")
            try:
                df = method.optimize()
                results[name] = df
                
                # Log summary
                logger.info(f"  Total solutions: {len(df)}")
                logger.info(f"  Feasible solutions: {df['is_feasible'].sum()}")
                if df['is_feasible'].sum() > 0:
                    feasible = df[df['is_feasible']]
                    logger.info(f"  Best cost: ${feasible['f1_total_cost_USD'].min():,.0f}")
                    logger.info(f"  Best recall: {feasible['detection_recall'].max():.3f}")
                    logger.info(f"  Best carbon: {feasible['f5_carbon_emissions_kgCO2e_year'].min():.0f} kgCO2e/year")
                
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                results[name] = pd.DataFrame()
        
        return results
    
    def run_method(self, method_name: str) -> pd.DataFrame:
        """Run a specific baseline method"""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}. "
                           f"Available: {list(self.methods.keys())}")
        
        return self.methods[method_name].optimize()
    
    def compare_with_pareto(self, pareto_df: pd.DataFrame, 
                           baseline_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare baseline results with Pareto front"""
        comparison = []
        
        # Analyze Pareto results
        comparison.append({
            'Method': 'NSGA-II',
            'Total_Solutions': len(pareto_df),
            'Feasible_Solutions': len(pareto_df),  # All Pareto solutions are feasible
            'Min_Cost': pareto_df['f1_total_cost_USD'].min(),
            'Max_Recall': pareto_df['detection_recall'].max(),
            'Min_Carbon': pareto_df['f5_carbon_emissions_kgCO2e_year'].min(),
            'Min_Latency': pareto_df['f3_latency_seconds'].min(),
            'Max_MTBF': pareto_df['system_MTBF_hours'].max()
        })
        
        # Analyze baseline results
        for method, df in baseline_results.items():
            if len(df) > 0:
                feasible = df[df['is_feasible']]
                comparison.append({
                    'Method': method.title(),
                    'Total_Solutions': len(df),
                    'Feasible_Solutions': len(feasible),
                    'Min_Cost': feasible['f1_total_cost_USD'].min() if len(feasible) > 0 else np.nan,
                    'Max_Recall': feasible['detection_recall'].max() if len(feasible) > 0 else np.nan,
                    'Min_Carbon': feasible['f5_carbon_emissions_kgCO2e_year'].min() if len(feasible) > 0 else np.nan,
                    'Min_Latency': feasible['f3_latency_seconds'].min() if len(feasible) > 0 else np.nan,
                    'Max_MTBF': feasible['system_MTBF_hours'].max() if len(feasible) > 0 else np.nan
                })
        
        return pd.DataFrame(comparison)
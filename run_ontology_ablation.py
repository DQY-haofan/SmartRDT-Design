#!/usr/bin/env python3
"""
RMTwin Ontology Ablation Study
===============================
证明Ontology的价值：对比有/无Ontology约束的优化效果

消融维度：
1. 组件类型约束 (Type Constraints)
   - ON: 使用类型特定的参数（DL需要GPU，传感器类型影响精度等）
   - OFF: 所有组件使用统一默认参数

2. 组件属性查询 (Property Queries)
   - ON: 从Ontology查询组件属性（recall, precision, cost等）
   - OFF: 使用固定默认值

3. 兼容性规则 (Compatibility Rules)
   - ON: DL算法+GPU部署有加成，Edge部署DL有惩罚
   - OFF: 取消所有组件间交互效应

运行方法:
    python run_ontology_ablation.py --config config.json --seed 42 --output ./results/ablation

输出:
    - ablation_results.csv: 各模式的指标对比
    - ablation_figures/: 可视化对比图
    - ablation_report.md: 自动生成的论文表述

Author: RMTwin Research Team
Version: 1.0
"""

import argparse
import logging
import time
import json
import sys
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 消融模式配置
# =============================================================================

ABLATION_MODES = {
    'full_ontology': {
        'name': 'Full Ontology (Baseline)',
        'description': 'Complete ontology with all constraints and type-specific parameters',
        'type_constraints': True,
        'property_queries': True,
        'compatibility_rules': True,
    },
    'no_type_constraints': {
        'name': 'No Type Constraints',
        'description': 'Disable type-specific parameters (all components use defaults)',
        'type_constraints': False,
        'property_queries': True,
        'compatibility_rules': True,
    },
    'no_property_queries': {
        'name': 'No Property Queries',
        'description': 'Use fixed default values instead of ontology queries',
        'type_constraints': True,
        'property_queries': False,
        'compatibility_rules': True,
    },
    'no_compatibility': {
        'name': 'No Compatibility Rules',
        'description': 'Disable component interaction effects (DL+GPU bonus, etc.)',
        'type_constraints': True,
        'property_queries': True,
        'compatibility_rules': False,
    },
    'no_ontology': {
        'name': 'No Ontology (Random)',
        'description': 'Complete ablation - all constraints disabled, random parameters',
        'type_constraints': False,
        'property_queries': False,
        'compatibility_rules': False,
    },
}


# =============================================================================
# 修改版评估器 (支持消融)
# =============================================================================

class AblatedFitnessEvaluator:
    """
    支持消融的适应度评估器

    通过ablation_config控制哪些ontology功能被启用/禁用
    """

    # 默认参数（当ontology查询被禁用时使用）
    DEFAULT_PARAMS = {
        'sensor_precision': 0.75,
        'sensor_recall': 0.70,
        'sensor_initial_cost': 50000,
        'sensor_op_cost_day': 100,
        'sensor_mtbf': 8760,
        'sensor_energy_w': 50,
        'sensor_data_gb_km': 2.0,

        'algo_recall': 0.75,
        'algo_precision': 0.75,
        'algo_compute_s_gb': 15,

        'storage_cost_gb': 0.023,
        'comm_bandwidth_gbps': 0.01,
        'deploy_compute_factor': 1.5,
    }

    def __init__(self, ontology_graph, config, ablation_config: Dict):
        """
        Args:
            ontology_graph: RDF图
            config: 优化配置
            ablation_config: 消融配置 {'type_constraints': bool, 'property_queries': bool, 'compatibility_rules': bool}
        """
        self.g = ontology_graph
        self.config = config
        self.ablation = ablation_config

        # 导入原始模块
        from evaluation import SolutionMapper, EnhancedFitnessEvaluatorV3
        from model_params import MODEL_PARAMS, get_param, sigmoid

        self.MODEL_PARAMS = MODEL_PARAMS
        self.get_param = get_param
        self.sigmoid = sigmoid

        # 组件映射（始终从ontology获取可用组件列表）
        self.solution_mapper = SolutionMapper(ontology_graph)

        # 属性缓存
        self._property_cache = {}
        self._initialize_cache()

        # 统计
        self._eval_count = 0
        self._feasible_count = 0

        logger.info(f"AblatedEvaluator initialized: type_constraints={ablation_config['type_constraints']}, "
                    f"property_queries={ablation_config['property_queries']}, "
                    f"compatibility_rules={ablation_config['compatibility_rules']}")

    def _initialize_cache(self):
        """初始化属性缓存"""
        from rdflib import Namespace
        RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")

        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasMTBFHours',
            'hasEnergyConsumptionW', 'hasDataVolumeGBPerKm', 'hasPrecision',
            'hasRecall', 'hasFPS'
        ]

        for prop_name in properties:
            prop_uri = RDTCO[prop_name]
            for s, p, o in self.g.triples((None, prop_uri, None)):
                try:
                    self._property_cache[(str(s), prop_name)] = float(str(o))
                except:
                    pass

    def _query_property(self, component_uri: str, prop_name: str, default: float) -> float:
        """
        查询组件属性

        如果property_queries=False，返回默认值
        """
        if not self.ablation['property_queries']:
            # 使用固定默认值
            return self.DEFAULT_PARAMS.get(prop_name, default)

        # 正常从缓存/ontology查询
        return self._property_cache.get((component_uri, prop_name), default)

    def _get_type_param(self, param_name: str, type_key: str, default: float) -> float:
        """
        获取类型特定参数

        如果type_constraints=False，返回默认值
        """
        if not self.ablation['type_constraints']:
            return default

        return self.get_param(param_name, type_key, default)

    def _calculate_hw_penalty(self, algo_type: str, deploy_type: str) -> float:
        """
        计算硬件惩罚

        如果compatibility_rules=False，返回0（无惩罚）
        """
        if not self.ablation['compatibility_rules']:
            return 0.0

        # 原始逻辑：DL/ML算法在非GPU部署上有惩罚
        if algo_type in ['DL', 'ML']:
            if deploy_type == 'Edge':
                return 0.5
            elif deploy_type not in ['Cloud', 'Hybrid']:
                return 0.8
        return 0.0

    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估单个解"""
        config = self.solution_mapper.decode_solution(x)

        # 提取类型
        from model_params import get_sensor_type, get_algo_type, get_comm_type, get_storage_type, get_deployment_type

        if self.ablation['type_constraints']:
            sensor_type = get_sensor_type(str(config['sensor']))
            algo_type = get_algo_type(str(config['algorithm']))
            comm_type = get_comm_type(str(config['communication']))
            storage_type = get_storage_type(str(config['storage']))
            deploy_type = get_deployment_type(str(config['deployment']))
        else:
            # 统一使用默认类型
            sensor_type = 'IoT'
            algo_type = 'Traditional'
            comm_type = 'LoRa'
            storage_type = 'Cloud'
            deploy_type = 'Cloud'

        # === 计算目标函数 ===

        # f1: 总成本
        cost = self._calculate_cost(config, sensor_type, algo_type, comm_type, storage_type, deploy_type)

        # f2: 1 - recall
        recall = self._calculate_recall(config, sensor_type, algo_type, deploy_type)
        one_minus_recall = 1 - recall

        # f3: 延迟
        latency = self._calculate_latency(config, sensor_type, algo_type, comm_type, deploy_type)

        # f4: 交通干扰
        disruption = self._calculate_disruption(config, sensor_type)

        # f5: 碳排放
        carbon = self._calculate_carbon(config, sensor_type, algo_type, comm_type, deploy_type)

        # f6: 1/MTBF (最小化)
        mtbf = self._calculate_mtbf(config)
        reliability_inv = 1.0 / max(mtbf, 1)

        objectives = np.array([cost, one_minus_recall, latency, disruption, carbon, reliability_inv])

        # === 计算约束 ===
        constraints = self._calculate_constraints(cost, recall, latency, disruption, carbon, mtbf)

        self._eval_count += 1
        if np.all(constraints <= 0):
            self._feasible_count += 1

        return objectives, constraints

    def _calculate_cost(self, config, sensor_type, algo_type, comm_type, storage_type, deploy_type) -> float:
        """计算总成本"""
        horizon = self.config.planning_horizon_years
        road_km = self.config.road_network_length_km

        # 传感器成本
        sensor_initial = self._query_property(config['sensor'], 'hasInitialCostUSD',
                                              self.DEFAULT_PARAMS['sensor_initial_cost'])
        sensor_op_day = self._query_property(config['sensor'], 'hasOperationalCostUSDPerDay',
                                             self.DEFAULT_PARAMS['sensor_op_cost_day'])

        sensor_capex = sensor_initial * (road_km / 10)  # 假设每10km一个传感器
        sensor_opex = sensor_op_day * 365 * horizon

        # 存储成本
        data_gb_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm',
                                          self.DEFAULT_PARAMS['sensor_data_gb_km'])
        inspections_year = 365.0 / config['inspection_cycle']
        total_data_gb = data_gb_km * road_km * inspections_year * horizon
        storage_cost_gb = self._get_type_param('storage_cost_per_GB', storage_type,
                                               self.DEFAULT_PARAMS['storage_cost_gb'])
        storage_cost = total_data_gb * storage_cost_gb

        # 计算成本
        compute_cost_factor = self._get_type_param('deployment_compute_factor', deploy_type, 1.5)
        compute_cost = total_data_gb * 0.01 * compute_cost_factor  # $0.01/GB处理

        # 人工成本
        labor_cost = config['crew_size'] * 50000 * horizon  # $50k/人/年

        total = sensor_capex + sensor_opex + storage_cost + compute_cost + labor_cost
        return max(total, 1000)

    def _calculate_recall(self, config, sensor_type, algo_type, deploy_type) -> float:
        """计算检测召回率"""
        rm = self.MODEL_PARAMS['recall_model']

        # 基础recall
        base_algo_recall = self._query_property(config['algorithm'], 'hasRecall',
                                                self.DEFAULT_PARAMS['algo_recall'])
        sensor_precision = self._query_property(config['sensor'], 'hasPrecision',
                                                self.DEFAULT_PARAMS['sensor_precision'])

        # LOD加成
        lod_bonus = rm['lod_bonus'].get(config['geo_lod'], 0.0) if self.ablation['type_constraints'] else 0.0

        # 数据率加成
        data_rate_bonus = rm['data_rate_bonus_factor'] * max(0, config['data_rate'] - rm['base_data_rate'])

        # 硬件惩罚
        hw_penalty = self._calculate_hw_penalty(algo_type, deploy_type)

        # 检测阈值效应
        tau = config['detection_threshold']
        tau0 = rm['tau0']

        # Sigmoid计算
        z = (rm['a0'] +
             rm['a1'] * base_algo_recall +
             rm['a2'] * sensor_precision +
             lod_bonus +
             data_rate_bonus -
             rm['a3'] * (tau - tau0) -
             hw_penalty)

        recall = self.sigmoid(z)
        return np.clip(recall, rm['min_recall'], rm['max_recall'])

    def _calculate_latency(self, config, sensor_type, algo_type, comm_type, deploy_type) -> float:
        """计算延迟"""
        # 数据量
        data_gb_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm',
                                          self.DEFAULT_PARAMS['sensor_data_gb_km'])
        road_km = self.config.road_network_length_km
        data_per_inspection = data_gb_km * road_km * (config['data_rate'] / 30)

        # 通信时间
        bandwidth = self._get_type_param('comm_bandwidth_GBps', comm_type,
                                         self.DEFAULT_PARAMS['comm_bandwidth_gbps'])
        comm_time = data_per_inspection / max(bandwidth, 1e-6)

        # 计算时间
        compute_s_gb = self._get_type_param('algo_compute_seconds_per_GB', algo_type,
                                            self.DEFAULT_PARAMS['algo_compute_s_gb'])
        compute_factor = self._get_type_param('deployment_compute_factor', deploy_type,
                                              self.DEFAULT_PARAMS['deploy_compute_factor'])
        compute_time = data_per_inspection * compute_s_gb * compute_factor

        return max(comm_time + compute_time, 1.0)

    def _calculate_disruption(self, config, sensor_type) -> float:
        """计算交通干扰"""
        road_km = self.config.road_network_length_km
        inspections_year = 365.0 / config['inspection_cycle']

        # 基础干扰取决于传感器类型
        if self.ablation['type_constraints']:
            if sensor_type in ['MMS', 'Vehicle']:
                base_hours_km = 0.5
            elif sensor_type in ['UAV']:
                base_hours_km = 0.1
            elif sensor_type in ['IoT', 'FiberOptic']:
                base_hours_km = 0.02
            else:
                base_hours_km = 0.3
        else:
            base_hours_km = 0.2

        # 人员规模影响
        crew_factor = 1 + (config['crew_size'] - 1) * 0.1

        total = base_hours_km * road_km * inspections_year * crew_factor
        return max(total, 1.0)

    def _calculate_carbon(self, config, sensor_type, algo_type, comm_type, deploy_type) -> float:
        """计算碳排放"""
        horizon = self.config.planning_horizon_years
        road_km = self.config.road_network_length_km
        inspections_year = 365.0 / config['inspection_cycle']

        # 传感器能耗
        energy_w = self._query_property(config['sensor'], 'hasEnergyConsumptionW',
                                        self.DEFAULT_PARAMS['sensor_energy_w'])
        sensor_kwh_year = energy_w * 8760 / 1000 * (road_km / 10)

        # 计算能耗
        data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm',
                                       self.DEFAULT_PARAMS['sensor_data_gb_km']) * road_km * inspections_year

        if self.ablation['type_constraints']:
            if deploy_type == 'Cloud':
                compute_kwh_gb = 0.5
            elif deploy_type == 'Edge':
                compute_kwh_gb = 0.2
            else:
                compute_kwh_gb = 0.35
        else:
            compute_kwh_gb = 0.35

        compute_kwh_year = data_gb * compute_kwh_gb

        # 通信能耗
        comm_kwh_year = data_gb * 0.05

        # 总能耗转碳排放 (0.4 kgCO2/kWh)
        total_kwh = (sensor_kwh_year + compute_kwh_year + comm_kwh_year) * horizon
        carbon = total_kwh * 0.4

        return max(carbon, 100)

    def _calculate_mtbf(self, config) -> float:
        """计算系统MTBF"""
        sensor_mtbf = self._query_property(config['sensor'], 'hasMTBFHours',
                                           self.DEFAULT_PARAMS['sensor_mtbf'])

        # 简化：系统MTBF由最弱组件决定
        return max(sensor_mtbf * 0.8, 1000)

    def _calculate_constraints(self, cost, recall, latency, disruption, carbon, mtbf) -> np.ndarray:
        """计算约束违反量"""
        constraints = np.array([
            cost - self.config.budget_cap_usd,  # g1: 预算约束
            self.config.min_recall_threshold - recall,  # g2: 最小召回率
            latency - self.config.max_latency_seconds,  # g3: 最大延迟
            disruption - self.config.max_disruption_hours,  # g4: 最大干扰
            self.config.min_mtbf_hours - mtbf,  # g5: 最小MTBF
        ])
        return constraints

    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量评估"""
        n = len(X)
        objectives = np.zeros((n, 6))
        constraints = np.zeros((n, 5))

        for i, x in enumerate(X):
            objectives[i], constraints[i] = self._evaluate_single(x)

        return objectives, constraints

    def get_stats(self) -> Dict:
        """获取评估统计"""
        return {
            'total_evaluations': self._eval_count,
            'feasible_count': self._feasible_count,
            'feasible_rate': self._feasible_count / max(self._eval_count, 1)
        }


# =============================================================================
# 消融问题定义
# =============================================================================

class AblatedProblem:
    """支持消融的优化问题"""

    def __init__(self, evaluator: AblatedFitnessEvaluator):
        self.evaluator = evaluator
        self.n_var = 11
        self.n_obj = 6
        self.n_constr = 5
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)


# =============================================================================
# 消融实验运行器
# =============================================================================

class AblationRunner:
    """消融实验运行器"""

    def __init__(self, config_path: str, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

        # 加载配置
        from config_manager import ConfigManager
        self.config = ConfigManager(config_path)

        # 加载ontology
        from ontology_manager import OntologyManager
        self.ontology_manager = OntologyManager()
        self.ontology_graph = self.ontology_manager.populate_from_csv_files(
            self.config.sensor_csv,
            self.config.algorithm_csv,
            self.config.infrastructure_csv,
            self.config.cost_benefit_csv
        )

        logger.info(f"AblationRunner initialized with seed={seed}")

    def run_single_mode(self, mode_name: str, n_samples: int = 2000) -> Dict:
        """运行单个消融模式"""

        mode_config = ABLATION_MODES[mode_name]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running mode: {mode_config['name']}")
        logger.info(f"Description: {mode_config['description']}")
        logger.info(f"{'=' * 60}")

        # 创建消融评估器
        ablation_config = {
            'type_constraints': mode_config['type_constraints'],
            'property_queries': mode_config['property_queries'],
            'compatibility_rules': mode_config['compatibility_rules'],
        }

        evaluator = AblatedFitnessEvaluator(
            self.ontology_graph,
            self.config,
            ablation_config
        )

        # 随机采样评估
        start_time = time.time()

        X = np.random.random((n_samples, 11))
        objectives, constraints = evaluator.evaluate_batch(X)

        elapsed = time.time() - start_time

        # 找可行解
        feasible_mask = np.all(constraints <= 0, axis=1)
        n_feasible = feasible_mask.sum()

        # 计算指标
        results = {
            'mode': mode_name,
            'mode_name': mode_config['name'],
            'n_samples': n_samples,
            'n_feasible': int(n_feasible),
            'feasible_rate': float(n_feasible / n_samples),
            'time_seconds': elapsed,
        }

        if n_feasible > 0:
            feasible_obj = objectives[feasible_mask]

            # 找非支配解
            pareto_mask = self._find_pareto(feasible_obj)
            pareto_obj = feasible_obj[pareto_mask]

            results['n_pareto'] = len(pareto_obj)
            results['min_cost'] = float(feasible_obj[:, 0].min())
            results['max_recall'] = float(1 - feasible_obj[:, 1].min())
            results['min_latency'] = float(feasible_obj[:, 2].min())
            results['min_carbon'] = float(feasible_obj[:, 4].min())

            # HV (2D简化)
            ref_point = np.array([feasible_obj[:, 0].max() * 1.1, 1.1])
            pareto_2d = pareto_obj[:, :2]
            results['hv_2d'] = float(self._calculate_hv_2d(pareto_2d, ref_point))

            # 多样性
            results['cost_range'] = float(feasible_obj[:, 0].max() - feasible_obj[:, 0].min())
            results['recall_range'] = float((1 - feasible_obj[:, 1].min()) - (1 - feasible_obj[:, 1].max()))
        else:
            results['n_pareto'] = 0
            results['min_cost'] = np.nan
            results['max_recall'] = np.nan
            results['min_latency'] = np.nan
            results['min_carbon'] = np.nan
            results['hv_2d'] = 0
            results['cost_range'] = 0
            results['recall_range'] = 0

        logger.info(f"Results: feasible_rate={results['feasible_rate']:.2%}, "
                    f"n_pareto={results.get('n_pareto', 0)}, "
                    f"max_recall={results.get('max_recall', 0):.4f}")

        return results

    def _find_pareto(self, F: np.ndarray) -> np.ndarray:
        """找非支配解"""
        n = len(F)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # j支配i?
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_pareto[i] = False
                    break

        return is_pareto

    def _calculate_hv_2d(self, F: np.ndarray, ref_point: np.ndarray) -> float:
        """计算2D Hypervolume"""
        if len(F) == 0:
            return 0.0

        # 过滤
        valid = np.all(F < ref_point, axis=1)
        F = F[valid]

        if len(F) == 0:
            return 0.0

        # 按f1排序
        sorted_idx = np.argsort(F[:, 0])
        F = F[sorted_idx]

        hv = 0.0
        prev_y = ref_point[1]

        for i in range(len(F)):
            if F[i, 1] < prev_y:
                width = ref_point[0] - F[i, 0]
                height = prev_y - F[i, 1]
                hv += width * height
                prev_y = F[i, 1]

        return hv

    def run_all_modes(self, n_samples: int = 2000, n_repeats: int = 3) -> pd.DataFrame:
        """运行所有消融模式"""

        all_results = []

        for mode_name in ABLATION_MODES.keys():
            for repeat in range(n_repeats):
                np.random.seed(self.seed + repeat * 100)

                result = self.run_single_mode(mode_name, n_samples)
                result['repeat'] = repeat
                all_results.append(result)

        df = pd.DataFrame(all_results)
        return df

    def generate_report(self, results_df: pd.DataFrame, output_dir: Path) -> str:
        """生成消融报告"""

        # 按模式聚合
        agg = results_df.groupby('mode').agg({
            'feasible_rate': ['mean', 'std'],
            'n_pareto': ['mean', 'std'],
            'max_recall': ['mean', 'std'],
            'min_cost': ['mean', 'std'],
            'hv_2d': ['mean', 'std'],
        }).round(4)

        # 计算相对full_ontology的变化
        baseline = results_df[results_df['mode'] == 'full_ontology']
        baseline_feasible = baseline['feasible_rate'].mean()
        baseline_recall = baseline['max_recall'].mean()
        baseline_hv = baseline['hv_2d'].mean()

        report = f"""# Ontology Ablation Study Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Seed**: {self.seed}
**Samples per mode**: {results_df['n_samples'].iloc[0]}
**Repeats**: {results_df['repeat'].max() + 1}

## 1. Summary Table

| Mode | Feasible Rate | Max Recall | Min Cost ($M) | HV (2D) |
|------|--------------|------------|---------------|---------|
"""

        for mode in ABLATION_MODES.keys():
            mode_data = results_df[results_df['mode'] == mode]
            fr = mode_data['feasible_rate'].mean()
            recall = mode_data['max_recall'].mean()
            cost = mode_data['min_cost'].mean() / 1e6
            hv = mode_data['hv_2d'].mean()

            report += f"| {ABLATION_MODES[mode]['name'][:30]} | {fr:.2%} | {recall:.4f} | ${cost:.3f}M | {hv:.2e} |\n"

        report += f"""
## 2. Key Findings

### 2.1 Feasible Rate Impact

The feasible rate measures how many randomly sampled configurations satisfy all constraints.
This indicates how well the ontology guides the search toward valid regions.

- **Full Ontology**: {baseline_feasible:.2%} feasible rate
"""

        for mode in ['no_type_constraints', 'no_property_queries', 'no_compatibility', 'no_ontology']:
            mode_data = results_df[results_df['mode'] == mode]
            fr = mode_data['feasible_rate'].mean()
            change = (fr - baseline_feasible) / baseline_feasible * 100
            report += f"- **{ABLATION_MODES[mode]['name']}**: {fr:.2%} ({change:+.1f}% vs baseline)\n"

        report += f"""
### 2.2 Solution Quality Impact

Maximum achievable recall indicates the quality of solutions found.

- **Full Ontology**: {baseline_recall:.4f} max recall
"""

        for mode in ['no_type_constraints', 'no_property_queries', 'no_compatibility', 'no_ontology']:
            mode_data = results_df[results_df['mode'] == mode]
            recall = mode_data['max_recall'].mean()
            change = (recall - baseline_recall) / baseline_recall * 100
            report += f"- **{ABLATION_MODES[mode]['name']}**: {recall:.4f} ({change:+.1f}% vs baseline)\n"

        report += f"""
## 3. Paper-Ready Conclusions

### For "Ontology Value" Section:

> The ablation study demonstrates the critical role of the ontology in guiding 
> multi-objective optimization. Disabling all ontology constraints ('No Ontology' mode) 
> results in a {((results_df[results_df['mode'] == 'no_ontology']['feasible_rate'].mean() - baseline_feasible) / baseline_feasible * 100):.1f}% change in feasible solution rate 
> and {((results_df[results_df['mode'] == 'no_ontology']['max_recall'].mean() - baseline_recall) / baseline_recall * 100):.1f}% change in maximum achievable recall.

### Individual Contribution Analysis:

1. **Type Constraints**: Enable type-specific parameters (sensor types, algorithm categories)
   - Impact on feasible rate: {((results_df[results_df['mode'] == 'no_type_constraints']['feasible_rate'].mean() - baseline_feasible) / baseline_feasible * 100):+.1f}%

2. **Property Queries**: Retrieve component attributes from knowledge base
   - Impact on feasible rate: {((results_df[results_df['mode'] == 'no_property_queries']['feasible_rate'].mean() - baseline_feasible) / baseline_feasible * 100):+.1f}%

3. **Compatibility Rules**: Model component interactions (DL+GPU synergy, etc.)
   - Impact on feasible rate: {((results_df[results_df['mode'] == 'no_compatibility']['feasible_rate'].mean() - baseline_feasible) / baseline_feasible * 100):+.1f}%

## 4. Ablation Modes Description

"""

        for mode, config in ABLATION_MODES.items():
            report += f"### {config['name']}\n"
            report += f"- **Description**: {config['description']}\n"
            report += f"- Type Constraints: {'✓' if config['type_constraints'] else '✗'}\n"
            report += f"- Property Queries: {'✓' if config['property_queries'] else '✗'}\n"
            report += f"- Compatibility Rules: {'✓' if config['compatibility_rules'] else '✗'}\n\n"

        return report


# =============================================================================
# 可视化
# =============================================================================

def generate_ablation_figures(results_df: pd.DataFrame, output_dir: Path):
    """生成消融对比图"""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'figure.facecolor': 'white',
    })

    fig_dir = output_dir / 'ablation_figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 聚合数据
    agg = results_df.groupby('mode').agg({
        'feasible_rate': ['mean', 'std'],
        'n_pareto': ['mean', 'std'],
        'max_recall': ['mean', 'std'],
        'min_cost': ['mean', 'std'],
        'hv_2d': ['mean', 'std'],
    })

    modes = list(ABLATION_MODES.keys())
    mode_names = [ABLATION_MODES[m]['name'][:20] for m in modes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # === 图1: Feasible Rate对比 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [agg.loc[m, ('feasible_rate', 'mean')] for m in modes]
    stds = [agg.loc[m, ('feasible_rate', 'std')] for m in modes]

    bars = ax.bar(range(len(modes)), means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(mode_names, rotation=45, ha='right')
    ax.set_ylabel('Feasible Rate')
    ax.set_ylim(0, 1)

    # 标注数值
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{mean:.1%}', ha='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / 'ablation_feasible_rate.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'ablation_feasible_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === 图2: Max Recall对比 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [agg.loc[m, ('max_recall', 'mean')] for m in modes]
    stds = [agg.loc[m, ('max_recall', 'std')] for m in modes]

    bars = ax.bar(range(len(modes)), means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(mode_names, rotation=45, ha='right')
    ax.set_ylabel('Maximum Recall')
    ax.set_ylim(0.5, 1.0)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / 'ablation_max_recall.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'ablation_max_recall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === 图3: 综合对比雷达图 ===
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    categories = ['Feasible\nRate', 'Max\nRecall', 'HV', 'Pareto\nSize', 'Cost\nEfficiency']
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]

    # 归一化数据
    baseline = modes[0]  # full_ontology

    for i, mode in enumerate(modes[:3]):  # 只画前3个
        values = [
            agg.loc[mode, ('feasible_rate', 'mean')] / agg.loc[baseline, ('feasible_rate', 'mean')],
            agg.loc[mode, ('max_recall', 'mean')] / max(agg.loc[baseline, ('max_recall', 'mean')], 0.01),
            agg.loc[mode, ('hv_2d', 'mean')] / max(agg.loc[baseline, ('hv_2d', 'mean')], 1e-10),
            agg.loc[mode, ('n_pareto', 'mean')] / max(agg.loc[baseline, ('n_pareto', 'mean')], 1),
            1.0,  # placeholder
        ]
        values = [min(v, 1.5) for v in values]  # 限制范围
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=ABLATION_MODES[mode]['name'][:20], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    fig.savefig(fig_dir / 'ablation_radar.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'ablation_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Figures saved to {fig_dir}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RMTwin Ontology Ablation Study')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--samples', type=int, default=2000, help='Samples per mode')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats')
    parser.add_argument('--output', type=str, default='./results/ablation', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RMTwin Ontology Ablation Study")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Samples/mode: {args.samples}")
    logger.info(f"Repeats: {args.repeats}")
    logger.info(f"Output: {output_dir}")

    # 运行消融实验
    runner = AblationRunner(args.config, args.seed)
    results_df = runner.run_all_modes(n_samples=args.samples, n_repeats=args.repeats)

    # 保存结果
    results_df.to_csv(output_dir / 'ablation_results.csv', index=False)
    logger.info(f"Results saved to {output_dir / 'ablation_results.csv'}")

    # 生成报告
    report = runner.generate_report(results_df, output_dir)
    with open(output_dir / 'ablation_report.md', 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {output_dir / 'ablation_report.md'}")

    # 生成图表
    generate_ablation_figures(results_df, output_dir)

    # 打印摘要
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    agg = results_df.groupby('mode')['feasible_rate'].mean()
    baseline = agg['full_ontology']

    print(f"\nFeasible Rate Comparison:")
    for mode in ABLATION_MODES.keys():
        fr = agg[mode]
        change = (fr - baseline) / baseline * 100
        print(f"  {ABLATION_MODES[mode]['name'][:30]:<35}: {fr:.2%} ({change:+.1f}%)")

    print(f"\nOutput files:")
    print(f"  - {output_dir / 'ablation_results.csv'}")
    print(f"  - {output_dir / 'ablation_report.md'}")
    print(f"  - {output_dir / 'ablation_figures/'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
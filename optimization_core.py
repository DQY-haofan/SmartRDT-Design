#!/usr/bin/env python3
"""
核心优化模块 - NSGA-III实现（修复版）
针对6目标优化进行了调整
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

logger = logging.getLogger(__name__)


class RMTwinProblem(Problem):
    """6目标优化问题"""
    
    def __init__(self, evaluator):
        # 问题维度
        n_var = 11  # 决策变量
        n_obj = 6   # 目标
        n_constr = 5  # 约束
        
        # 变量边界
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )
        
        self.evaluator = evaluator
        self._eval_count = 0
        
    def _evaluate(self, X, out, *args, **kwargs):
        """评估种群"""
        objectives, constraints = self.evaluator.evaluate_batch(X)
        
        out["F"] = objectives
        out["G"] = constraints
        
        # 记录进度
        self._eval_count += len(X)
        if self._eval_count % 1000 == 0:
            logger.info(f"已评估 {self._eval_count} 个解决方案...")
            
            # 记录当前最佳值
            best_indices = {
                'cost': objectives[:, 0].argmin(),
                'recall': objectives[:, 1].argmin(),
                'latency': objectives[:, 2].argmin(),
                'disruption': objectives[:, 3].argmin(),
                'carbon': objectives[:, 4].argmin(),
                'reliability': objectives[:, 5].argmin()
            }
            
            for name, idx in best_indices.items():
                if idx is not None:
                    logger.debug(f"当前最佳 {name}: {objectives[idx]}")


class RMTwinOptimizer:
    """6目标优化的主协调器"""
    
    def __init__(self, ontology_graph, config):
        self.ontology_graph = ontology_graph
        self.config = config
        
        # 导入评估器
        from evaluation import EnhancedFitnessEvaluatorV3
        
        # 初始化评估器
        self.evaluator = EnhancedFitnessEvaluatorV3(ontology_graph, config)
        
        # 初始化问题
        self.problem = RMTwinProblem(self.evaluator)
        
        # 配置算法
        self.algorithm = self._configure_algorithm()
        
    def _configure_algorithm(self):
        """配置NSGA-II/III算法"""
        if self.config.n_objectives <= 3:
            # 对于3个或更少的目标，使用NSGA-II
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=self.config.crossover_eta, 
                             prob=self.config.crossover_prob),
                mutation=PM(eta=self.config.mutation_eta, 
                           prob=1.0/self.problem.n_var),
                eliminate_duplicates=True
            )
            logger.info(f"使用NSGA-II，种群大小 {self.config.population_size}")
        else:
            # 对于多目标（4-6），使用NSGA-III
            # 根据目标数量调整分区以确保 ref_dirs <= pop_size
            if self.config.n_objectives == 4:
                # 6个分区给出70个参考点
                ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
            elif self.config.n_objectives == 5:
                # 4个分区给出70个参考点
                ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=4)
            else:  # 6个目标
                # 3个分区给出84个参考点
                ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=3)
            
            # 确保种群大小至少与参考方向一样大
            pop_size = max(self.config.population_size, len(ref_dirs) + 20)
            
            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=self.config.crossover_eta, 
                             prob=self.config.crossover_prob),
                mutation=PM(eta=self.config.mutation_eta, 
                           prob=1.0/self.problem.n_var),
                eliminate_duplicates=True
            )
            
            logger.info(f"使用NSGA-III，{len(ref_dirs)} 个参考方向，"
                       f"种群大小 {pop_size}")
        
        return algorithm
    
    def optimize(self) -> Tuple[pd.DataFrame, Dict]:
        """运行优化并返回结果"""
        logger.info(f"开始 {self.algorithm.__class__.__name__} 优化，"
                   f"{self.config.n_objectives} 个目标...")
        
        # 配置终止条件
        termination = get_termination("n_gen", self.config.n_generations)
        
        # 运行优化
        res = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=42,
            save_history=True,
            verbose=True
        )
        
        # 处理结果
        pareto_df = self._process_results(res)
        
        # 提取历史（修复n_eval访问）
        history = {
            'n_evals': self.problem._eval_count,
            'exec_time': res.exec_time if hasattr(res, 'exec_time') else 0,
            'n_gen': self.config.n_generations,
            'history': res.history if hasattr(res, 'history') else None
        }
        
        return pareto_df, history
    
    def _process_results(self, res) -> pd.DataFrame:
        """将优化结果转换为DataFrame"""
        if res.X is None or (hasattr(res.X, '__len__') and len(res.X) == 0):
            logger.warning("未找到可行解！检查约束。")
            # 返回带有预期列的空DataFrame
            return pd.DataFrame(columns=[
                'solution_id', 'sensor', 'algorithm', 'data_rate_Hz',
                'geometric_LOD', 'condition_LOD', 'detection_threshold',
                'storage', 'communication', 'deployment', 'crew_size',
                'inspection_cycle_days', 'f1_total_cost_USD', 'f2_one_minus_recall',
                'f3_latency_seconds', 'f4_traffic_disruption_hours',
                'f5_carbon_emissions_kgCO2e_year', 'f6_system_reliability_inverse_MTBF',
                'detection_recall', 'system_MTBF_hours', 'system_MTBF_years',
                'annual_cost_USD', 'cost_per_km_year', 'carbon_footprint_tons_CO2_year',
                'is_feasible'
            ])
        
        # 确保数组是2D的
        X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
        F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
        
        results = []
        
        for i in range(len(X)):
            # 解码配置
            config = self.evaluator.solution_mapper.decode_solution(X[i])
            
            # 获取目标
            objectives = F[i]
            
            # 创建结果字典
            result = {
                'solution_id': i + 1,
                
                # 配置详情
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
                
                # 原始目标（6个目标）
                'f1_total_cost_USD': float(objectives[0]),
                'f2_one_minus_recall': float(objectives[1]),
                'f3_latency_seconds': float(objectives[2]),
                'f4_traffic_disruption_hours': float(objectives[3]),
                'f5_carbon_emissions_kgCO2e_year': float(objectives[4]),
                'f6_system_reliability_inverse_MTBF': float(objectives[5]),
                
                # 派生指标
                'detection_recall': float(1 - objectives[1]),
                'system_MTBF_hours': float(1/objectives[5] if objectives[5] > 0 else 1e6),
                'system_MTBF_years': float(1/objectives[5]/8760 if objectives[5] > 0 else 100),
                'annual_cost_USD': float(objectives[0] / self.config.planning_horizon_years),
                'cost_per_km_year': float(objectives[0] / self.config.planning_horizon_years / 
                                         self.config.road_network_length_km),
                'carbon_footprint_tons_CO2_year': float(objectives[4] / 1000),
                
                # 所有Pareto解都是可行的
                'is_feasible': True
            }
            
            results.append(result)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('f1_total_cost_USD')
            
            # 添加排名
            df['cost_rank'] = df['f1_total_cost_USD'].rank()
            df['recall_rank'] = df['detection_recall'].rank(ascending=False)
            df['carbon_rank'] = df['f5_carbon_emissions_kgCO2e_year'].rank()
            df['reliability_rank'] = df['system_MTBF_hours'].rank(ascending=False)
        
        logger.info(f"处理了 {len(df)} 个Pareto最优解")
        
        return df


class OptimizationAnalyzer:
    """分析优化结果"""
    
    @staticmethod
    def calculate_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
        """计算超体积指标"""
        try:
            from pymoo.indicators.hv import HV
            ind = HV(ref_point=ref_point)
            return ind(F)
        except ImportError:
            logger.warning("pymoo HV指标不可用")
            return 0.0
    
    @staticmethod
    def calculate_igd(F: np.ndarray, pareto_front: np.ndarray) -> float:
        """计算反世代距离"""
        try:
            from pymoo.indicators.igd import IGD
            ind = IGD(pareto_front)
            return ind(F)
        except ImportError:
            logger.warning("pymoo IGD指标不可用")
            return 0.0
    
    @staticmethod
    def analyze_convergence(history) -> Dict:
        """分析收敛特性"""
        if not history or not hasattr(history, 'data'):
            return {}
        
        results = {
            'n_generations': len(history),
            'convergence_metrics': []
        }
        
        # 跟踪各代指标
        for gen, algo in enumerate(history):
            if hasattr(algo, 'pop') and algo.pop is not None:
                F = algo.pop.get('F')
                if F is not None:
                    metrics = {
                        'generation': gen,
                        'n_solutions': len(F),
                        'best_cost': F[:, 0].min(),
                        'best_recall': 1 - F[:, 1].min(),
                        'avg_cost': F[:, 0].mean(),
                        'avg_recall': 1 - F[:, 1].mean()
                    }
                    
                    # 添加6目标指标（如果可用）
                    if F.shape[1] > 4:
                        metrics['best_carbon'] = F[:, 4].min()
                        metrics['avg_carbon'] = F[:, 4].mean()
                    
                    if F.shape[1] > 5:
                        metrics['best_reliability'] = 1/F[:, 5].max() if F[:, 5].max() > 0 else 0
                        metrics['avg_reliability'] = np.mean([1/x if x > 0 else 0 for x in F[:, 5]])
                    
                    results['convergence_metrics'].append(metrics)
        
        return results
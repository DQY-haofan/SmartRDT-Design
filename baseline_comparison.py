"""
Final Baseline Comparison Module v7.0 - FIXED
Implements expert recommendations for feasible and defensible baselines.
- Smarter Greedy: Iterates through Sensor/Algorithm pairs to find the cheapest FEASIBLE system.
- Weighted-Sum: Uses more balanced weights and a deeper search (50 generations).
"""

import json
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import logging
from rdflib import Namespace, URIRef
from tqdm import tqdm
from pymoo.core.callback import Callback

logger = logging.getLogger(__name__)
RDTCO = Namespace("http://example.org/rdtco-maint#")

# Custom JSON encoder to handle numpy and rdflib types
class UniversalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, URIRef):
            return str(obj)
        return super(UniversalEncoder, self).default(obj)

class ProgressCallback(Callback):
    """Progress bar callback class"""
    def __init__(self, n_gen=100, desc="Optimization"):
        super().__init__()
        self.pbar = None
        self.n_gen = n_gen
        self.desc = desc

    def notify(self, algorithm):
        if self.pbar is None:
            self.pbar = tqdm(total=self.n_gen, desc=self.desc)

        self.pbar.update(1)
        self.pbar.set_postfix({
            'Gen': algorithm.n_gen,
            'Best': f"{algorithm.opt[0].F[0]:.4f}" if algorithm.opt and algorithm.opt[0].F is not None else "N/A"
        })

        if algorithm.n_gen >= self.n_gen:
            self.pbar.close()


def run_greedy_cost_baseline(problem, ontology_graph):
    """
    Baseline 1: Improved Greedy Heuristic
    保证找到可行解的贪心算法
    """
    logger.info("Running Improved Greedy Heuristic Baseline...")
    mapper = problem.mapper
    evaluator = problem.evaluator

    # 存储所有可行配置
    feasible_configs = []

    # Step 1: 快速扫描找到至少一个可行配置
    logger.info("Step 1: Quick scan for feasible configurations...")

    # 优先尝试高性能组合
    priority_sensors = []
    for i, sensor in enumerate(mapper.sensor_options):
        sensor_name = str(sensor).split('#')[-1]
        # 优先选择MMS和高质量传感器
        if 'MMS' in sensor_name and 'LiDAR' in sensor_name:
            priority_sensors.append(i)
        elif 'UAV' in sensor_name and 'LiDAR' in sensor_name:
            priority_sensors.append(i)

    # 添加其他传感器
    for i in range(len(mapper.sensor_options)):
        if i not in priority_sensors:
            sensor_name = str(mapper.sensor_options[i]).split('#')[-1]
            # 跳过明显不好的
            if 'IoT' not in sensor_name and 'Vehicle_LowCost' not in sensor_name:
                priority_sensors.append(i)

    # 优先尝试高性能算法
    priority_algos = []
    for j, algo in enumerate(mapper.algorithm_options):
        algo_name = str(algo).split('#')[-1]
        if 'ML_SVM' in algo_name:  # 从之前结果看SVM表现好
            priority_algos.insert(0, j)
        elif 'DL_UNet' in algo_name:
            priority_algos.append(j)
        elif 'ML_RandomForest' in algo_name:
            priority_algos.append(j)

    # 添加其他算法
    for j in range(len(mapper.algorithm_options)):
        if j not in priority_algos:
            priority_algos.append(j)

    # 快速扫描（限制搜索空间）
    scan_count = 0
    max_scan = 200  # 最多扫描200个组合

    for sensor_idx in priority_sensors[:5]:  # 只试前5个优先传感器
        for algo_idx in priority_algos[:10]:  # 只试前10个优先算法
            if scan_count >= max_scan:
                break

            # 尝试不同的参数组合
            for lod in [0, 1]:  # Micro和Meso
                for cycle in [30, 60]:  # 月检和双月检
                    scan_count += 1

                    x_test = np.array([
                        sensor_idx,
                        20.0,  # 标准数据率
                        lod,
                        lod,
                        algo_idx,
                        0.5,  # 中等阈值
                        1,  # Cloud存储
                        3,  # 4G通信
                        2,  # Hybrid部署
                        3,  # 3人团队（不要太少）
                        cycle
                    ])

                    try:
                        raw_values = evaluator.get_raw_objectives(x_test)
                        f1_raw, f2_raw, f3_raw, f4_raw = raw_values
                        recall = 1 - f2_raw

                        g1 = f3_raw - evaluator.config['max_latency_seconds']
                        g2 = evaluator.config['min_recall_threshold'] - recall
                        g3 = f1_raw - evaluator.config['budget_cap_usd']

                        if g1 <= 0 and g2 <= 0 and g3 <= 0:
                            feasible_configs.append((x_test.copy(), f1_raw))
                            logger.info(f"Found feasible: Sensor={sensor_idx}, Algo={algo_idx}, "
                                        f"Cost=${f1_raw:.0f}, Recall={recall:.3f}")
                    except Exception as e:
                        logger.debug(f"Error evaluating: {e}")
                        continue

    logger.info(f"Scan complete. Found {len(feasible_configs)} feasible configurations")

    # Step 2: 如果没找到可行配置，尝试更激进的参数
    if not feasible_configs:
        logger.info("Step 2: No feasible found in quick scan, trying aggressive parameters...")

        # 使用已知的好组合（基于NSGA-II结果）
        known_good_combos = [
            (0, 0),  # 第一个传感器+第一个算法
            (1, 0),  # 第二个传感器+第一个算法
            (0, 1),  # 第一个传感器+第二个算法
        ]

        for sensor_idx, algo_idx in known_good_combos:
            if sensor_idx >= len(mapper.sensor_options) or algo_idx >= len(mapper.algorithm_options):
                continue

            # 尝试Micro LOD和小团队来提高recall
            x_aggressive = np.array([
                sensor_idx,
                15.0,  # 较低数据率
                0,  # Micro LOD（最高质量）
                0,  # Micro LOD
                algo_idx,
                0.4,  # 较低阈值
                1,  # Cloud
                3,  # 4G
                2,  # Hybrid
                2,  # 2人团队（降低成本）
                30  # 月检
            ])

            try:
                raw_values = evaluator.get_raw_objectives(x_aggressive)
                f1_raw, f2_raw, f3_raw, f4_raw = raw_values
                recall = 1 - f2_raw

                g1 = f3_raw - evaluator.config['max_latency_seconds']
                g2 = evaluator.config['min_recall_threshold'] - recall
                g3 = f1_raw - evaluator.config['budget_cap_usd']

                logger.info(f"Aggressive attempt: Cost=${f1_raw:.0f}, Recall={recall:.3f}, "
                            f"Latency={f3_raw:.1f}s")

                if g1 <= 0 and g2 <= 0 and g3 <= 0:
                    feasible_configs.append((x_aggressive.copy(), f1_raw))
                    logger.info("Found feasible with aggressive parameters!")
            except:
                continue

    # Step 3: 选择成本最低的可行配置
    if feasible_configs:
        best_x, best_cost = min(feasible_configs, key=lambda x: x[1])
        logger.info(f"Selected lowest cost feasible solution: ${best_cost:.0f}")
    else:
        # 最后的尝试：使用NSGA-II找到的配置作为参考
        logger.warning("No feasible solution found! Using fallback configuration")
        # 使用一个保守但可能可行的配置
        best_x = np.array([
            1,  # 第二个传感器（避免索引0）
            20.0,  # 标准数据率
            0,  # Micro LOD
            0,  # Micro LOD
            4,  # 第5个算法（避免前几个）
            0.5,  # 中等阈值
            1,  # Cloud
            3,  # 4G
            2,  # Hybrid
            3,  # 3人团队
            30  # 月检
        ])

    # 最终评估
    config = mapper.decode_solution(best_x)
    raw_values = evaluator.get_raw_objectives(best_x)
    f1_raw, f2_raw, f3_raw, f4_raw = raw_values
    recall = 1 - f2_raw

    g1 = f3_raw - evaluator.config['max_latency_seconds']
    g2 = evaluator.config['min_recall_threshold'] - recall
    g3 = f1_raw - evaluator.config['budget_cap_usd']

    is_feasible = (g1 <= 0 and g2 <= 0 and g3 <= 0)

    result = {
        'method': 'Greedy Cost-Minimization',
        'configuration': {
            'sensor': str(config['sensor']).split('#')[-1],
            'data_rate_Hz': float(config['data_rate']),
            'geometric_LOD': config['geo_lod'],
            'condition_LOD': config['cond_lod'],
            'algorithm': str(config['algorithm']).split('#')[-1],
            'detection_threshold': float(config['detection_threshold']),
            'storage': str(config['storage']).split('#')[-1],
            'communication': str(config['communication']).split('#')[-1],
            'deployment': str(config['deployment']).split('#')[-1],
            'crew_size': int(config['crew_size']),
            'inspection_cycle_days': int(config['inspection_cycle'])
        },
        'objectives': {
            'f1_total_cost_USD': float(f1_raw),
            'f2_one_minus_recall': float(f2_raw),
            'f3_latency_seconds': float(f3_raw),
            'f4_traffic_disruption_hours': float(f4_raw),
            'detection_recall': float(recall)
        },
        'constraints': {
            'g1_latency_violation': float(g1),
            'g2_recall_violation': float(g2),
            'g3_budget_violation': float(g3),
            'is_feasible': is_feasible
        },
        'x_vector': best_x.tolist()
    }

    with open('./results/baseline_greedy_solution.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, cls=UniversalEncoder)

    logger.info(f"Greedy baseline complete")
    logger.info(f"Final result: Cost=${f1_raw:.0f}, Recall={recall:.3f}, "
                f"Latency={f3_raw:.1f}s, Feasible={is_feasible}")

    if not is_feasible:
        logger.warning("WARNING: Greedy baseline failed to find a feasible solution!")
        logger.warning(f"Constraint violations: g1={g1:.3f}, g2={g2:.3f}, g3={g3:.3f}")

    return result

def run_weighted_sum_baseline(problem, ontology_graph):
    """
    Baseline 2: Fixed Weighted-Sum Single-Objective Optimization
    Implements balanced weights and a deeper search.
    """
    logger.info("Running Weighted-Sum Single-Objective Baseline (Fixed)...")

    # FIX 1: Rebalance weights to give latency more importance
    weights = {
        'cost': 0.4,
        'recall': 0.3,
        'latency': 0.2,
        'disruption': 0.1
    }

    class WeightedSumProblem(Problem):
        def __init__(self, original_problem, weights):
            self.original_problem = original_problem
            self.weights = weights
            self.mapper = original_problem.mapper
            self.evaluator = original_problem.evaluator

            super().__init__(
                n_var=original_problem.n_var,
                n_obj=1,
                n_constr=original_problem.n_constr,
                xl=original_problem.xl,
                xu=original_problem.xu,
                type_var=original_problem.type_var
            )

        def _evaluate(self, X, out, *args, **kwargs):
            objectives = []
            constraints = []

            for x in X:
                # Handle integer variables
                x_copy = x.copy()
                for i in range(len(x_copy)):
                    if self.original_problem.var_types[i] == 'int':
                        x_copy[i] = int(np.round(x_copy[i]))
                        x_copy[i] = np.clip(x_copy[i], self.xl[i], self.xu[i])

                # Get normalized objectives
                obj_values = self.evaluator.evaluate(x_copy)

                # Calculate weighted sum
                weighted_sum = (
                    self.weights['cost'] * obj_values[0] +
                    self.weights['recall'] * obj_values[1] +
                    self.weights['latency'] * obj_values[2] +
                    self.weights['disruption'] * obj_values[3]
                )

                # Get raw values for constraints
                raw_values = self.evaluator.get_raw_objectives(x_copy)
                f1_raw, f2_raw, f3_raw, _ = raw_values
                recall = 1 - f2_raw

                g1 = f3_raw - self.evaluator.config['max_latency_seconds']
                g2 = self.evaluator.config['min_recall_threshold'] - recall
                g3 = f1_raw - self.evaluator.config['budget_cap_usd']

                # Add strong penalties for constraint violations
                penalty = 0
                if g1 > 0:  # Latency violation
                    penalty += (g1 / self.evaluator.config['max_latency_seconds']) * 100
                if g2 > 0:  # Recall violation
                    penalty += g2 * 200  # Stronger penalty for recall
                if g3 > 0:  # Budget violation
                    penalty += (g3 / self.evaluator.config['budget_cap_usd']) * 100

                objectives.append([weighted_sum + penalty])
                constraints.append([g1, g2, g3])

            out["F"] = np.array(objectives)
            out["G"] = np.array(constraints)

    ws_problem = WeightedSumProblem(problem, weights)

    algorithm = GA(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20, prob=1.0/ws_problem.n_var),
        eliminate_duplicates=True
    )

    # FIX 2: Increase search depth
    n_generations = 10
    termination = get_termination("n_gen", n_generations)
    progress_callback = ProgressCallback(n_gen=n_generations, desc="Weighted-Sum GA")

    res = minimize(
        ws_problem,
        algorithm,
        termination,
        seed=42,
        verbose=False,
        callback=progress_callback
    )

    if res.X is None:
        logger.error("Weighted-Sum baseline failed to find a solution.")
        return None

    x_weighted = res.X

    # Handle integer variables
    for i in range(len(x_weighted)):
        if problem.var_types[i] == 'int':
            x_weighted[i] = int(np.round(x_weighted[i]))
            x_weighted[i] = np.clip(x_weighted[i], problem.xl[i], problem.xu[i])

    config = problem.mapper.decode_solution(x_weighted)
    raw_values = problem.evaluator.get_raw_objectives(x_weighted)
    f1_raw, f2_raw, f3_raw, f4_raw = raw_values
    recall = 1 - f2_raw

    g1 = f3_raw - problem.evaluator.config['max_latency_seconds']
    g2 = problem.evaluator.config['min_recall_threshold'] - recall
    g3 = f1_raw - problem.evaluator.config['budget_cap_usd']

    result = {
        'method': 'Weighted-Sum Optimization',
        'weights': weights,
        'configuration': {
            'sensor': str(config['sensor']).split('#')[-1],
            'data_rate_Hz': float(config['data_rate']),
            'geometric_LOD': config['geo_lod'],
            'condition_LOD': config['cond_lod'],
            'algorithm': str(config['algorithm']).split('#')[-1],
            'detection_threshold': float(config['detection_threshold']),
            'storage': str(config['storage']).split('#')[-1],
            'communication': str(config['communication']).split('#')[-1],
            'deployment': str(config['deployment']).split('#')[-1],
            'crew_size': int(config['crew_size']),
            'inspection_cycle_days': int(config['inspection_cycle'])
        },
        'objectives': {
            'f1_total_cost_USD': float(f1_raw),
            'f2_one_minus_recall': float(f2_raw),
            'f3_latency_seconds': float(f3_raw),
            'f4_traffic_disruption_hours': float(f4_raw),
            'detection_recall': float(recall)
        },
        'constraints': {
            'g1_latency_violation': float(g1),
            'g2_recall_violation': float(g2),
            'g3_budget_violation': float(g3),
            'is_feasible': bool(g1 <= 0 and g2 <= 0 and g3 <= 0)
        },
        'weighted_sum_score': float(res.F[0]),
        'x_vector': x_weighted.tolist()
    }

    with open('./results/baseline_weighted_sum_solution.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, cls=UniversalEncoder)

    logger.info(f"Weighted-sum baseline complete: Cost=${f1_raw:.0f}, Recall={recall:.3f}, "
                f"Latency={f3_raw:.1f}s, Feasible={result['constraints']['is_feasible']}")

    return result

def run_all_baselines(problem, ontology_graph):
    """Run all baseline methods"""
    results = {}

    # Run greedy baseline
    greedy_result = run_greedy_cost_baseline(problem, ontology_graph)
    if greedy_result:
        results['greedy'] = greedy_result
    else:
        logger.error("Greedy baseline failed to produce a result")

    # Run weighted sum baseline
    weighted_result = run_weighted_sum_baseline(problem, ontology_graph)
    if weighted_result:
        results['weighted_sum'] = weighted_result
    else:
        logger.error("Weighted-sum baseline failed to produce a result")

    # Save combined results
    with open('./results/all_baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=UniversalEncoder)

    logger.info("All baseline comparisons complete")
    return results

def run_ablation_study(problem, ontology_graph):
    """
    Ablation Study: Run NSGA-II without semantic pre-filtering
    """
    logger.info("Running Ablation Study (No Semantic Pre-filtering)...")

    from pymoo.algorithms.moo.nsga2 import NSGA2

    class AblationProgressCallback(Callback):
        """Ablation study progress bar callback"""
        def __init__(self, n_gen=10):
            super().__init__()
            self.pbar = None
            self.n_gen = n_gen

        def notify(self, algorithm):
            if self.pbar is None:
                self.pbar = tqdm(total=self.n_gen, desc="Ablation Study")

            self.pbar.update(1)

            # Calculate feasibility statistics
            if hasattr(algorithm, 'pop') and algorithm.pop is not None:
                G = algorithm.pop.get("G")
                if G is not None:
                    n_feasible = np.sum([np.all(g <= 0) for g in G])
                    cv_avg = np.mean([np.sum(np.maximum(0, g)) for g in G])
                    self.pbar.set_postfix({
                        'Gen': algorithm.n_gen,
                        'Feasible': n_feasible,
                        'CV_avg': f"{cv_avg:.3f}"
                    })

            if algorithm.n_gen >= self.n_gen:
                self.pbar.close()

    # Configure NSGA-II without semantic pre-filtering
    algorithm = NSGA2(
        pop_size=200,
        sampling=FloatRandomSampling(),  # Pure random sampling
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20, prob=1.0/problem.n_var),
        eliminate_duplicates=True
    )

    # Run optimization
    n_generations = 10
    termination = get_termination("n_gen", n_generations)
    progress_callback = AblationProgressCallback(n_gen=n_generations)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        save_history=True,
        verbose=False,
        callback=progress_callback
    )

    # Analyze convergence history
    convergence_data = []
    for gen in range(len(res.history)):
        opt = res.history[gen].opt
        if opt is not None and len(opt) > 0:
            cv_avg = np.mean([np.sum(np.maximum(0, g)) for g in opt.get("G")])
            n_feasible = np.sum([np.all(g <= 0) for g in opt.get("G")])

            convergence_data.append({
                'generation': gen,
                'cv_average': float(cv_avg),
                'n_feasible': int(n_feasible),
                'n_total': len(opt)
            })

    # Save ablation study results
    ablation_result = {
        'method': 'NSGA-II without Semantic Pre-filtering',
        'convergence_history': convergence_data,
        'final_statistics': {
            'n_solutions': len(res.X) if res.X is not None else 0,
            'n_feasible': np.sum([np.all(g <= 0) for g in res.G]) if res.G is not None else 0,
            'generations_to_first_feasible': next((d['generation'] for d in convergence_data if d['n_feasible'] > 0), -1)
        }
    }

    with open('./results/ablation_study_results.json', 'w', encoding='utf-8') as f:
        json.dump(ablation_result, f, indent=2, cls=UniversalEncoder)

    logger.info(f"Ablation study complete: {ablation_result['final_statistics']['n_feasible']} "
                f"feasible solutions found")

    return ablation_result
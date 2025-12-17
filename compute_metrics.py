#!/usr/bin/env python3
"""
RMTwin Professional Multi-Objective Metrics (6D + Run-Level)
=============================================================
修复审稿级问题：
1. 6D指标计算（而非2D投影）
2. Run-level统计检验（而非solution-level）
3. 双向Coverage正确解释

Author: RMTwin Research Team
Version: 2.0 (Publication-Ready)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy import stats
import json
import glob
import warnings

warnings.filterwarnings('ignore')


class MultiObjectiveMetrics6D:
    """
    6维多目标优化指标计算器

    目标定义（全部最小化形式）：
    f1: total_cost_USD (min)
    f2: 1 - detection_recall (min)
    f3: latency_seconds (min)
    f4: traffic_disruption_hours (min)
    f5: carbon_emissions_kgCO2e_year (min)
    f6: 可选的第6个目标
    """

    def __init__(self, n_objectives: int = 5):
        """
        Args:
            n_objectives: 目标数量 (5 或 6)
        """
        self.n_objectives = n_objectives

        # 定义目标列名（最小化形式）
        self.objective_cols = [
            'f1_total_cost_USD',
            'f2_one_minus_recall',  # 需要从detection_recall计算
            'f3_latency_seconds',
            'f4_traffic_disruption_hours',
            'f5_carbon_emissions_kgCO2e_year',
        ]

        if n_objectives >= 6:
            self.objective_cols.append('f6_maintenance_complexity')

    def _prepare_objective_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        准备目标矩阵（全部最小化形式）
        """
        df = df.copy()

        # 转换recall为1-recall（最小化形式）
        if 'f2_one_minus_recall' not in df.columns:
            if 'detection_recall' in df.columns:
                df['f2_one_minus_recall'] = 1 - df['detection_recall']
            else:
                raise ValueError("Missing recall column")

        # 提取可用的目标列
        available_cols = [c for c in self.objective_cols if c in df.columns]

        if len(available_cols) < 2:
            raise ValueError(f"Not enough objective columns. Found: {available_cols}")

        return df[available_cols].values, available_cols

    def _dominates(self, p: np.ndarray, q: np.ndarray) -> bool:
        """
        检查p是否支配q（全部最小化）
        p dominates q iff: ∀i: p[i] ≤ q[i] AND ∃j: p[j] < q[j]
        """
        return np.all(p <= q) and np.any(p < q)

    def _is_nondominated(self, point: np.ndarray, others: np.ndarray) -> bool:
        """检查point是否为非支配解"""
        for other in others:
            if self._dominates(other, point):
                return False
        return True

    def hypervolume_monte_carlo(self, df: pd.DataFrame, ref_point: np.ndarray = None,
                                n_samples: int = 100000) -> float:
        """
        蒙特卡洛方法计算高维Hypervolume

        对于6D问题，精确计算HV是NP-hard的，使用蒙特卡洛近似
        """
        F, cols = self._prepare_objective_matrix(df)
        n_obj = F.shape[1]

        if len(F) == 0:
            return 0.0

        # 设置参考点（如果未提供，使用1.1倍最大值）
        if ref_point is None:
            ref_point = F.max(axis=0) * 1.1

        # 理想点
        ideal_point = F.min(axis=0)

        # 归一化
        range_vals = ref_point - ideal_point
        range_vals[range_vals == 0] = 1
        F_norm = (F - ideal_point) / range_vals
        ref_norm = np.ones(n_obj)

        # 蒙特卡洛采样
        samples = np.random.uniform(0, 1, size=(n_samples, n_obj))

        # 计算被Pareto前沿支配的样本比例
        dominated_count = 0
        for sample in samples:
            # 检查是否被任意Pareto解支配
            for sol in F_norm:
                if np.all(sol <= sample):
                    dominated_count += 1
                    break

        # HV = 支配比例 × 总体积
        total_volume = np.prod(ref_norm)  # = 1 for normalized
        hv = (dominated_count / n_samples) * total_volume

        return float(hv)

    def hypervolume_2d(self, df: pd.DataFrame, ref_point: np.ndarray = None) -> float:
        """
        精确计算2D Hypervolume（用于cost-recall投影）
        """
        F, _ = self._prepare_objective_matrix(df)
        F_2d = F[:, :2]  # 只取前两个目标

        if len(F_2d) == 0:
            return 0.0

        if ref_point is None:
            ref_point = F_2d.max(axis=0) * 1.1
        else:
            ref_point = ref_point[:2]

        # 过滤
        valid = np.all(F_2d < ref_point, axis=1)
        F_2d = F_2d[valid]

        if len(F_2d) == 0:
            return 0.0

        # 按第一个目标排序
        sorted_idx = np.argsort(F_2d[:, 0])
        F_2d = F_2d[sorted_idx]

        # 计算面积
        hv = 0.0
        prev_y = ref_point[1]

        for i in range(len(F_2d)):
            if F_2d[i, 1] < prev_y:
                width = ref_point[0] - F_2d[i, 0]
                height = prev_y - F_2d[i, 1]
                hv += width * height
                prev_y = F_2d[i, 1]

        return float(hv)

    def coverage_6d(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[float, float]:
        """
        计算6D Coverage指标

        Returns:
            (C(A,B), C(B,A)) - A支配B的比例, B支配A的比例
        """
        F_a, _ = self._prepare_objective_matrix(df_a)
        F_b, _ = self._prepare_objective_matrix(df_b)

        if len(F_a) == 0 or len(F_b) == 0:
            return (0.0, 0.0)

        # A支配B的数量
        a_dom_b = 0
        for b in F_b:
            for a in F_a:
                if self._dominates(a, b):
                    a_dom_b += 1
                    break

        # B支配A的数量
        b_dom_a = 0
        for a in F_a:
            for b in F_b:
                if self._dominates(b, a):
                    b_dom_a += 1
                    break

        c_ab = a_dom_b / len(F_b) * 100
        c_ba = b_dom_a / len(F_a) * 100

        return (c_ab, c_ba)

    def contribution_6d(self, pareto_df: pd.DataFrame,
                        baseline_dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        计算各方法对6D合并Pareto前沿的贡献
        """
        all_dfs = {'NSGA-III': pareto_df, **baseline_dfs}

        all_F = []
        sources = []

        for name, df in all_dfs.items():
            if len(df) == 0:
                continue
            try:
                F, _ = self._prepare_objective_matrix(df)
                for f in F:
                    all_F.append(f)
                    sources.append(name)
            except:
                continue

        if len(all_F) == 0:
            return {name: 0.0 for name in all_dfs}

        all_F = np.array(all_F)

        # 找非支配解
        non_dom_idx = []
        for i in range(len(all_F)):
            others = np.delete(all_F, i, axis=0)
            if self._is_nondominated(all_F[i], others):
                non_dom_idx.append(i)

        # 统计贡献
        contrib = {name: 0 for name in all_dfs}
        for idx in non_dom_idx:
            contrib[sources[idx]] += 1

        total = len(non_dom_idx)
        return {k: v / total * 100 if total > 0 else 0 for k, v in contrib.items()}

    def spacing_6d(self, df: pd.DataFrame) -> float:
        """计算6D Spacing"""
        F, _ = self._prepare_objective_matrix(df)
        if len(F) < 2:
            return 0.0

        # 归一化
        ideal, nadir = F.min(axis=0), F.max(axis=0)
        range_vals = nadir - ideal
        range_vals[range_vals == 0] = 1
        F_norm = (F - ideal) / range_vals

        # 最近邻距离
        dist_matrix = cdist(F_norm, F_norm)
        np.fill_diagonal(dist_matrix, np.inf)
        d_i = dist_matrix.min(axis=1)
        d_mean = d_i.mean()

        return float(np.sqrt(np.sum((d_i - d_mean) ** 2) / (len(F) - 1)))


class RunLevelAnalyzer:
    """
    Run-level统计分析器

    正确的统计推断：以"run"为单位，而非"solution"
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.metrics = MultiObjectiveMetrics6D()

    def find_all_runs(self) -> List[Path]:
        """查找所有实验运行目录"""
        runs = []
        for pattern in ['*/pareto_solutions.csv', '*/*/pareto_solutions.csv']:
            runs.extend(self.results_dir.glob(pattern))
        return [p.parent for p in runs]

    def compute_run_metrics(self, run_dir: Path) -> Dict:
        """计算单次run的所有指标"""

        pareto_file = run_dir / 'pareto_solutions.csv'
        if not pareto_file.exists():
            return None

        pareto_df = pd.read_csv(pareto_file)

        # 加载baselines
        baseline_dfs = {}
        for f in run_dir.glob('baseline_*.csv'):
            name = f.stem.replace('baseline_', '')
            df = pd.read_csv(f)
            if 'is_feasible' in df.columns:
                df = df[df['is_feasible']]
            baseline_dfs[name] = df

        # 计算共同参考点（用于HV计算的一致性）
        all_dfs = [pareto_df] + [df for df in baseline_dfs.values() if len(df) > 0]
        if not all_dfs:
            return None

        all_data = pd.concat(all_dfs, ignore_index=True)

        try:
            F_all, cols = self.metrics._prepare_objective_matrix(all_data)
            ref_point = F_all.max(axis=0) * 1.1
        except:
            return None

        results = {
            'run_dir': str(run_dir),
            'n_pareto_solutions': len(pareto_df),
            'n_objectives': len(cols),
            'objectives': cols,
        }

        # === 6D指标 ===
        try:
            # HV (蒙特卡洛)
            results['hv_6d'] = self.metrics.hypervolume_monte_carlo(pareto_df, ref_point)

            # 2D HV (精确，用于辅助说明)
            results['hv_2d'] = self.metrics.hypervolume_2d(pareto_df, ref_point[:2])

            # Spacing
            results['spacing_6d'] = self.metrics.spacing_6d(pareto_df)

            # Contribution
            contrib = self.metrics.contribution_6d(pareto_df, baseline_dfs)
            results['contribution_nsga'] = contrib.get('NSGA-III', 0)
            results['contribution_all'] = contrib

        except Exception as e:
            results['error'] = str(e)

        # === 与各baseline的Coverage ===
        for name, df in baseline_dfs.items():
            if len(df) == 0:
                results[f'coverage_nsga_dom_{name}'] = 0
                results[f'coverage_{name}_dom_nsga'] = 0
                results[f'hv_6d_{name}'] = 0
                continue

            try:
                c_ab, c_ba = self.metrics.coverage_6d(pareto_df, df)
                results[f'coverage_nsga_dom_{name}'] = c_ab
                results[f'coverage_{name}_dom_nsga'] = c_ba
                results[f'net_coverage_vs_{name}'] = c_ab - c_ba

                # Baseline的HV
                results[f'hv_6d_{name}'] = self.metrics.hypervolume_monte_carlo(df, ref_point)
            except:
                pass

        return results

    def analyze_all_runs(self) -> pd.DataFrame:
        """分析所有runs并汇总"""

        runs = self.find_all_runs()
        print(f"Found {len(runs)} runs")

        all_results = []
        for run_dir in runs:
            print(f"  Processing: {run_dir.name}")
            result = self.compute_run_metrics(run_dir)
            if result:
                all_results.append(result)

        if not all_results:
            print("No valid runs found!")
            return pd.DataFrame()

        return pd.DataFrame(all_results)

    def run_level_statistics(self, df: pd.DataFrame) -> Dict:
        """
        基于run-level指标进行统计检验

        这是正确的统计推断方式！
        """

        if len(df) < 2:
            return {'error': 'Need at least 2 runs for statistics'}

        results = {
            'n_runs': len(df),
            'hv_6d_mean': df['hv_6d'].mean(),
            'hv_6d_std': df['hv_6d'].std(),
            'hv_2d_mean': df['hv_2d'].mean(),
            'contribution_mean': df['contribution_nsga'].mean(),
        }

        # 如果有多个seed，可以做更多统计
        if len(df) >= 3:
            # 95%置信区间
            from scipy.stats import sem, t
            n = len(df)
            hv_mean = df['hv_6d'].mean()
            hv_sem = sem(df['hv_6d'])
            ci = t.ppf(0.975, n - 1) * hv_sem
            results['hv_6d_ci_95'] = (hv_mean - ci, hv_mean + ci)

        return results


def compute_metrics_6d(pareto_path: str, output_dir: str = './results/metrics_6d'):
    """
    计算6D指标的主函数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_df = pd.read_csv(pareto_path)
    pareto_dir = Path(pareto_path).parent

    # 加载baselines
    baseline_dfs = {}
    for f in pareto_dir.glob('baseline_*.csv'):
        name = f.stem.replace('baseline_', '')
        df = pd.read_csv(f)
        if 'is_feasible' in df.columns:
            df = df[df['is_feasible']]
        baseline_dfs[name] = df

    metrics = MultiObjectiveMetrics6D()

    print("=" * 70)
    print("6D MULTI-OBJECTIVE OPTIMIZATION METRICS")
    print("=" * 70)

    # 检测目标数量
    try:
        F, cols = metrics._prepare_objective_matrix(pareto_df)
        n_obj = len(cols)
        print(f"\nDetected {n_obj} objectives: {cols}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 计算参考点
    all_dfs = [pareto_df] + [df for df in baseline_dfs.values() if len(df) > 0]
    all_data = pd.concat(all_dfs, ignore_index=True)
    F_all, _ = metrics._prepare_objective_matrix(all_data)
    ref_point = F_all.max(axis=0) * 1.1

    print(f"Reference point: {ref_point}")

    # === 1. HV指标 ===
    print("\n[1] HYPERVOLUME (6D Monte Carlo + 2D Exact)")
    print("-" * 50)

    hv_results = []

    hv_6d = metrics.hypervolume_monte_carlo(pareto_df, ref_point)
    hv_2d = metrics.hypervolume_2d(pareto_df, ref_point[:2])
    hv_results.append({'Method': 'NSGA-III', 'HV_6D': hv_6d, 'HV_2D': hv_2d, 'N': len(pareto_df)})
    print(f"NSGA-III:  HV_6D={hv_6d:.4f}  HV_2D={hv_2d:.2e}  N={len(pareto_df)}")

    for name, df in baseline_dfs.items():
        if len(df) == 0:
            hv_results.append({'Method': name, 'HV_6D': 0, 'HV_2D': 0, 'N': 0})
            print(f"{name:<10}: No feasible solutions")
            continue

        hv_6d = metrics.hypervolume_monte_carlo(df, ref_point)
        hv_2d = metrics.hypervolume_2d(df, ref_point[:2])
        hv_results.append({'Method': name, 'HV_6D': hv_6d, 'HV_2D': hv_2d, 'N': len(df)})
        print(f"{name:<10}: HV_6D={hv_6d:.4f}  HV_2D={hv_2d:.2e}  N={len(df)}")

    pd.DataFrame(hv_results).to_csv(output_dir / 'hypervolume_6d.csv', index=False)

    # === 2. Coverage (6D, 双向) ===
    print("\n[2] COVERAGE (6D, Bidirectional)")
    print("-" * 50)
    print("Note: Coverage is bidirectional - both directions must be reported!")
    print()

    coverage_results = []
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            coverage_results.append({
                'Comparison': f'NSGA-III vs {name}',
                'C(NSGA,Baseline)': 0, 'C(Baseline,NSGA)': 0, 'Net': 0,
                'Interpretation': 'Baseline has no feasible solutions'
            })
            continue

        c_ab, c_ba = metrics.coverage_6d(pareto_df, df)
        net = c_ab - c_ba

        # 正确的解释
        if c_ab > c_ba + 10:
            interp = "NSGA-III has clear advantage"
        elif c_ba > c_ab + 10:
            interp = "Baseline has advantage"
        else:
            interp = "Both methods find different trade-off regions"

        coverage_results.append({
            'Comparison': f'NSGA-III vs {name}',
            'C(NSGA,Baseline)': c_ab,
            'C(Baseline,NSGA)': c_ba,
            'Net': net,
            'Interpretation': interp
        })

        print(f"NSGA-III vs {name}:")
        print(f"  C(NSGA→{name}) = {c_ab:.1f}%  (NSGA dominates this % of {name})")
        print(f"  C({name}→NSGA) = {c_ba:.1f}%  ({name} dominates this % of NSGA)")
        print(f"  Net advantage = {net:+.1f}%")
        print(f"  → {interp}")
        print()

    pd.DataFrame(coverage_results).to_csv(output_dir / 'coverage_6d.csv', index=False)

    # === 3. Contribution (6D) ===
    print("[3] CONTRIBUTION TO COMBINED 6D PARETO FRONT")
    print("-" * 50)

    contrib = metrics.contribution_6d(pareto_df, baseline_dfs)

    contrib_results = []
    for name, pct in sorted(contrib.items(), key=lambda x: -x[1]):
        n_sols = len(pareto_df) if name == 'NSGA-III' else len(baseline_dfs.get(name, []))
        efficiency = pct / n_sols if n_sols > 0 else 0
        contrib_results.append({
            'Method': name,
            'Contribution_%': pct,
            'N_Solutions': n_sols,
            'Efficiency': efficiency
        })
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"{name:<12}: {bar} {pct:5.1f}% (n={n_sols}, eff={efficiency:.3f}%/sol)")

    pd.DataFrame(contrib_results).to_csv(output_dir / 'contribution_6d.csv', index=False)

    # === 4. 汇总 ===
    summary = {
        'n_objectives': n_obj,
        'objectives': cols,
        'reference_point': ref_point.tolist(),
        'hypervolume': hv_results,
        'coverage': coverage_results,
        'contribution': contrib_results,
        'note': 'HV_6D uses Monte Carlo approximation (100k samples). Coverage is bidirectional.'
    }

    with open(output_dir / 'metrics_6d_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 70)

    # === 5. 生成严谨的结论模板 ===
    print("\n[RECOMMENDED PAPER STATEMENTS]")
    print("-" * 50)

    nsga_hv = [r['HV_6D'] for r in hv_results if r['Method'] == 'NSGA-III'][0]
    best_baseline_hv = max([r['HV_6D'] for r in hv_results if r['Method'] != 'NSGA-III'], default=0)

    print(f"""
Based on {n_obj}-objective analysis:

1. Hypervolume: "NSGA-III achieves HV={nsga_hv:.4f} on the {n_obj}D objective space,
   compared to {best_baseline_hv:.4f} for the best baseline."

2. Coverage (example for one baseline):
   "The bidirectional coverage analysis shows C(NSGA,B)={coverage_results[0]['C(NSGA,Baseline)']:.1f}% 
   and C(B,NSGA)={coverage_results[0]['C(Baseline,NSGA)']:.1f}%, indicating {coverage_results[0]['Interpretation'].lower()}."

3. Contribution: "NSGA-III contributes {contrib.get('NSGA-III', 0):.1f}% of the combined 
   non-dominated front with {len(pareto_df)} solutions."

NOTE: For statistically rigorous claims, run multiple seeds and report 
mean±std with confidence intervals.
""")

    return summary


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single run:  python compute_metrics_6d.py <pareto_csv> [output_dir]")
        print("  Multi run:   python compute_metrics_6d.py --multi <results_dir> [output_dir]")
        sys.exit(1)

    if sys.argv[1] == '--multi':
        # Run-level分析
        results_dir = sys.argv[2] if len(sys.argv) > 2 else './results/runs'
        output_dir = sys.argv[3] if len(sys.argv) > 3 else './results/metrics_runs'

        analyzer = RunLevelAnalyzer(results_dir)
        df = analyzer.analyze_all_runs()

        if len(df) > 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            df.to_csv(f'{output_dir}/run_level_metrics.csv', index=False)

            stats = analyzer.run_level_statistics(df)
            with open(f'{output_dir}/run_level_statistics.json', 'w') as f:
                json.dump(stats, f, indent=2, default=float)

            print(f"\nRun-level statistics saved to: {output_dir}")
    else:
        # 单次run分析
        pareto_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else './results/metrics_6d'

        compute_metrics_6d(pareto_path, output_dir)
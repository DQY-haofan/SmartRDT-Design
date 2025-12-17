#!/usr/bin/env python3
"""
RMTwin Professional Multi-Objective Optimization Metrics
=========================================================
计算专业的多目标优化性能指标，用于顶刊论文

指标包括:
1. Hypervolume (HV) - Pareto前沿覆盖的超体积，越大越好
2. Spacing (SP) - 解分布的均匀性，越小越好
3. Maximum Spread (MS) - 前沿的延展程度，越大越好
4. Coverage (C) - 支配关系统计
5. Contribution - 各方法对合并前沿的贡献
6. Cliff's Delta - 效应量

Author: RMTwin Research Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from scipy.spatial.distance import cdist
from scipy import stats
import json
import warnings

warnings.filterwarnings('ignore')


class MultiObjectiveMetrics:
    """专业多目标优化指标计算器"""

    def __init__(self):
        self.objectives = ['f1_total_cost_USD', 'f2_one_minus_recall']

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """准备目标矩阵 (最小化形式)"""
        df = df.copy()
        if 'f2_one_minus_recall' not in df.columns:
            df['f2_one_minus_recall'] = 1 - df['detection_recall']
        return df[self.objectives].values

    def _normalize(self, F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
        """归一化到[0,1]"""
        range_vals = nadir - ideal
        range_vals[range_vals == 0] = 1
        return (F - ideal) / range_vals

    def hypervolume_2d(self, df: pd.DataFrame, ref_point: np.ndarray = None) -> float:
        """计算2D Hypervolume - 核心指标"""
        F = self._prepare_data(df)

        if ref_point is None:
            ref_point = F.max(axis=0) * 1.1

        # 过滤被参考点支配的点
        valid = np.all(F < ref_point, axis=1)
        F = F[valid]

        if len(F) == 0:
            return 0.0

        # 按第一个目标排序
        sorted_idx = np.argsort(F[:, 0])
        F = F[sorted_idx]

        # 计算面积
        hv = 0.0
        prev_y = ref_point[1]

        for i in range(len(F)):
            if F[i, 1] < prev_y:
                width = ref_point[0] - F[i, 0]
                height = prev_y - F[i, 1]
                hv += width * height
                prev_y = F[i, 1]

        return float(hv)

    def spacing(self, df: pd.DataFrame) -> float:
        """计算Spacing - 解分布均匀性，越小越好"""
        F = self._prepare_data(df)
        if len(F) < 2:
            return 0.0

        ideal, nadir = F.min(axis=0), F.max(axis=0)
        F_norm = self._normalize(F, ideal, nadir)

        dist_matrix = cdist(F_norm, F_norm)
        np.fill_diagonal(dist_matrix, np.inf)
        d_i = dist_matrix.min(axis=1)
        d_mean = d_i.mean()

        return float(np.sqrt(np.sum((d_i - d_mean) ** 2) / (len(F) - 1)))

    def spread(self, df: pd.DataFrame) -> float:
        """计算Maximum Spread - 前沿延展度，越大越好"""
        F = self._prepare_data(df)
        ideal, nadir = F.min(axis=0), F.max(axis=0)
        F_norm = self._normalize(F, ideal, nadir)
        ranges = F_norm.max(axis=0) - F_norm.min(axis=0)
        return float(np.sqrt(np.sum(ranges ** 2)))

    def coverage(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[float, float]:
        """计算Coverage: A支配B的比例, B支配A的比例"""
        F_a = self._prepare_data(df_a)
        F_b = self._prepare_data(df_b)

        def dominates(p, q):
            return np.all(p <= q) and np.any(p < q)

        a_dom_b = sum(1 for b in F_b if any(dominates(a, b) for a in F_a))
        b_dom_a = sum(1 for a in F_a if any(dominates(b, a) for b in F_b))

        return (a_dom_b / len(F_b) * 100 if len(F_b) > 0 else 0,
                b_dom_a / len(F_a) * 100 if len(F_a) > 0 else 0)

    def contribution_to_combined_front(self, pareto_df: pd.DataFrame,
                                       baseline_dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """计算各方法对合并Pareto前沿的贡献 - 关键指标"""
        all_dfs = {'NSGA-III': pareto_df, **baseline_dfs}

        all_F = []
        sources = []
        for name, df in all_dfs.items():
            F = self._prepare_data(df)
            for f in F:
                all_F.append(f)
                sources.append(name)

        all_F = np.array(all_F)

        def is_dominated(p, others):
            for q in others:
                if np.all(q <= p) and np.any(q < p):
                    return True
            return False

        non_dom_idx = [i for i in range(len(all_F))
                       if not is_dominated(all_F[i], np.delete(all_F, i, axis=0))]

        contrib = {name: 0 for name in all_dfs}
        for idx in non_dom_idx:
            contrib[sources[idx]] += 1

        total = len(non_dom_idx)
        return {k: v / total * 100 if total > 0 else 0 for k, v in contrib.items()}

    def cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """计算Cliff's Delta效应量"""
        n_x, n_y = len(x), len(y)
        more = sum(1 for xi in x for yj in y if xi > yj)
        less = sum(1 for xi in x for yj in y if xi < yj)
        delta = (more - less) / (n_x * n_y)

        abs_d = abs(delta)
        if abs_d < 0.147:
            effect = "negligible"
        elif abs_d < 0.33:
            effect = "small"
        elif abs_d < 0.474:
            effect = "medium"
        else:
            effect = "large"

        return delta, effect


def compute_all_metrics(pareto_path: str, output_dir: str = './results/metrics'):
    """计算所有指标"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_df = pd.read_csv(pareto_path)
    pareto_dir = Path(pareto_path).parent

    baseline_dfs = {}
    for f in pareto_dir.glob('baseline_*.csv'):
        name = f.stem.replace('baseline_', '')
        df = pd.read_csv(f)
        if 'is_feasible' in df.columns:
            df = df[df['is_feasible']]
        baseline_dfs[name] = df

    metrics = MultiObjectiveMetrics()

    print("=" * 70)
    print("MULTI-OBJECTIVE OPTIMIZATION METRICS REPORT")
    print("=" * 70)

    # 参考点
    all_data = pd.concat([pareto_df] + list(baseline_dfs.values()))
    all_F = metrics._prepare_data(all_data)
    ref_point = all_F.max(axis=0) * 1.1

    # === 1. 质量指标 ===
    print("\n[1] PARETO FRONT QUALITY METRICS")
    print("-" * 50)
    print(f"{'Method':<12} {'HV':>12} {'Spacing':>10} {'Spread':>10} {'N':>6}")
    print("-" * 50)

    quality_results = []

    hv = metrics.hypervolume_2d(pareto_df, ref_point)
    sp = metrics.spacing(pareto_df)
    ms = metrics.spread(pareto_df)
    quality_results.append({'Method': 'NSGA-III', 'HV': hv, 'Spacing': sp, 'Spread': ms, 'N': len(pareto_df)})
    print(f"{'NSGA-III':<12} {hv:>12.2e} {sp:>10.4f} {ms:>10.4f} {len(pareto_df):>6}")

    for name, df in baseline_dfs.items():
        hv = metrics.hypervolume_2d(df, ref_point)
        sp = metrics.spacing(df)
        ms = metrics.spread(df)
        quality_results.append({'Method': name, 'HV': hv, 'Spacing': sp, 'Spread': ms, 'N': len(df)})
        print(f"{name:<12} {hv:>12.2e} {sp:>10.4f} {ms:>10.4f} {len(df):>6}")

    pd.DataFrame(quality_results).to_csv(output_dir / 'quality_metrics.csv', index=False)

    # === 2. 支配关系 ===
    print("\n[2] DOMINANCE COVERAGE (NSGA-III vs Baselines)")
    print("-" * 50)

    coverage_results = []
    for name, df in baseline_dfs.items():
        c_ab, c_ba = metrics.coverage(pareto_df, df)
        coverage_results.append({
            'Baseline': name, 'NSGA_Dominates_%': c_ab,
            'Baseline_Dominates_%': c_ba, 'Net_%': c_ab - c_ba
        })
        winner = "NSGA-III ✓" if c_ab > c_ba else name if c_ba > c_ab else "Tie"
        print(f"vs {name:<10}: NSGA {c_ab:5.1f}% | {name} {c_ba:5.1f}% → {winner}")

    pd.DataFrame(coverage_results).to_csv(output_dir / 'coverage_metrics.csv', index=False)

    # === 3. 贡献度 ===
    print("\n[3] CONTRIBUTION TO COMBINED PARETO FRONT")
    print("-" * 50)

    contrib = metrics.contribution_to_combined_front(pareto_df, baseline_dfs)
    for name, pct in sorted(contrib.items(), key=lambda x: -x[1]):
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"{name:<12}: {bar} {pct:5.1f}%")

    pd.DataFrame([{'Method': k, 'Contribution_%': v} for k, v in contrib.items()]).to_csv(
        output_dir / 'contribution_metrics.csv', index=False)

    # === 4. 统计检验 ===
    print("\n[4] STATISTICAL TESTS")
    print("-" * 50)

    nsga_costs = pareto_df['f1_total_cost_USD'].values
    nsga_recalls = pareto_df['detection_recall'].values

    stat_results = []
    for name, df in baseline_dfs.items():
        baseline_costs = df['f1_total_cost_USD'].values
        baseline_recalls = df['detection_recall'].values

        _, p_cost = stats.mannwhitneyu(nsga_costs, baseline_costs, alternative='two-sided')
        _, p_recall = stats.mannwhitneyu(nsga_recalls, baseline_recalls, alternative='two-sided')

        d_cost, eff_cost = metrics.cliffs_delta(baseline_costs, nsga_costs)
        d_recall, eff_recall = metrics.cliffs_delta(nsga_recalls, baseline_recalls)

        stat_results.append({
            'Comparison': f'vs {name}',
            'Cost_p': p_cost, 'Cost_d': d_cost, 'Cost_Effect': eff_cost,
            'Recall_p': p_recall, 'Recall_d': d_recall, 'Recall_Effect': eff_recall
        })

        sig_c = '***' if p_cost < 0.001 else '**' if p_cost < 0.01 else '*' if p_cost < 0.05 else 'ns'
        sig_r = '***' if p_recall < 0.001 else '**' if p_recall < 0.01 else '*' if p_recall < 0.05 else 'ns'

        print(f"vs {name:<10}: Cost p={p_cost:.4f}({sig_c}) δ={d_cost:+.2f}({eff_cost:<10})")
        print(f"             Recall p={p_recall:.4f}({sig_r}) δ={d_recall:+.2f}({eff_recall:<10})")

    pd.DataFrame(stat_results).to_csv(output_dir / 'statistical_tests.csv', index=False)

    # === 5. 高质量区域 ===
    print("\n[5] HIGH-QUALITY REGION (Recall ≥ 0.95)")
    print("-" * 50)

    hq_results = []
    for method, df in [('NSGA-III', pareto_df)] + list(baseline_dfs.items()):
        hq = df[df['detection_recall'] >= 0.95]
        min_cost = hq['f1_total_cost_USD'].min() / 1e6 if len(hq) > 0 else float('nan')
        hq_results.append({'Method': method, 'HQ_N': len(hq), 'HQ_MinCost_M': min_cost})
        print(f"{method:<12}: {len(hq):4d} solutions, Min Cost: ${min_cost:.3f}M")

    pd.DataFrame(hq_results).to_csv(output_dir / 'high_quality_region.csv', index=False)

    # === 保存汇总 ===
    summary = {
        'quality': quality_results,
        'coverage': coverage_results,
        'contribution': contrib,
        'statistical': stat_results,
        'high_quality': hq_results
    }

    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compute_metrics.py <pareto_csv> [output_dir]")
        sys.exit(1)

    compute_all_metrics(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else './results/metrics')
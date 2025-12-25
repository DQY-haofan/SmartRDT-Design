#!/usr/bin/env python3
"""
RMTwin 结果诊断脚本 (导师要求)
==============================
解释异常：100% dominance、算法多样性低、成本差异极端、recall集中

输出:
1) feasible-only 统计表
2) 6D coverage（双向，仅feasible）
3) 组件频率对比
4) 约束违反分布
5) 成本分解表
6) recall 分布 (median/p90)

Usage:
    python audit_results.py <run_dir> [output_dir]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys


# ============================================================================
# 核心诊断类
# ============================================================================

class ResultsAuditor:
    """结果诊断器 - 定位异常原因"""

    def __init__(self, run_dir: str, output_dir: str = None):
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir) if output_dir else self.run_dir / 'audit'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pareto_df = None
        self.baseline_dfs = {}
        self.all_dfs = {}

        # 约束列（根据你的config）
        self.constraint_cols = {
            'recall': ('detection_recall', '>=', 0.6),
            'latency': ('f3_latency_seconds', '<=', 500),
            'disruption': ('f4_traffic_disruption_hours', '<=', 300),
            'carbon': ('f5_carbon_emissions_kgCO2e_year', '<=', 300000),
            'mtbf': ('system_MTBF_hours', '>=', 1500),
            'budget': ('f1_total_cost_USD', '<=', 20000000),
        }

        # 目标列（6D）
        self.objective_cols = [
            'f1_total_cost_USD', 'f2_one_minus_recall', 'f3_latency_seconds',
            'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year',
            'f6_system_reliability_inverse_MTBF'
        ]

        # 成本分解列（如果存在）
        self.cost_breakdown_cols = [
            'sensor_capex', 'sensor_opex', 'comm_cost', 'storage_cost',
            'compute_cost', 'crew_cost', 'annotation_cost', 'retrain_cost'
        ]

    def load_data(self):
        """加载所有数据"""
        print("=" * 60)
        print("📂 加载数据")
        print("=" * 60)

        # Pareto
        pareto_path = self.run_dir / 'pareto_solutions.csv'
        if pareto_path.exists():
            self.pareto_df = pd.read_csv(pareto_path)
            self.all_dfs['NSGA-III'] = self.pareto_df
            print(f"  NSGA-III: {len(self.pareto_df)} solutions")

        # Baselines
        for f in self.run_dir.glob('baseline_*.csv'):
            name = f.stem.replace('baseline_', '')
            df = pd.read_csv(f)
            self.baseline_dfs[name] = df
            self.all_dfs[name.title()] = df
            print(f"  {name}: {len(df)} solutions")

    # =========================================================================
    # 1) Feasible-only 统计
    # =========================================================================
    def audit_feasibility(self) -> pd.DataFrame:
        """诊断1: 各方法可行解统计"""
        print("\n" + "=" * 60)
        print("📊 诊断1: Feasibility 统计")
        print("=" * 60)

        rows = []
        for name, df in self.all_dfs.items():
            n_total = len(df)

            if 'is_feasible' in df.columns:
                n_feasible = df['is_feasible'].sum()
            else:
                # 手动检查约束
                n_feasible = self._count_feasible(df)

            rows.append({
                'Method': name,
                'Total': n_total,
                'Feasible': n_feasible,
                'Feasible_Rate': n_feasible / n_total * 100 if n_total > 0 else 0
            })

        result_df = pd.DataFrame(rows)
        print(result_df.to_string(index=False))

        # 保存
        result_df.to_csv(self.output_dir / 'audit_1_feasibility.csv', index=False)

        # ⚠️ 警告检查
        for _, row in result_df.iterrows():
            if row['Feasible_Rate'] < 10:
                print(f"  ⚠️ {row['Method']}: 可行率仅 {row['Feasible_Rate']:.1f}%")

        return result_df

    def _count_feasible(self, df: pd.DataFrame) -> int:
        """手动检查约束"""
        feasible = np.ones(len(df), dtype=bool)

        for name, (col, op, threshold) in self.constraint_cols.items():
            if col not in df.columns:
                continue
            if op == '>=':
                feasible &= (df[col] >= threshold)
            elif op == '<=':
                feasible &= (df[col] <= threshold)

        return feasible.sum()

    # =========================================================================
    # 2) 6D Coverage（仅feasible）
    # =========================================================================
    def audit_coverage_6d(self) -> pd.DataFrame:
        """诊断2: 6D双向Coverage（仅在feasible上计算）"""
        print("\n" + "=" * 60)
        print("📊 诊断2: 6D Coverage (Feasible-Only)")
        print("=" * 60)

        # 获取feasible子集
        pareto_F = self._get_feasible_objectives('NSGA-III')

        if pareto_F is None or len(pareto_F) == 0:
            print("  ❌ NSGA-III 无可行解!")
            return pd.DataFrame()

        rows = []
        for name in self.baseline_dfs.keys():
            baseline_F = self._get_feasible_objectives(name.title())

            if baseline_F is None or len(baseline_F) == 0:
                rows.append({
                    'Baseline': name.title(),
                    'N_Feasible_Baseline': 0,
                    'N_Feasible_NSGA': len(pareto_F),
                    'C(NSGA,Baseline)': 'N/A',
                    'C(Baseline,NSGA)': 'N/A',
                    'Interpretation': 'Baseline has no feasible solutions'
                })
                continue

            # 计算双向coverage
            c_nsga_base = self._coverage(pareto_F, baseline_F)
            c_base_nsga = self._coverage(baseline_F, pareto_F)

            # 解释
            if c_nsga_base > 80 and c_base_nsga < 20:
                interp = "NSGA-III clearly dominates"
            elif c_base_nsga > 80 and c_nsga_base < 20:
                interp = "Baseline dominates (unusual!)"
            elif c_nsga_base > c_base_nsga:
                interp = "NSGA-III advantage"
            else:
                interp = "Mixed/Comparable"

            rows.append({
                'Baseline': name.title(),
                'N_Feasible_Baseline': len(baseline_F),
                'N_Feasible_NSGA': len(pareto_F),
                'C(NSGA,Baseline)': f"{c_nsga_base:.1f}%",
                'C(Baseline,NSGA)': f"{c_base_nsga:.1f}%",
                'Interpretation': interp
            })

        result_df = pd.DataFrame(rows)
        print(result_df.to_string(index=False))

        result_df.to_csv(self.output_dir / 'audit_2_coverage_6d.csv', index=False)

        # ⚠️ 警告
        for _, row in result_df.iterrows():
            if row['C(NSGA,Baseline)'] == '100.0%':
                print(f"  ⚠️ {row['Baseline']}: 100% 被支配 - 检查基线是否太弱!")

        return result_df

    def _get_feasible_objectives(self, method_name: str) -> np.ndarray:
        """获取feasible子集的目标矩阵"""
        df = self.all_dfs.get(method_name)
        if df is None:
            return None

        if 'is_feasible' in df.columns:
            df = df[df['is_feasible']]

        cols = [c for c in self.objective_cols if c in df.columns]
        if len(cols) < 6:
            print(f"  ⚠️ {method_name}: 只有 {len(cols)} 个目标列")

        if len(df) == 0:
            return None

        return df[cols].values

    def _coverage(self, A: np.ndarray, B: np.ndarray) -> float:
        """计算 C(A,B) = A支配B的比例"""
        dominated = 0
        for b in B:
            for a in A:
                if np.all(a <= b) and np.any(a < b):
                    dominated += 1
                    break
        return dominated / len(B) * 100

    # =========================================================================
    # 3) 组件频率对比
    # =========================================================================
    def audit_component_frequency(self) -> Dict[str, pd.DataFrame]:
        """诊断3: 组件类型出现频率"""
        print("\n" + "=" * 60)
        print("📊 诊断3: 组件频率对比")
        print("=" * 60)

        component_cols = ['sensor', 'algorithm', 'communication', 'storage', 'deployment']
        results = {}

        for col in component_cols:
            if col not in self.pareto_df.columns:
                continue

            freq_data = {}
            for name, df in self.all_dfs.items():
                if col in df.columns:
                    # 只看feasible
                    if 'is_feasible' in df.columns:
                        df = df[df['is_feasible']]

                    if len(df) > 0:
                        freq = df[col].value_counts(normalize=True) * 100
                        freq_data[name] = freq

            if freq_data:
                freq_df = pd.DataFrame(freq_data).fillna(0)
                results[col] = freq_df

                print(f"\n  [{col.upper()}] 频率 (%):")
                print(freq_df.round(1).to_string())

        # 保存
        if results:
            with pd.ExcelWriter(self.output_dir / 'audit_3_component_frequency.xlsx') as writer:
                for col, df in results.items():
                    df.to_excel(writer, sheet_name=col)

            # 也保存CSV
            for col, df in results.items():
                df.to_csv(self.output_dir / f'audit_3_{col}_frequency.csv')

        # ⚠️ 多样性警告
        if 'algorithm' in results:
            pareto_algos = results['algorithm'].get('NSGA-III', pd.Series())
            n_used = (pareto_algos > 0).sum()
            if n_used <= 2:
                print(f"\n  ⚠️ 算法多样性极低: 仅 {n_used} 种算法被选中!")
                print("     可能原因: 其他算法违反约束(latency/mtbf)被淘汰")

        return results

    # =========================================================================
    # 4) 约束违反分布
    # =========================================================================
    def audit_constraint_violations(self) -> pd.DataFrame:
        """诊断4: 各约束的违反分布"""
        print("\n" + "=" * 60)
        print("📊 诊断4: 约束违反分布")
        print("=" * 60)

        rows = []

        for method_name, df in self.all_dfs.items():
            row = {'Method': method_name, 'N_Total': len(df)}

            for constraint_name, (col, op, threshold) in self.constraint_cols.items():
                if col not in df.columns:
                    row[f'{constraint_name}_violation_rate'] = 'N/A'
                    continue

                if op == '>=':
                    violations = (df[col] < threshold).sum()
                else:
                    violations = (df[col] > threshold).sum()

                row[f'{constraint_name}_violation_rate'] = violations / len(df) * 100
                row[f'{constraint_name}_violations'] = violations

            rows.append(row)

        result_df = pd.DataFrame(rows)

        # 只显示violation rate列
        display_cols = ['Method', 'N_Total'] + [c for c in result_df.columns if 'violation_rate' in c]
        print(result_df[display_cols].round(1).to_string(index=False))

        result_df.to_csv(self.output_dir / 'audit_4_constraint_violations.csv', index=False)

        # ⚠️ 找出主要杀手约束
        for method_name, df in self.all_dfs.items():
            if method_name == 'NSGA-III':
                continue

            max_viol = 0
            killer = None
            for constraint_name, (col, op, threshold) in self.constraint_cols.items():
                if col not in df.columns:
                    continue
                if op == '>=':
                    viol = (df[col] < threshold).sum() / len(df) * 100
                else:
                    viol = (df[col] > threshold).sum() / len(df) * 100

                if viol > max_viol:
                    max_viol = viol
                    killer = constraint_name

            if killer and max_viol > 50:
                print(f"  ⚠️ {method_name}: '{killer}' 约束违反率 {max_viol:.0f}%")

        return result_df

    # =========================================================================
    # 5) 成本分解
    # =========================================================================
    def audit_cost_breakdown(self) -> pd.DataFrame:
        """诊断5: 成本分解统计"""
        print("\n" + "=" * 60)
        print("📊 诊断5: 成本分解")
        print("=" * 60)

        # 检查是否有成本分解列
        available_cols = [c for c in self.cost_breakdown_cols if c in self.pareto_df.columns]

        if not available_cols:
            print("  ⚠️ 没有成本分解列，需要在 evaluation.py 中添加输出")
            print("     建议添加: sensor_capex, sensor_opex, comm_cost, storage_cost, etc.")

            # 至少显示总成本统计
            rows = []
            for method_name, df in self.all_dfs.items():
                if 'is_feasible' in df.columns:
                    df = df[df['is_feasible']]

                if len(df) > 0 and 'f1_total_cost_USD' in df.columns:
                    rows.append({
                        'Method': method_name,
                        'N_Feasible': len(df),
                        'Cost_Min_M': df['f1_total_cost_USD'].min() / 1e6,
                        'Cost_Max_M': df['f1_total_cost_USD'].max() / 1e6,
                        'Cost_Mean_M': df['f1_total_cost_USD'].mean() / 1e6,
                        'Cost_Std_M': df['f1_total_cost_USD'].std() / 1e6,
                    })

            result_df = pd.DataFrame(rows)
            print(result_df.round(3).to_string(index=False))
            result_df.to_csv(self.output_dir / 'audit_5_cost_summary.csv', index=False)
            return result_df

        # 有成本分解列
        rows = []
        for method_name, df in self.all_dfs.items():
            if 'is_feasible' in df.columns:
                df = df[df['is_feasible']]

            if len(df) == 0:
                continue

            row = {'Method': method_name, 'N_Feasible': len(df)}
            for col in available_cols:
                if col in df.columns:
                    row[f'{col}_mean'] = df[col].mean()
                    row[f'{col}_pct'] = df[col].mean() / df['f1_total_cost_USD'].mean() * 100

            rows.append(row)

        result_df = pd.DataFrame(rows)
        print(result_df.round(2).to_string(index=False))

        result_df.to_csv(self.output_dir / 'audit_5_cost_breakdown.csv', index=False)
        return result_df

    # =========================================================================
    # 6) Recall 分布
    # =========================================================================
    def audit_recall_distribution(self) -> pd.DataFrame:
        """诊断6: Recall分布 (median/p90 替代 max)"""
        print("\n" + "=" * 60)
        print("📊 诊断6: Recall 分布 (Feasible-Only)")
        print("=" * 60)

        rows = []
        for method_name, df in self.all_dfs.items():
            if 'is_feasible' in df.columns:
                df = df[df['is_feasible']]

            if len(df) == 0 or 'detection_recall' not in df.columns:
                continue

            recall = df['detection_recall']
            rows.append({
                'Method': method_name,
                'N_Feasible': len(df),
                'Recall_Min': recall.min(),
                'Recall_P25': recall.quantile(0.25),
                'Recall_Median': recall.median(),
                'Recall_P75': recall.quantile(0.75),
                'Recall_P90': recall.quantile(0.90),
                'Recall_Max': recall.max(),
            })

        result_df = pd.DataFrame(rows)
        print(result_df.round(4).to_string(index=False))

        result_df.to_csv(self.output_dir / 'audit_6_recall_distribution.csv', index=False)

        # ⚠️ 警告
        if len(result_df) > 0:
            max_vals = result_df['Recall_Max'].values
            if len(set(max_vals.round(3))) == 1:
                print(f"  ⚠️ 所有方法 Max Recall 相同 ({max_vals[0]:.3f}) - 可能存在 clip/上限")

        return result_df

    # =========================================================================
    # 综合诊断
    # =========================================================================
    def run_full_audit(self):
        """运行完整诊断"""
        print("\n" + "=" * 70)
        print("🔍 RMTwin 结果诊断 (导师要求)")
        print("=" * 70)

        self.load_data()

        results = {
            'feasibility': self.audit_feasibility(),
            'coverage_6d': self.audit_coverage_6d(),
            'component_freq': self.audit_component_frequency(),
            'constraint_violations': self.audit_constraint_violations(),
            'cost_breakdown': self.audit_cost_breakdown(),
            'recall_distribution': self.audit_recall_distribution(),
        }

        # 生成诊断摘要
        self._generate_summary(results)

        print("\n" + "=" * 70)
        print(f"✅ 诊断完成! 结果保存到: {self.output_dir}")
        print("=" * 70)

        return results

    def _generate_summary(self, results: Dict):
        """生成诊断摘要"""
        print("\n" + "=" * 60)
        print("📋 诊断摘要与建议")
        print("=" * 60)

        issues = []

        # 检查100% dominance
        cov_df = results.get('coverage_6d')
        if cov_df is not None and len(cov_df) > 0:
            for _, row in cov_df.iterrows():
                if row['C(NSGA,Baseline)'] == '100.0%':
                    issues.append(f"⚠️ {row['Baseline']}: 100%被支配 → 检查基线是否太弱")

        # 检查多样性
        comp_freq = results.get('component_freq', {})
        if 'algorithm' in comp_freq:
            pareto_algos = comp_freq['algorithm'].get('NSGA-III', pd.Series())
            n_used = (pareto_algos > 0).sum()
            if n_used <= 2:
                issues.append(f"⚠️ 算法多样性: 仅{n_used}种 → 检查约束是否淘汰其他算法")

        # 检查成本差异
        cost_df = results.get('cost_breakdown')
        if cost_df is not None and len(cost_df) > 1:
            costs = cost_df[cost_df['N_Feasible'] > 0]['Cost_Min_M'].values
            if len(costs) > 1:
                ratio = costs.max() / costs.min() if costs.min() > 0 else float('inf')
                if ratio > 10:
                    issues.append(f"⚠️ 成本差异: {ratio:.0f}倍 → 检查成本模型/规模因子")

        # 检查recall clip
        recall_df = results.get('recall_distribution')
        if recall_df is not None and len(recall_df) > 0:
            max_vals = recall_df['Recall_Max'].values
            if len(set(np.round(max_vals, 3))) == 1:
                issues.append(f"⚠️ Recall上限: {max_vals[0]:.3f} → 可能存在clip")

        if issues:
            print("\n发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n  ✅ 未发现明显异常")

        # 保存摘要
        summary = {
            'issues': issues,
            'n_issues': len(issues),
            'recommendation': '请根据上述诊断结果调整基线强度或检查成本模型'
        }

        with open(self.output_dir / 'audit_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================================
# 命令行接口
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_results.py <run_dir> [output_dir]")
        print("\nExample:")
        print("  python audit_results.py ./results/runs/20241225_123456_seed42")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    auditor = ResultsAuditor(run_dir, output_dir)
    auditor.run_full_audit()


if __name__ == '__main__':
    main()
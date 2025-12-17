#!/usr/bin/env python3
"""
Unified Visualization Module for RMTwin
========================================
All visualization + paper results in one file.

Usage:
    # Called by main.py automatically
    from visualization import Visualizer
    viz = Visualizer(config)
    viz.create_all_figures(pareto_df, baseline_dfs, history)

    # Standalone: basic figures
    python visualization.py --pareto ./results/pareto_solutions_6obj_fixed.csv

    # Standalone: paper figures + tables
    python visualization.py --pareto ./results/pareto_solutions_6obj_fixed.csv --paper

Author: RMTwin Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import logging
import json
import argparse
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# Settings
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

SINGLE_COL, DOUBLE_COL = 3.5, 7.16

COLORS = {
    'nsga3': '#2E86AB', 'random': '#A23B72', 'grid': '#F18F01',
    'weighted': '#C73E1D', 'expert': '#3A7D44', 'pareto': '#2E86AB',
}

METHOD_COLORS = {
    'NSGA-III': '#2E86AB', 'Random': '#A23B72', 'Grid': '#F18F01',
    'Weighted': '#C73E1D', 'WeightedSum': '#C73E1D', 'Expert': '#3A7D44',
}

METHOD_MARKERS = {
    'NSGA-III': 'o', 'Random': 's', 'Grid': '^',
    'Weighted': 'D', 'WeightedSum': 'D', 'Expert': 'v',
}

VARIABLE_LABELS = {
    'sensor': 'Sensor', 'algorithm': 'Algorithm', 'communication': 'Comm',
    'deployment': 'Deploy', 'data_rate': 'Data Rate', 'detection_threshold': 'Threshold',
    'crew_size': 'Crew', 'inspection_cycle': 'Cycle',
}


# ============================================================================
# Metrics
# ============================================================================
class MetricsCalculator:
    @staticmethod
    def calculate_feasible_rate(df):
        if 'is_feasible' not in df.columns:
            return 1.0
        return df['is_feasible'].sum() / len(df) if len(df) > 0 else 0.0

    @staticmethod
    def vargha_delaney_a12(x, y):
        m, n = len(x), len(y)
        r = sum(1 if xi > yi else 0.5 if xi == yi else 0 for xi in x for yi in y)
        return r / (m * n)

    @staticmethod
    def select_representative_solutions(pareto_df):
        selected = []
        obj_cols = [c for c in ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                                'f5_carbon_emissions_kgCO2e_year'] if c in pareto_df.columns]

        extremes = {'f1_total_cost_USD': ('Min Cost', 'min'), 'detection_recall': ('Max Recall', 'max'),
                    'f3_latency_seconds': ('Min Latency', 'min'),
                    'f5_carbon_emissions_kgCO2e_year': ('Min Carbon', 'min')}

        for col in obj_cols:
            if col in extremes:
                label, d = extremes[col]
                idx = pareto_df[col].idxmin() if d == 'min' else pareto_df[col].idxmax()
                row = pareto_df.loc[idx].copy()
                row['solution_type'] = label
                selected.append(row)

        # Knee point
        if len(pareto_df) > 0 and len(obj_cols) > 0:
            norm = pareto_df[obj_cols].copy()
            for c in obj_cols:
                rng = norm[c].max() - norm[c].min()
                if rng > 0:
                    norm[c] = (norm[c] - norm[c].min()) / rng if c != 'detection_recall' else 1 - (
                                norm[c] - norm[c].min()) / rng
            distances = np.sqrt((norm ** 2).sum(axis=1))
            knee = pareto_df.loc[distances.idxmin()].copy()
            knee['solution_type'] = 'Balanced'
            selected.append(knee)

        return pd.DataFrame(selected).drop_duplicates(subset=obj_cols[:2], keep='first') if selected else pd.DataFrame()


# ============================================================================
# Visualizer
# ============================================================================
class Visualizer:
    def __init__(self, config, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(config.output_dir) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.metrics = MetricsCalculator()

    def create_all_figures(self, pareto_results, baseline_results=None, optimization_history=None):
        """Main entry: generate all figures."""
        logger.info("Generating figures...")

        if optimization_history:
            self.plot_convergence(optimization_history)

        self.plot_pareto_2d_main(pareto_results, baseline_results)
        self.plot_pareto_3d(pareto_results)
        self.plot_parallel_coordinates(pareto_results)
        self.plot_decision_distribution(pareto_results)
        self.create_all_2d_pareto_fronts(pareto_results)

        if baseline_results:
            self.create_enhanced_baseline_comparison(pareto_results, baseline_results)

        self.plot_technology_impact(pareto_results)
        logger.info(f"Figures saved to {self.output_dir}")

    def create_enhanced_baseline_comparison(self, pareto_df, baseline_dfs):
        self.plot_method_summary(pareto_df, baseline_dfs)
        self.plot_cost_recall_comparison(pareto_df, baseline_dfs)
        self.plot_feasibility_comparison(pareto_df, baseline_dfs)

    # ========== Core Plots ==========
    def plot_convergence(self, history, save_name='convergence.pdf'):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3))
        conv = history.get('convergence', {})

        if conv.get('hv'):
            gens = range(1, len(conv['hv']) + 1)
            axes[0].plot(gens, conv['hv'], '-o', color=COLORS['nsga3'], markersize=3)
            axes[0].set_xlabel('Generation');
            axes[0].set_ylabel('Hypervolume')
            axes[0].set_title('(a) HV Convergence', fontsize=10)

        if conv.get('n_nds'):
            axes[1].plot(gens, conv['n_nds'], '-s', color=COLORS['grid'], markersize=3)
            axes[1].set_xlabel('Generation');
            axes[1].set_ylabel('Pareto Size')
            axes[1].set_title('(b) Pareto Front Size', fontsize=10)

        plt.tight_layout()
        self._save(fig, save_name)

    def plot_pareto_2d_main(self, df, baselines=None, save_name='pareto_2d_main.pdf'):
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 3.5))
        x, y = df['f1_total_cost_USD'] / 1e6, df['detection_recall']

        if 'f3_latency_seconds' in df.columns:
            sc = ax.scatter(x, y, c=df['f3_latency_seconds'], cmap='viridis', s=60, alpha=0.8)
            fig.colorbar(sc, ax=ax, shrink=0.8, label='Latency (s)')
        else:
            ax.scatter(x, y, c=COLORS['pareto'], s=60, alpha=0.8)

        ax.plot(x.iloc[np.argsort(x)], y.iloc[np.argsort(x)], '--', color=COLORS['pareto'], alpha=0.5)

        if baselines:
            for name, bdf in baselines.items():
                if bdf is not None and len(bdf) > 0:
                    feas = bdf[bdf['is_feasible']] if 'is_feasible' in bdf.columns else bdf
                    if len(feas) > 0:
                        ax.scatter(feas['f1_total_cost_USD'] / 1e6, feas['detection_recall'],
                                   c=METHOD_COLORS.get(name.title(), '#999'),
                                   marker=METHOD_MARKERS.get(name.title(), 's'),
                                   s=30, alpha=0.5, label=name.title())

        ax.set_xlabel('Cost (M$)');
        ax.set_ylabel('Recall')
        if baselines: ax.legend(loc='lower right', fontsize=8)
        self._save(fig, save_name)

    def plot_pareto_3d(self, df, save_name='pareto_3d.pdf'):
        fig = plt.figure(figsize=(SINGLE_COL * 2, 4))
        ax = fig.add_subplot(111, projection='3d')
        x, y = df['f1_total_cost_USD'] / 1e6, df['detection_recall']
        z = df.get('f5_carbon_emissions_kgCO2e_year', df.get('f3_latency_seconds', np.zeros(len(df)))) / 1000

        ax.scatter(x, y, z, c=COLORS['pareto'], s=50, alpha=0.8)
        ax.set_xlabel('Cost (M$)');
        ax.set_ylabel('Recall');
        ax.set_zlabel('Carbon (t)')
        ax.view_init(20, 45)
        self._save(fig, save_name)

    def plot_parallel_coordinates(self, df, save_name='parallel_coords.pdf'):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
        cols = [c for c in ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds',
                            'f5_carbon_emissions_kgCO2e_year'] if c in df.columns][:6]

        norm = df[cols].copy()
        for c in cols:
            rng = norm[c].max() - norm[c].min()
            if rng > 0:
                norm[c] = (norm[c] - norm[c].min()) / rng if c != 'detection_recall' else 1 - (
                            norm[c] - norm[c].min()) / rng

        colors = plt.cm.viridis((df['f1_total_cost_USD'] - df['f1_total_cost_USD'].min()) /
                                (df['f1_total_cost_USD'].max() - df['f1_total_cost_USD'].min() + 1e-10))

        for i in range(len(norm)):
            ax.plot(range(len(cols)), norm.iloc[i].values, c=colors[i], alpha=0.4, linewidth=1)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(['Cost', 'Recall', 'Latency', 'Carbon'][:len(cols)], fontsize=9)
        ax.set_ylabel('Normalized (1=Best)')
        ax.set_ylim(-0.05, 1.05)
        self._save(fig, save_name)

    def plot_decision_distribution(self, df, save_name='decision_dist.pdf'):
        cols = [c for c in ['sensor', 'algorithm', 'communication', 'deployment'] if c in df.columns]
        if not cols: return

        fig, axes = plt.subplots(1, len(cols), figsize=(DOUBLE_COL, 2.5))
        axes = [axes] if len(cols) == 1 else axes

        for ax, col in zip(axes, cols):
            counts = df[col].value_counts()
            ax.bar(range(len(counts)), counts.values, color=COLORS['pareto'], alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:8] for x in counts.index], rotation=45, ha='right', fontsize=7)
            ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=9)

        plt.tight_layout()
        self._save(fig, save_name)

    def create_all_2d_pareto_fronts(self, df):
        objs = [('f1_total_cost_USD', 'Cost', 1000), ('detection_recall', 'Recall', 1),
                ('f3_latency_seconds', 'Latency', 1)]
        if 'f5_carbon_emissions_kgCO2e_year' in df.columns:
            objs.append(('f5_carbon_emissions_kgCO2e_year', 'Carbon', 1000))

        n = 0
        for i, (c1, l1, s1) in enumerate(objs):
            for c2, l2, s2 in objs[i + 1:]:
                n += 1
                fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
                ax.scatter(df[c1] / s1, df[c2] / s2, c=COLORS['pareto'], s=40, alpha=0.7)
                ax.set_xlabel(l1);
                ax.set_ylabel(l2)
                self._save(fig, f'pareto_2d_{n:02d}')

    def plot_technology_impact(self, df, save_name='tech_impact.pdf'):
        cols = [c for c in ['sensor', 'algorithm', 'communication', 'deployment'] if c in df.columns]
        if not cols: return

        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, 2.5))
        axes = [axes] if n == 1 else axes

        for ax, col in zip(axes, cols):
            counts = df[col].value_counts()
            ax.bar(range(len(counts)), counts.values, color=COLORS['pareto'], alpha=0.8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:8] for x in counts.index], rotation=45, ha='right', fontsize=7)
            ax.set_title(col.title(), fontsize=9)

        plt.tight_layout()
        self._save(fig, save_name)

    # ========== Baseline Comparisons ==========
    def plot_method_summary(self, pareto_df, baselines, save_name='method_summary.pdf'):
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5))

        methods = ['NSGA-III']
        metrics = {'NSGA-III': {'n': len(pareto_df), 'cost': pareto_df['f1_total_cost_USD'].min() / 1e6,
                                'recall': pareto_df['detection_recall'].max()}}

        for name, df in baselines.items():
            if df is not None and len(df) > 0:
                methods.append(name.title())
                feas = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                metrics[name.title()] = {
                    'n': len(feas),
                    'cost': feas['f1_total_cost_USD'].min() / 1e6 if len(feas) > 0 else np.nan,
                    'recall': feas['detection_recall'].max() if len(feas) > 0 else np.nan
                }

        colors = [METHOD_COLORS.get(m, '#999') for m in methods]

        # (a) Solutions
        vals = [metrics[m]['n'] for m in methods]
        axes[0, 0].bar(methods, vals, color=colors, alpha=0.8)
        axes[0, 0].set_ylabel('Feasible');
        axes[0, 0].set_title('(a) Solutions', fontsize=10)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # (b) Cost
        vals = [metrics[m]['cost'] for m in methods]
        axes[0, 1].bar(methods, vals, color=colors, alpha=0.8)
        axes[0, 1].set_ylabel('Min Cost (M$)');
        axes[0, 1].set_title('(b) Cost', fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # (c) Recall
        vals = [metrics[m]['recall'] for m in methods]
        axes[1, 0].bar(methods, vals, color=colors, alpha=0.8)
        axes[1, 0].set_ylabel('Max Recall');
        axes[1, 0].set_title('(c) Recall', fontsize=10)
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # (d) Scatter
        for name, df in [('NSGA-III', pareto_df)] + list(baselines.items()):
            if df is not None and len(df) > 0:
                data = df if name == 'NSGA-III' else (df[df['is_feasible']] if 'is_feasible' in df.columns else df)
                if len(data) > 0:
                    n = name.title() if name != 'NSGA-III' else name
                    axes[1, 1].scatter(data['f1_total_cost_USD'] / 1e6, data['detection_recall'],
                                       c=METHOD_COLORS.get(n, '#999'), marker=METHOD_MARKERS.get(n, 'o'),
                                       s=40 if name == 'NSGA-III' else 20, alpha=0.7, label=n)
        axes[1, 1].set_xlabel('Cost (M$)');
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('(d) Trade-off', fontsize=10)
        axes[1, 1].legend(fontsize=7)

        plt.tight_layout()
        self._save(fig, save_name)

    def plot_cost_recall_comparison(self, pareto_df, baselines, save_name='cost_recall_cmp.pdf'):
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 3.5))

        ax.scatter(pareto_df['f1_total_cost_USD'] / 1e6, pareto_df['detection_recall'],
                   c=COLORS['nsga3'], s=60, alpha=0.8, label='NSGA-III')

        for name, df in baselines.items():
            if df is not None and len(df) > 0:
                feas = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feas) > 0:
                    ax.scatter(feas['f1_total_cost_USD'] / 1e6, feas['detection_recall'],
                               c=METHOD_COLORS.get(name.title(), '#999'), marker=METHOD_MARKERS.get(name.title(), 's'),
                               s=30, alpha=0.5, label=name.title())

        ax.set_xlabel('Cost (M$)');
        ax.set_ylabel('Recall')
        ax.legend(fontsize=8)
        self._save(fig, save_name)

    def plot_feasibility_comparison(self, pareto_df, baselines, save_name='feasibility_cmp.pdf'):
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3))

        methods, rates = ['NSGA-III'], [100.0]
        for name, df in baselines.items():
            if df is not None and len(df) > 0:
                methods.append(name.title())
                rates.append(self.metrics.calculate_feasible_rate(df) * 100)

        colors = [METHOD_COLORS.get(m, '#999') for m in methods]
        ax.bar(methods, rates, color=colors, alpha=0.8)
        ax.set_ylabel('Feasible Rate (%)')
        ax.set_ylim(0, 110)
        ax.tick_params(axis='x', rotation=45)
        self._save(fig, save_name)

    # ========== Paper Results ==========
    def generate_paper_results(self, pareto_df, baselines=None, history=None):
        """Generate paper-ready figures + tables."""
        logger.info("=" * 50)
        logger.info("Generating Paper Results")
        logger.info("=" * 50)

        paper_dir = self.output_dir.parent / 'paper'
        (paper_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (paper_dir / 'tables').mkdir(parents=True, exist_ok=True)

        # Figures
        orig = self.output_dir
        self.output_dir = paper_dir / 'figures'
        self.create_all_figures(pareto_df, baselines, history)
        self.output_dir = orig

        # Tables
        self._gen_representative_table(pareto_df, paper_dir / 'tables')
        self._gen_comparison_table(pareto_df, baselines, paper_dir / 'tables')
        self._gen_statistical_tests(pareto_df, baselines, paper_dir / 'tables')
        self._gen_summary_json(pareto_df, baselines, paper_dir)

        logger.info(f"Paper results: {paper_dir}")

    def _gen_representative_table(self, df, out_dir):
        reps = self.metrics.select_representative_solutions(df)
        if len(reps) == 0: return

        data = [{'Type': r.get('solution_type', '?'), 'Cost($M)': f"{r['f1_total_cost_USD'] / 1e6:.2f}",
                 'Recall': f"{r['detection_recall']:.4f}", 'Latency': f"{r['f3_latency_seconds']:.1f}"}
                for _, r in reps.iterrows()]

        tdf = pd.DataFrame(data)
        tdf.to_csv(out_dir / 'representative_solutions.csv', index=False)
        with open(out_dir / 'representative_solutions.tex', 'w') as f:
            f.write(tdf.to_latex(index=False))
        logger.info(f"Representative table: {len(data)} solutions")

    def _gen_comparison_table(self, pareto_df, baselines, out_dir):
        data = [{'Method': 'NSGA-III', 'Feasible': len(pareto_df),
                 'MinCost': f"{pareto_df['f1_total_cost_USD'].min() / 1e6:.2f}",
                 'MaxRecall': f"{pareto_df['detection_recall'].max():.4f}"}]

        if baselines:
            for name, df in baselines.items():
                if df is not None and len(df) > 0:
                    feas = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                    data.append({'Method': name.title(), 'Feasible': len(feas),
                                 'MinCost': f"{feas['f1_total_cost_USD'].min() / 1e6:.2f}" if len(feas) > 0 else 'N/A',
                                 'MaxRecall': f"{feas['detection_recall'].max():.4f}" if len(feas) > 0 else 'N/A'})

        tdf = pd.DataFrame(data)
        tdf.to_csv(out_dir / 'method_comparison.csv', index=False)
        with open(out_dir / 'method_comparison.tex', 'w') as f:
            f.write(tdf.to_latex(index=False))
        logger.info("Comparison table generated")

    def _gen_statistical_tests(self, pareto_df, baselines, out_dir):
        if not baselines: return
        tests = []
        pr = pareto_df['detection_recall'].values

        for name, df in baselines.items():
            if df is not None and len(df) > 0:
                feas = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feas) > 10:
                    br = feas['detection_recall'].values
                    stat, p = stats.mannwhitneyu(pr, br, alternative='greater')
                    a12 = self.metrics.vargha_delaney_a12(pr, br)
                    tests.append({'Comparison': f'NSGA-III vs {name.title()}', 'U': f"{stat:.0f}",
                                  'p-value': f"{p:.4f}" if p >= 0.0001 else "<0.0001",
                                  'A12': f"{a12:.3f}", 'Sig': 'Yes' if p < 0.05 else 'No'})

        if tests:
            tdf = pd.DataFrame(tests)
            tdf.to_csv(out_dir / 'statistical_tests.csv', index=False)
            with open(out_dir / 'statistical_tests.tex', 'w') as f:
                f.write(tdf.to_latex(index=False))
            logger.info(f"Statistical tests: {len(tests)}")

    def _gen_summary_json(self, pareto_df, baselines, out_dir):
        summary = {
            'generated': datetime.now().isoformat(),
            'pareto': {'count': len(pareto_df),
                       'cost_range': [float(pareto_df['f1_total_cost_USD'].min()),
                                      float(pareto_df['f1_total_cost_USD'].max())],
                       'recall_range': [float(pareto_df['detection_recall'].min()),
                                        float(pareto_df['detection_recall'].max())]},
            'baselines': {
                n: {'total': len(d), 'feasible': int(d['is_feasible'].sum()) if 'is_feasible' in d.columns else len(d)}
                for n, d in (baselines or {}).items() if d is not None}
        }
        with open(out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def _save(self, fig, name):
        for fmt in ['pdf', 'png']:
            fig.savefig(self.output_dir / f'{name}.{fmt}', dpi=300 if fmt == 'png' else None)
        plt.close(fig)
        logger.info(f"Saved: {name}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='RMTwin Visualization')
    parser.add_argument('--pareto', default='./results/pareto_solutions_6obj_fixed.csv')
    parser.add_argument('--baselines', default='./results')
    parser.add_argument('--output', default='./results')
    parser.add_argument('--paper', action='store_true', help='Generate paper results with tables')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    pareto_df = pd.read_csv(args.pareto)

    baselines = {}
    bdir = Path(args.baselines)
    for m in ['random', 'grid', 'weighted', 'expert']:
        p = bdir / f'baseline_{m}.csv'
        if p.exists():
            baselines[m] = pd.read_csv(p)

    class Cfg:
        output_dir = args.output

    viz = Visualizer(Cfg())

    if args.paper:
        viz.generate_paper_results(pareto_df, baselines)
    else:
        viz.create_all_figures(pareto_df, baselines)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Paper Results Generator for RMTwin
====================================
One-click generation of publication-quality figures and statistical tables.

Based on expert recommendations:
- Group A: Algorithm proof (HV convergence, feasible rate, quality metrics)
- Group B: Pareto structure (2D plots, parallel coords, radar charts)
- Group C: Ontology ablation

Output:
- results/paper/figures/*.pdf + *.png
- results/paper/tables/*.csv (optional *.tex)

Author: RMTwin Research Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import argparse
from datetime import datetime

# Import visualization module
from visualization import Visualizer, MetricsCalculator, run_sensitivity_analysis

logger = logging.getLogger(__name__)


class PaperResultsGenerator:
    """
    Generate all figures and tables needed for publication.
    """
    
    def __init__(self, results_dir: str = './results', output_dir: str = None):
        """
        Initialize generator.
        
        Args:
            results_dir: Directory containing optimization results
            output_dir: Output directory (default: results_dir/paper)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / 'paper'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        
        self.metrics = MetricsCalculator()
        
        # Load data
        self.pareto_df = None
        self.baseline_dfs = {}
        self.history = None
        self.final_population = None
        
    def load_data(self):
        """Load all result files."""
        logger.info("Loading optimization results...")
        
        # Load Pareto solutions
        pareto_path = self.results_dir / 'pareto_solutions_6obj_fixed.csv'
        if pareto_path.exists():
            self.pareto_df = pd.read_csv(pareto_path)
            logger.info(f"Loaded {len(self.pareto_df)} Pareto solutions")
        else:
            raise FileNotFoundError(f"Pareto solutions not found: {pareto_path}")
        
        # Load final population (if exists)
        pop_path = self.results_dir / 'final_population.csv'
        if pop_path.exists():
            self.final_population = pd.read_csv(pop_path)
            logger.info(f"Loaded {len(self.final_population)} final population")
        
        # Load baselines
        for method in ['random', 'grid', 'weighted', 'expert']:
            baseline_path = self.results_dir / f'baseline_{method}.csv'
            if baseline_path.exists():
                self.baseline_dfs[method] = pd.read_csv(baseline_path)
                logger.info(f"Loaded {method} baseline: {len(self.baseline_dfs[method])} solutions")
        
        # Load history (if exists)
        history_path = self.results_dir / 'optimization_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            logger.info("Loaded optimization history")
    
    def generate_all(self):
        """Generate all figures and tables."""
        logger.info("=" * 60)
        logger.info("Generating Publication Results")
        logger.info("=" * 60)
        
        # Load data
        self.load_data()
        
        # Generate figures
        self._generate_figures()
        
        # Generate tables
        self._generate_tables()
        
        # Generate summary statistics
        self._generate_statistics()
        
        logger.info("=" * 60)
        logger.info(f"All results saved to: {self.output_dir}")
        logger.info("=" * 60)
    
    def _generate_figures(self):
        """Generate all publication figures."""
        logger.info("\n--- Generating Figures ---")
        
        # Create config-like object for visualizer
        class Config:
            output_dir = str(self.output_dir / 'figures')
            population_size = 100
        
        config = Config()
        viz = Visualizer(config, output_dir=str(self.output_dir / 'figures'))
        
        # Generate all figures
        viz.create_all_figures(
            self.pareto_df,
            self.baseline_dfs,
            self.history
        )
        
        logger.info(f"Figures saved to: {self.output_dir / 'figures'}")
    
    def _generate_tables(self):
        """Generate all publication tables."""
        logger.info("\n--- Generating Tables ---")
        
        # Table 1: Representative solutions
        self._generate_representative_solutions_table()
        
        # Table 2: Method comparison summary
        self._generate_method_comparison_table()
        
        # Table 3: Statistical tests
        self._generate_statistical_tests_table()
        
        logger.info(f"Tables saved to: {self.output_dir / 'tables'}")
    
    def _generate_representative_solutions_table(self):
        """Generate table of representative Pareto solutions."""
        logger.info("Generating representative solutions table...")
        
        representatives = self.metrics.select_representative_solutions(self.pareto_df)
        
        if len(representatives) == 0:
            logger.warning("No representative solutions found")
            return
        
        # Select columns for table
        obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']
        var_cols = ['sensor', 'algorithm', 'communication', 'deployment']
        
        # Add optional columns
        for col in ['f4_traffic_disruption_hours', 'f4_traffic_disruption_hours_year',
                   'f5_carbon_emissions_kgCO2e_year', 'system_MTBF_hours']:
            if col in representatives.columns:
                obj_cols.append(col)
        
        var_cols = [c for c in var_cols if c in representatives.columns]
        
        # Create formatted table
        table_data = []
        for _, row in representatives.iterrows():
            row_dict = {
                'Solution Type': row.get('solution_type', 'Unknown'),
                'Cost ($M)': f"{row['f1_total_cost_USD'] / 1e6:.2f}",
                'Recall': f"{row['detection_recall']:.4f}",
                'Latency (s)': f"{row['f3_latency_seconds']:.1f}",
            }
            
            if 'f5_carbon_emissions_kgCO2e_year' in row:
                row_dict['Carbon (t)'] = f"{row['f5_carbon_emissions_kgCO2e_year'] / 1000:.1f}"
            
            for col in var_cols:
                row_dict[col.title()] = str(row[col])[:15]
            
            table_data.append(row_dict)
        
        table_df = pd.DataFrame(table_data)
        
        # Save CSV
        table_df.to_csv(self.output_dir / 'tables' / 'representative_solutions.csv', index=False)
        
        # Save LaTeX
        latex_str = table_df.to_latex(index=False, escape=True)
        with open(self.output_dir / 'tables' / 'representative_solutions.tex', 'w') as f:
            f.write(latex_str)
        
        logger.info(f"Generated table with {len(table_df)} representative solutions")
    
    def _generate_method_comparison_table(self):
        """Generate method comparison summary table."""
        logger.info("Generating method comparison table...")
        
        methods = ['NSGA-III']
        data = []
        
        # NSGA-III metrics
        nsga_row = {
            'Method': 'NSGA-III',
            'Total Solutions': len(self.pareto_df),
            'Feasible Solutions': len(self.pareto_df),
            'Feasible Rate (%)': 100.0,
            'Min Cost ($M)': self.pareto_df['f1_total_cost_USD'].min() / 1e6,
            'Max Recall': self.pareto_df['detection_recall'].max(),
            'Mean Recall': self.pareto_df['detection_recall'].mean(),
        }
        
        if 'f5_carbon_emissions_kgCO2e_year' in self.pareto_df.columns:
            nsga_row['Min Carbon (t)'] = self.pareto_df['f5_carbon_emissions_kgCO2e_year'].min() / 1000
        
        data.append(nsga_row)
        
        # Baseline metrics
        for name, df in self.baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                
                row = {
                    'Method': name.title(),
                    'Total Solutions': len(df),
                    'Feasible Solutions': len(feasible),
                    'Feasible Rate (%)': len(feasible) / len(df) * 100 if len(df) > 0 else 0,
                    'Min Cost ($M)': feasible['f1_total_cost_USD'].min() / 1e6 if len(feasible) > 0 else np.nan,
                    'Max Recall': feasible['detection_recall'].max() if len(feasible) > 0 else np.nan,
                    'Mean Recall': feasible['detection_recall'].mean() if len(feasible) > 0 else np.nan,
                }
                
                if 'f5_carbon_emissions_kgCO2e_year' in df.columns and len(feasible) > 0:
                    row['Min Carbon (t)'] = feasible['f5_carbon_emissions_kgCO2e_year'].min() / 1000
                
                data.append(row)
        
        table_df = pd.DataFrame(data)
        
        # Format numerical columns
        for col in ['Min Cost ($M)', 'Max Recall', 'Mean Recall', 'Feasible Rate (%)', 'Min Carbon (t)']:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                )
        
        # Save
        table_df.to_csv(self.output_dir / 'tables' / 'method_comparison.csv', index=False)
        
        latex_str = table_df.to_latex(index=False, escape=True)
        with open(self.output_dir / 'tables' / 'method_comparison.tex', 'w') as f:
            f.write(latex_str)
        
        logger.info(f"Generated comparison table for {len(data)} methods")
    
    def _generate_statistical_tests_table(self):
        """Generate statistical significance tests."""
        logger.info("Generating statistical tests table...")
        
        from scipy import stats
        
        tests = []
        
        # Compare NSGA-III with each baseline on recall
        pareto_recall = self.pareto_df['detection_recall'].values
        
        for name, df in self.baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                if len(feasible) > 10:  # Need enough samples for test
                    baseline_recall = feasible['detection_recall'].values
                    
                    # Mann-Whitney U test
                    stat, p_value = stats.mannwhitneyu(
                        pareto_recall, baseline_recall, alternative='greater'
                    )
                    
                    # Effect size (Vargha-Delaney A12)
                    a12 = self._vargha_delaney_a12(pareto_recall, baseline_recall)
                    
                    tests.append({
                        'Comparison': f'NSGA-III vs {name.title()}',
                        'Metric': 'Recall',
                        'NSGA-III Mean': f"{pareto_recall.mean():.4f}",
                        'Baseline Mean': f"{baseline_recall.mean():.4f}",
                        'Mann-Whitney U': f"{stat:.1f}",
                        'p-value': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
                        'A12 Effect Size': f"{a12:.3f}",
                        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No',
                    })
        
        if tests:
            table_df = pd.DataFrame(tests)
            table_df.to_csv(self.output_dir / 'tables' / 'statistical_tests.csv', index=False)
            
            latex_str = table_df.to_latex(index=False, escape=True)
            with open(self.output_dir / 'tables' / 'statistical_tests.tex', 'w') as f:
                f.write(latex_str)
            
            logger.info(f"Generated {len(tests)} statistical tests")
        else:
            logger.warning("Not enough data for statistical tests")
    
    def _vargha_delaney_a12(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Vargha-Delaney A12 effect size.
        
        Returns:
            A12 value (0.5 = no effect, >0.5 = x is better, <0.5 = y is better)
        """
        m, n = len(x), len(y)
        r = 0
        for xi in x:
            for yi in y:
                if xi > yi:
                    r += 1
                elif xi == yi:
                    r += 0.5
        return r / (m * n)
    
    def _generate_statistics(self):
        """Generate summary statistics JSON."""
        logger.info("\n--- Generating Summary Statistics ---")
        
        stats = {
            'generated_at': datetime.now().isoformat(),
            'pareto_solutions': {
                'count': len(self.pareto_df),
                'cost_range': [
                    float(self.pareto_df['f1_total_cost_USD'].min()),
                    float(self.pareto_df['f1_total_cost_USD'].max())
                ],
                'recall_range': [
                    float(self.pareto_df['detection_recall'].min()),
                    float(self.pareto_df['detection_recall'].max())
                ],
            },
            'baselines': {}
        }
        
        for name, df in self.baseline_dfs.items():
            if df is not None and len(df) > 0:
                feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
                stats['baselines'][name] = {
                    'total': len(df),
                    'feasible': len(feasible),
                    'feasible_rate': len(feasible) / len(df) if len(df) > 0 else 0
                }
        
        # Calculate HV if possible
        try:
            obj_cols = ['f1_total_cost_USD', 'detection_recall', 'f3_latency_seconds']
            obj_cols = [c for c in obj_cols if c in self.pareto_df.columns]
            
            if len(obj_cols) >= 2:
                obj_data = self.pareto_df[obj_cols].values.copy()
                # Convert recall to minimization
                if 'detection_recall' in obj_cols:
                    idx = obj_cols.index('detection_recall')
                    obj_data[:, idx] = 1 - obj_data[:, idx]
                
                hv = self.metrics.calculate_hypervolume(obj_data)
                stats['hypervolume'] = float(hv)
        except Exception as e:
            logger.warning(f"Could not calculate HV: {e}")
        
        # Save
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Generated summary statistics")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate publication results')
    parser.add_argument('--results', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/paper)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Generate results
    generator = PaperResultsGenerator(args.results, args.output)
    generator.generate_all()


if __name__ == '__main__':
    main()

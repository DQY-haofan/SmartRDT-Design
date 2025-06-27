#!/usr/bin/env python3
"""
Configuration Manager for RMTwin Optimization
Handles all configuration parameters
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfigManager:
    """Central configuration management"""
    
    # File paths
    config_file: str = 'config.json'
    sensor_csv: str = 'sensors_data.txt'
    algorithm_csv: str = 'algorithms_data.txt'
    infrastructure_csv: str = 'infrastructure_data.txt'
    cost_benefit_csv: str = 'cost_benefit_data.txt'
    
    # Network parameters
    road_network_length_km: float = 500.0
    planning_horizon_years: int = 10
    budget_cap_usd: float = 10_000_000
    
    # Operational parameters
    daily_wage_per_person: float = 1500
    fos_sensor_spacing_km: float = 0.1
    depreciation_rate: float = 0.1
    scenario_type: str = 'urban'
    carbon_intensity_factor: float = 0.417
    
    # Constraints
    min_recall_threshold: float = 0.70
    max_latency_seconds: float = 180.0
    max_disruption_hours: float = 100.0
    max_energy_kwh_year: float = 50_000
    min_mtbf_hours: float = 5_000
    
    # Optimization parameters
    n_objectives: int = 6
    population_size: int = 200
    n_generations: int = 100
    crossover_prob: float = 0.9
    crossover_eta: float = 20
    mutation_eta: float = 20
    
    # Baseline parameters
    n_random_samples: int = 500
    grid_resolution: int = 5
    weight_combinations: int = 50
    
    # Parallel computing
    use_parallel: bool = True
    n_processes: int = -1  # -1 means use all available cores - 1
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path('./results'))
    log_dir: Path = field(default_factory=lambda: Path('./results/logs'))
    figure_format: List[str] = field(default_factory=lambda: ['png', 'pdf'])
    
    # Advanced parameters
    class_imbalance_penalties: Dict[str, float] = field(default_factory=lambda: {
        'Traditional': 0.05,
        'ML': 0.02,
        'DL': 0.01,
        'PC': 0.03
    })
    
    network_quality_factors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'rural': {'Fiber': 0.8, '5G': 0.7, '4G': 0.9, 'LoRaWAN': 1.0},
        'urban': {'Fiber': 1.0, '5G': 1.0, '4G': 1.0, 'LoRaWAN': 0.9},
        'mixed': {'Fiber': 0.9, '5G': 0.85, '4G': 0.95, 'LoRaWAN': 0.95}
    })
    
    redundancy_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Cloud': 10.0,
        'OnPremise': 1.5,
        'Edge': 2.0,
        'Hybrid': 5.0
    })
    
    # Traffic parameters
    traffic_volume_hourly: int = 2000
    peak_hour_factor: float = 2.0
    night_hour_factor: float = 0.3
    default_lane_closure_ratio: float = 0.3
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize configuration from file if exists"""
        self.config_file = config_file
        
        # Set default values
        self._set_defaults()
        
        # Load from file if exists
        if Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Process computed values
        self._process_config()
        
        # Create directories
        self._create_directories()
    
    def _set_defaults(self):
        """Set any computed default values"""
        import multiprocessing as mp
        if self.n_processes == -1:
            self.n_processes = max(1, mp.cpu_count() - 1)
    
    def load_from_file(self, filepath: str):
        """Load configuration from JSON file"""
        logger.info(f"Loading configuration from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update attributes
            for key, value in data.items():
                if hasattr(self, key):
                    # Handle Path objects
                    if key in ['output_dir', 'log_dir']:
                        value = Path(value)
                    setattr(self, key, value)
                    logger.debug(f"Set {key} = {value}")
                else:
                    logger.warning(f"Unknown configuration key: {key}")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _process_config(self):
        """Process and validate configuration"""
        # Ensure paths are Path objects
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
        
        # Validate numeric ranges
        if self.population_size < 10:
            logger.warning(f"Population size {self.population_size} is very small")
        
        if self.n_generations < 10:
            logger.warning(f"Number of generations {self.n_generations} is very small")
        
        if self.budget_cap_usd < 100_000:
            logger.warning(f"Budget cap ${self.budget_cap_usd} may be too restrictive")
        
        # Validate file existence
        for csv_attr in ['sensor_csv', 'algorithm_csv', 'infrastructure_csv', 'cost_benefit_csv']:
            filepath = getattr(self, csv_attr)
            if not Path(filepath).exists():
                logger.error(f"Required data file not found: {filepath}")
    
    def _create_directories(self):
        """Create output directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'baseline').mkdir(exist_ok=True)
    
    def save_to_file(self, filepath: Optional[str] = None):
        """Save configuration to JSON file"""
        filepath = filepath or self.config_file
        
        # Convert to dictionary
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # Convert Path objects to strings
                if isinstance(value, Path):
                    value = str(value)
                config_dict[key] = value
        
        # Save
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {filepath}")
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        summary = [
            "Configuration Summary:",
            f"  Network: {self.road_network_length_km} km",
            f"  Budget: ${self.budget_cap_usd:,.0f}",
            f"  Planning Horizon: {self.planning_horizon_years} years",
            f"  Objectives: {self.n_objectives}",
            f"  Algorithm: NSGA-II/III",
            f"  Population: {self.population_size}",
            f"  Generations: {self.n_generations}",
            f"  Parallel Processing: {self.use_parallel} ({self.n_processes} cores)",
            f"  Output Directory: {self.output_dir}"
        ]
        return '\n'.join(summary)
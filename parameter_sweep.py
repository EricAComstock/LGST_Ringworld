#!/usr/bin/env python3
"""
Parameter Sweep Wrapper for Ringworld Atmospheric Simulations

This script allows running multiple simulations with different parameter combinations
and saves results in an organized manner for analysis.

Usage:
    python parameter_sweep.py --config config.json
    python parameter_sweep.py --quick-test
    python parameter_sweep.py --gravity-sweep
    
Author: Generated for Siyona Project
Date: 2025-09-26
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import traceback
from typing import Dict, List, Any, Optional
import itertools

# Import the main simulation function
from StochasticInputRK45Solver_Vectorized import main as run_simulation


class ParameterSweepRunner:
    """
    Manages parameter sweeps for atmospheric simulations.
    """
    
    def __init__(self, output_dir: str = "parameter_sweep_results", log_level: str = "INFO", 
                 save_format: str = "both"):
        """
        Initialize the parameter sweep runner.
        
        Args:
            output_dir: Directory to save results
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            save_format: Format to save results ("csv", "excel", "both")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate and store save format
        valid_formats = ["csv", "excel", "both"]
        if save_format not in valid_formats:
            raise ValueError(f"save_format must be one of {valid_formats}, got: {save_format}")
        self.save_format = save_format
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Default ringworld parameters
        self.default_params = {
            'radius': 8.19381e+12 * 1000,  # Convert km to meters
            'gravity': 2.743176313,  # m/s²
            't_max': 1e6,
            'dt': 100,
            'is_rotating': True,
            'num_particles': 100,
            'save_results': False,  # We'll handle saving ourselves
            'show_plots': False,
            'find_leak_rate': True,
            'temperature': 289,  # K
            'y_min': 149597870691 + 218 * 1000,  # Minimum spawn altitude (m)
            'y_max': 149597870691 + 218 * 1000 + 10 * 1000,  # Maximum spawn altitude (m)
            'z_length': 81938128337 * 1000,  # Convert width to meters
            'y_floor': 149597870691,  # Ringworld floor (1 AU)
            'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # diatomic oxygen
        }
        
        self.results_summary = []
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"parameter_sweep_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Parameter sweep initialized. Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {log_file}")
    
    def generate_parameter_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """
        Generate all combinations of parameters from ranges.
        
        Args:
            param_ranges: Dictionary where keys are parameter names and values are lists of values to try
            
        Returns:
            List of parameter dictionaries
        """
        # Get parameter names and their value lists
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_simulation(self, params: Dict[str, Any], run_id: str) -> Optional[Dict]:
        """
        Run a single simulation with given parameters.
        
        Args:
            params: Simulation parameters
            run_id: Unique identifier for this run
            
        Returns:
            Dictionary with simulation results and metadata
        """
        self.logger.info(f"Starting simulation {run_id}")
        self.logger.debug(f"Parameters: {params}")
        
        try:
            # Merge with default parameters
            sim_params = self.default_params.copy()
            sim_params.update(params)
            
            # Run simulation - pass key parameters as individual arguments
            start_time = datetime.now()
            results_df = run_simulation(
                radius=sim_params.get('radius'),
                gravity=sim_params.get('gravity'),
                t_max=sim_params.get('t_max'),
                dt=sim_params.get('dt'),
                is_rotating=sim_params.get('is_rotating'),
                num_particles=sim_params.get('num_particles'),
                save_results=sim_params.get('save_results'),
                show_plots=sim_params.get('show_plots'),
                find_leak_rate=sim_params.get('find_leak_rate'),
                comp_list=sim_params.get('comp_list'),
                sim_params=sim_params
            )
            end_time = datetime.now()
            
            if results_df is None or results_df.empty:
                self.logger.warning(f"Simulation {run_id} returned no results")
                return None
            
            # Calculate summary statistics
            total_particles = len(results_df)
            escaped = len(results_df[results_df['Result'] == 'escaped'])
            recaptured = len(results_df[results_df['Result'] == 'recaptured'])
            resimulate = len(results_df[results_df['Result'] == 'resimulate'])
            
            escape_fraction = escaped / total_particles if total_particles > 0 else 0
            recapture_fraction = recaptured / total_particles if total_particles > 0 else 0
            resimulate_fraction = resimulate / total_particles if total_particles > 0 else 0
            
            # Calculate additional statistics
            avg_beta_crossings = results_df['Beta crossings'].mean() if 'Beta crossings' in results_df.columns else 0
            
            # Prepare result summary
            result_summary = {
                'run_id': run_id,
                'timestamp': start_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'parameters': params,
                'total_particles': total_particles,
                'escaped': escaped,
                'recaptured': recaptured,
                'resimulate': resimulate,
                'escape_fraction': escape_fraction,
                'recapture_fraction': recapture_fraction,
                'resimulate_fraction': resimulate_fraction,
                'avg_beta_crossings': avg_beta_crossings,
                'success': True
            }
            
            # Save detailed results
            self.save_detailed_results(results_df, result_summary, run_id)
            
            self.logger.info(f"Simulation {run_id} completed successfully")
            self.logger.info(f"  Escaped: {escaped}/{total_particles} ({escape_fraction*100:.2f}%)")
            self.logger.info(f"  Recaptured: {recaptured}/{total_particles} ({recapture_fraction*100:.2f}%)")
            self.logger.info(f"  Resimulate: {resimulate}/{total_particles} ({resimulate_fraction*100:.2f}%)")
            
            return result_summary
            
        except Exception as e:
            self.logger.error(f"Simulation {run_id} failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            return {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'error': str(e),
                'success': False
            }
    
    def save_detailed_results(self, results_df: pd.DataFrame, summary: Dict, run_id: str):
        """
        Save detailed simulation results to files.
        
        Args:
            results_df: DataFrame with particle trajectory results
            summary: Summary statistics dictionary
            run_id: Unique run identifier
        """
        # Create run-specific directory
        run_dir = self.output_dir / "detailed_results" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save particle data based on format preference
        if self.save_format in ["csv", "both"]:
            csv_file = run_dir / f"{run_id}_particles.csv"
            results_df.to_csv(csv_file, index=False)
            self.logger.debug(f"Saved CSV: {csv_file}")
        
        if self.save_format in ["excel", "both"]:
            excel_file = run_dir / f"{run_id}_particles.xlsx"
            results_df.to_excel(excel_file, index=False)
            self.logger.debug(f"Saved Excel: {excel_file}")
        
        # Save run metadata
        metadata_file = run_dir / f"{run_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.debug(f"Detailed results saved to {run_dir}")
    
    def run_parameter_sweep(self, param_ranges: Dict[str, List], 
                          sweep_name: str = "parameter_sweep") -> pd.DataFrame:
        """
        Run a complete parameter sweep.
        
        Args:
            param_ranges: Dictionary of parameter ranges
            sweep_name: Name for this sweep
            
        Returns:
            DataFrame with summary results
        """
        self.logger.info(f"Starting parameter sweep: {sweep_name}")
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations(param_ranges)
        
        # Run simulations
        sweep_results = []
        for i, params in enumerate(combinations, 1):
            run_id = f"{sweep_name}_{i:04d}"
            
            self.logger.info(f"Running simulation {i}/{len(combinations)}: {run_id}")
            
            result = self.run_single_simulation(params, run_id)
            if result:
                sweep_results.append(result)
        
        # Create summary DataFrame
        if sweep_results:
            summary_df = pd.DataFrame(sweep_results)
            
            # Save summary results based on format preference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            saved_files = []
            if self.save_format in ["csv", "both"]:
                summary_file = self.output_dir / f"{sweep_name}_summary_{timestamp}.csv"
                summary_df.to_csv(summary_file, index=False)
                saved_files.append(str(summary_file))
            
            if self.save_format in ["excel", "both"]:
                excel_summary_file = self.output_dir / f"{sweep_name}_summary_{timestamp}.xlsx"
                summary_df.to_excel(excel_summary_file, index=False)
                saved_files.append(str(excel_summary_file))
            
            self.logger.info(f"Parameter sweep completed. Summary saved to: {', '.join(saved_files)}")
            
            # Print summary statistics
            successful_runs = summary_df[summary_df['success'] == True]
            if not successful_runs.empty:
                self.logger.info(f"Successful runs: {len(successful_runs)}/{len(summary_df)}")
                self.logger.info(f"Average escape fraction: {successful_runs['escape_fraction'].mean():.4f}")
                self.logger.info(f"Average recapture fraction: {successful_runs['recapture_fraction'].mean():.4f}")
            
            return summary_df
        else:
            self.logger.warning("No successful simulations in parameter sweep")
            return pd.DataFrame()


def load_config_file(config_path: str) -> Dict:
    """Load parameter configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_example_config():
    """Create an example configuration file."""
    example_config = {
        "sweep_name": "gravity_temperature_sweep",
        "description": "Sweep over gravity and temperature parameters",
        "parameter_ranges": {
            "gravity": [1.0, 2.743176313, 5.0, 10.0],
            "temperature": [200, 250, 289, 350, 400],
            "num_particles": [50, 100]
        },
        "output_settings": {
            "log_level": "INFO"
        }
    }
    
    config_file = "example_parameter_config.json"
    with open(config_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example configuration saved to {config_file}")
    return config_file


def quick_test_sweep(save_format: str = "both"):
    """Run a quick test with a few parameter combinations."""
    param_ranges = {
        "gravity": [2.743176313, 5.0],
        "temperature": [289, 350],
        "num_particles": [25]  # Small number for quick testing
    }
    
    runner = ParameterSweepRunner(output_dir="quick_test_results", save_format=save_format)
    return runner.run_parameter_sweep(param_ranges, "quick_test")


def gravity_sweep(save_format: str = "both"):
    """Run a sweep over different gravity values."""
    param_ranges = {
        "gravity": [0.5, 1.0, 2.743176313, 5.0, 10.0, 20.0],
        "num_particles": [100]
    }
    
    runner = ParameterSweepRunner(output_dir="gravity_sweep_results", save_format=save_format)
    return runner.run_parameter_sweep(param_ranges, "gravity_sweep")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run parameter sweeps for atmospheric simulations")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test sweep")
    parser.add_argument("--gravity-sweep", action="store_true", help="Run gravity parameter sweep")
    parser.add_argument("--create-example", action="store_true", help="Create example configuration file")
    parser.add_argument("--output-dir", type=str, default="parameter_sweep_results", 
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--save-format", type=str, default="both", 
                       choices=["csv", "excel", "both"], 
                       help="Format to save results (csv, excel, or both)")
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config()
        return
    
    if args.quick_test:
        print("Running quick test sweep...")
        results = quick_test_sweep(save_format=args.save_format)
        print(f"Quick test completed. Results shape: {results.shape}")
        return
    
    if args.gravity_sweep:
        print("Running gravity parameter sweep...")
        results = gravity_sweep(save_format=args.save_format)
        print(f"Gravity sweep completed. Results shape: {results.shape}")
        return
    
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)
        
        # Get save format from config file or command line argument
        config_save_format = config.get("output_settings", {}).get("save_format", args.save_format)
        
        runner = ParameterSweepRunner(
            output_dir=args.output_dir,
            log_level=args.log_level,
            save_format=config_save_format
        )
        
        results = runner.run_parameter_sweep(
            config["parameter_ranges"],
            config.get("sweep_name", "custom_sweep")
        )
        
        print(f"Parameter sweep completed. Results shape: {results.shape}")
        return
    
    # If no specific action, show help
    parser.print_help()


if __name__ == "__main__":
    main()

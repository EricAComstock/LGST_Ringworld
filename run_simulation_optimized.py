#!/usr/bin/env python3
"""
run_simulation_optimized.py - Enhanced simulation runner for ringworld atmospheric simulations

This script reads parameters from ringworld_parameters.csv and runs simulations
using the optimized LGST simulation code. Results are saved with comprehensive
logging for reproducibility.

Key Features:
- Reads parameters from CSV file
- Uses optimized vectorized solver
- Comprehensive logging with reproducibility information
- Organized output directory structure
- Command-line interface for flexible execution

Usage:
    python run_simulation_optimized.py                              # Run all simulations
    python run_simulation_optimized.py --priority 1                 # Run specific priority
    python run_simulation_optimized.py --designation "Bishop Ring"  # Run specific ringworld
    python run_simulation_optimized.py --row 0                      # Run specific row (0-indexed)
    python run_simulation_optimized.py --max-runs 3                 # Limit number of runs
    python run_simulation_optimized.py --particles 10000            # Override particle count

Author: Siyona Agarwal
Date: October 2025
Version: 1.0 - Optimized with comprehensive logging
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
import traceback
import math
from datetime import datetime
from pathlib import Path
import json

# Import optimized simulation components
from LGST_Simulation_Wrapper import run_simulation as run_optimized_simulation
from SolverSharedCodePlusSolar_Optimized import compute_motion, SSCPSVarInput
from StochasticInput import stochastic_initial_conditions, SIVarInput
from TrajectoryClassification_numpy import classify_trajectory, TCVarInput
from LeakRate import LRVarInput


class SimulationLogger:
    """Handles comprehensive logging for reproducibility."""
    
    def __init__(self, log_dir: str, run_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.log_file = self.log_dir / f"{run_id}.log"
        
        # Set up file logging
        self.logger = logging.getLogger(run_id)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_system_info(self):
        """Log system and environment information."""
        self.logger.info("=" * 70)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 70)
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"NumPy version: {np.__version__}")
        self.logger.info(f"Pandas version: {pd.__version__}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Try to get git commit info for reproducibility
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                stderr=subprocess.DEVNULL).decode('utf-8').strip()
            self.logger.info(f"Git commit: {git_commit}")
        except:
            self.logger.info("Git commit: Not available")
    
    def log_code_versions(self):
        """Log which code files are being used."""
        self.logger.info("=" * 70)
        self.logger.info("CODE VERSIONS")
        self.logger.info("=" * 70)
        self.logger.info("Main simulation: LGST_Simulation_Wrapper.py (parameter-free wrapper)")
        self.logger.info("Solver: StochasticInputRK45Solver_Vectorized.py")
        self.logger.info("Physics: SolverSharedCodePlusSolar_Optimized.py")
        self.logger.info("Classification: TrajectoryClassification_numpy.py")
        self.logger.info("Initial conditions: StochasticInput.py")
        self.logger.info("Leak rate: LeakRate.py")
    
    def log_ringworld_params(self, row: pd.Series):
        """Log ringworld parameters from CSV."""
        self.logger.info("=" * 70)
        self.logger.info("RINGWORLD PARAMETERS")
        self.logger.info("=" * 70)
        self.logger.info(f"Designation: {row.get('Designation', 'Unknown')}")
        self.logger.info(f"Priority: {row.get('Priority', 'N/A')}")
        self.logger.info(f"Width: {row.get('Width (km)', 'N/A')} km")
        self.logger.info(f"Radius: {row.get('Radius (km)', 'N/A')} km")
        self.logger.info(f"Gravity: {row.get('Gravity (m/s^2)', 'N/A')} m/s²")
        self.logger.info(f"Angular velocity: {row.get('Ringworld angular velocity (rad/s)', 'N/A')} rad/s")
        self.logger.info(f"Central mass: {row.get('Central mass', 'None')}")
    
    def log_simulation_params(self, params: dict):
        """Log simulation parameters."""
        self.logger.info("=" * 70)
        self.logger.info("SIMULATION PARAMETERS")
        self.logger.info("=" * 70)
        self.logger.info(f"\nSimulation parameters:")
        self.logger.info(f"  Number of particles: {params.get('num_particles', 'N/A')}")
        
        # Show orbital information if available
        orbital_period = params.get('orbital_period')
        if orbital_period:
            t_max = params.get('t_max', 0)
            num_orbits = t_max / orbital_period if orbital_period > 0 else 0
            
            # Format orbital period nicely
            if orbital_period < 3600:
                period_str = f"{orbital_period/60:.1f} minutes"
            elif orbital_period < 86400:
                period_str = f"{orbital_period/3600:.1f} hours"
            else:
                period_str = f"{orbital_period/86400:.1f} days"
            
            self.logger.info(f"  Orbital period: {period_str} ({orbital_period:.0f} s)")
            self.logger.info(f"  Simulation time: {t_max:.0f} s ({t_max/3600:.2f} hours) = {num_orbits:.1f} orbits")
        else:
            self.logger.info(f"  Simulation time: {params.get('t_max', 'N/A')} s ({params.get('t_max', 0)/3600:.2f} hours)")
        
        self.logger.info(f"  Time step: {params.get('dt', 'N/A')} s")
        self.logger.info(f"  Temperature: {params.get('temperature', 'N/A')} K")
        self.logger.info(f"  Rotating frame: {params.get('is_rotating', 'N/A')}")
        self.logger.info(f"  Find leak rate: {params.get('find_leak_rate', 'N/A')}")
        
        # Geometric parameters
        self.logger.info(f"\nGeometric parameters:")
        self.logger.info(f"  Radius (y_floor): {params.get('radius', 'N/A')} m")
        self.logger.info(f"  Width (z_length): {params.get('z_length', 'N/A')} m")
        self.logger.info(f"  Atmosphere thickness: {params.get('atmosphere_thickness_m', 'N/A')} m ({params.get('atmosphere_thickness_m', 0)/1000:.1f} km)")
        self.logger.info(f"  Spawn range: {params.get('spawn_range_m', 'N/A')} m ({params.get('spawn_range_m', 0)/1000:.1f} km)")
        self.logger.info(f"  Spawn altitude min (y_min): {params.get('y_min', 'N/A')} m")
        self.logger.info(f"  Spawn altitude max (y_max): {params.get('y_max', 'N/A')} m")
        
        # Atmospheric composition
        if 'comp_list' in params:
            self.logger.info(f"\nAtmospheric composition:")
            for comp in params['comp_list']:
                self.logger.info(f"  {comp[0]}: mass={comp[1]} kg, charge={comp[2]}, density={comp[3]} particles/m³")
    
    def log_optimization_params(self, params: dict):
        """Log optimization and performance parameters."""
        self.logger.info("=" * 70)
        self.logger.info("OPTIMIZATION PARAMETERS")
        self.logger.info("=" * 70)
        self.logger.info(f"Parallel processes: {params.get('num_processes', 'Auto-detect')}")
        self.logger.info(f"Batch size: {params.get('batch_size', 'Auto-detect')}")
        self.logger.info(f"Using vectorized solver: Yes")
        self.logger.info(f"Using NumPy classification: Yes")
    
    def log_results(self, results, duration: float):
        """Log simulation results."""
        self.logger.info("=" * 70)
        self.logger.info("SIMULATION RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Check if results is a DataFrame (from optimized simulation)
        if results is not None and not (isinstance(results, pd.DataFrame) and results.empty):
            # Handle DataFrame results
            if isinstance(results, pd.DataFrame):
                total_particles = len(results)
                
                # Count classifications - check for both 'Result' (vectorized) and 'classification' (legacy)
                result_col = None
                if 'Result' in results.columns:
                    result_col = 'Result'
                elif 'classification' in results.columns:
                    result_col = 'classification'
                
                if result_col:
                    escaped = (results[result_col] == 'escaped').sum()
                    recaptured = (results[result_col] == 'recaptured').sum()
                    resimulate = (results[result_col] == 'resimulate').sum()
                    
                    self.logger.info(f"Total particles: {total_particles}")
                    self.logger.info(f"Escaped: {escaped} ({escaped/total_particles*100:.2f}%)")
                    self.logger.info(f"Recaptured: {recaptured} ({recaptured/total_particles*100:.2f}%)")
                    self.logger.info(f"Resimulate: {resimulate} ({resimulate/total_particles*100:.2f}%)")
                    
                    # Diagnostic info for resimulate particles
                    if resimulate > 0:
                        self.logger.warning(f"\n⚠️  {resimulate} particles need resimulation (simulation time may be too short)")
                        self.logger.warning(f"   These particles were still falling when t_max was reached")
                        self.logger.warning(f"   Consider increasing --t-max if resimulate % is high")
                    
                    # Also log beta crossings if available
                    if 'Beta crossings' in results.columns:
                        total_crossings = results['Beta crossings'].sum()
                        self.logger.info(f"Total beta crossings: {total_crossings}")
                else:
                    self.logger.info(f"Total particles: {total_particles}")
                    self.logger.info("Classification column not found in results")
            
            # Handle dict results (legacy format)
            elif isinstance(results, dict):
                self.logger.info(f"Total particles: {results.get('total_particles', 'N/A')}")
                self.logger.info(f"Escaped: {results.get('escaped', 'N/A')} ({results.get('escape_fraction', 0)*100:.4f}%)")
                self.logger.info(f"Recaptured: {results.get('recaptured', 'N/A')} ({results.get('recapture_fraction', 0)*100:.4f}%)")
                
                if 'resimulate' in results:
                    self.logger.info(f"Resimulate: {results.get('resimulate', 'N/A')} ({results.get('resimulate_fraction', 0)*100:.4f}%)")
                
                if 'leak_rate' in results:
                    self.logger.info(f"\nLeak rate analysis:")
                    self.logger.info(f"  Leak rate: {results.get('leak_rate', 'N/A')}")
                    self.logger.info(f"  Atmospheric lifetime: {results.get('atmospheric_lifetime', 'N/A')}")
        else:
            self.logger.error("No results returned from simulation")
    
    def log_error(self, error: Exception):
        """Log error information."""
        self.logger.error("=" * 70)
        self.logger.error("SIMULATION ERROR")
        self.logger.error("=" * 70)
        self.logger.error(f"Error type: {type(error).__name__}")
        self.logger.error(f"Error message: {str(error)}")
        self.logger.error("\nFull traceback:")
        self.logger.error(traceback.format_exc())


class RingworldSimulationRunner:
    """Main class for running ringworld simulations from CSV parameters."""
    
    # Default atmospheric structure constants (can be overridden per ringworld)
    DEFAULT_ATMOSPHERE_THICKNESS_M = 218 * 1000  # 218 km atmosphere thickness [m]
    DEFAULT_SPAWN_RANGE_M = 10 * 1000  # 10 km spawn range [m]
    
    def __init__(self, csv_file: str, results_dir: str):
        self.csv_file = csv_file
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Physical constants
        self.G = 6.6743e-11  # Universal gravitational constant [m³/kg/s²]
        
        # Atmospheric parameters (defaults)
        self.T = 289  # Temperature [K]
        self.P_0 = 101325  # Atmospheric pressure at sea level [Pa]
        self.K_b = 1.380649e-23  # Boltzmann constant [J/K]
        self.m = 2.6566962e-26 * 2  # Mass of diatomic molecule [kg]
        self.n_0 = 2.687e25  # Molecular density [1/m³]
        self.d = 3.59e-10  # Molecular diameter [m]
        
        # Planetary mass dictionary (in kg)
        self.planetary_masses = {
            'sun': 1.989e30,
            'jupiter': 1.898e27,
            'saturn': 5.683e26,
            'neptune': 1.024e26,
            'uranus': 8.681e25,
            'earth': 5.972e24,
            'venus': 4.867e24,
            'mars': 6.39e23,
            'mercury': 3.301e23,
            'moon': 7.342e22
        }
    
    def calculate_orbital_sim_params(self, row: pd.Series, num_orbits: float = 5.0) -> tuple:
        """
        Calculate simulation time and time step based on orbital period.
        
        Args:
            row: CSV row with ringworld parameters
            num_orbits: Number of orbits to simulate (default: 5.0)
            
        Returns:
            tuple: (t_max, dt, orbital_period_seconds)
        """
        angular_velocity = row.get('Ringworld angular velocity (rad/s)', 0)
        
        if angular_velocity == 0:
            # No rotation - use default fixed time
            return 5000, 0.1, None
        
        # Calculate orbital period
        orbital_period = 2 * math.pi / angular_velocity  # seconds
        
        # Simulation time = num_orbits * orbital_period
        t_max = orbital_period * num_orbits
        
        # Time step: aim for ~100-200 steps per orbit for good resolution
        steps_per_orbit = 1500
        dt = orbital_period / steps_per_orbit
        
        # Ensure reasonable bounds
        dt = max(0.1, np.sqrt(dt))  # Between 0.1 and 100 seconds
        print('Orbital Period (s):', orbital_period)
        print('Time Step (s):', dt)
        
        return t_max, dt, orbital_period
    
    def parse_central_mass(self, mass_str: str) -> float:
        """Parse central mass string like '1xSun' or '0.05*Jupiter'."""
        if pd.isna(mass_str) or str(mass_str).lower() in ['none', 'nan', '']:
            return None
        
        try:
            mass_str = str(mass_str).strip().lower()
            
            # Handle different formats
            if 'x' in mass_str:
                parts = mass_str.split('x')
                scale_factor = float(parts[0])
                object_name = parts[1].strip()
            elif '*' in mass_str:
                parts = mass_str.split('*')
                scale_factor = float(parts[0])
                object_name = parts[1].strip()
            else:
                parts = mass_str.split()
                if len(parts) == 1:
                    scale_factor = 1.0
                    object_name = parts[0].lower()
                elif len(parts) == 2:
                    scale_factor = float(parts[0])
                    object_name = parts[1].lower()
                else:
                    return None
            
            if object_name in self.planetary_masses:
                return scale_factor * self.planetary_masses[object_name]
            else:
                return None
        except:
            return None
    
    def load_csv(self) -> pd.DataFrame:
        """Load ringworld parameters from CSV."""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} ringworld configurations from {self.csv_file}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            raise
    
    def convert_row_to_params(self, row: pd.Series, override_params: dict = None, num_orbits: float = 5.0) -> dict:
        """Convert CSV row to simulation parameters."""
        params = {}
        
        # Basic parameters from CSV
        radius_km = row.get('Radius (km)', 1000)
        width_km = row.get('Width (km)', 200)
        gravity = row.get('Gravity (m/s^2)', 9.81)
        angular_velocity = row.get('Ringworld angular velocity (rad/s)', 0)
        
        # Atmospheric structure parameters (can be specified in CSV or use defaults)
        atmosphere_thickness_km = row.get('Atmosphere Thickness (km)', 218)
        spawn_range_km = row.get('Spawn Range (km)', 10)
        
        # Convert to meters
        radius_m = radius_km * 1000
        width_m = width_km * 1000
        atmosphere_thickness_m = atmosphere_thickness_km * 1000
        spawn_range_m = spawn_range_km * 1000
        
        # Store atmospheric parameters for later use
        params['atmosphere_thickness_m'] = atmosphere_thickness_m
        params['spawn_range_m'] = spawn_range_m
        
        # Geometric parameters
        params['radius'] = radius_m
        params['z_length'] = width_m
        params['gravity'] = gravity
        params['y_floor'] = radius_m
        # Spawn particles in the atmosphere (below the floor surface)
        params['y_min'] = radius_m - atmosphere_thickness_m - spawn_range_m  # Bottom of spawn region
        params['y_max'] = radius_m - atmosphere_thickness_m  # Top of spawn region (at atmosphere boundary)
        
        # Calculate orbital-based simulation parameters
        t_max, dt, orbital_period = self.calculate_orbital_sim_params(row, num_orbits=num_orbits)
        
        # Simulation parameters (defaults, can be overridden)
        params['t_max'] = t_max
        params['dt'] = dt
        params['orbital_period'] = orbital_period  # Store for logging
        params['num_particles'] = 10000  # Default particle count
        params['temperature'] = 289  # K
        params['is_rotating'] = False  # Disable rotation by default
        params['save_results'] = True
        params['show_plots'] = False
        params['find_leak_rate'] = True
        
        # Atmospheric composition
        diatomic_oxygen = ("O2", 2.6566962e-26 * 2, 0, 100)
        params['comp_list'] = [diatomic_oxygen]
        
        # Parse central mass if present
        central_mass_str = row.get('Central mass', 'None')
        central_mass = self.parse_central_mass(central_mass_str)
        if central_mass:
            params['central_mass'] = central_mass
            params['solar_mu'] = self.G * central_mass
            params['is_rotating'] = True  # Enable rotation if central mass present
        
        # Performance parameters
        params['num_processes'] = None  # Auto-detect
        params['batch_size'] = None  # Auto-detect
        
        # Apply overrides
        if override_params:
            params.update(override_params)
        
        return params
    
    def initialize_modules(self, params: dict):
        """Initialize all simulation modules with parameters."""
        # Calculate derived parameters
        alpha = params['y_floor'] - params['atmosphere_thickness_m']  # Atmosphere boundary
        beta = params['z_length'] / 2  # Lateral boundary
        
        # Initialize modules
        SSCPSVarInput(self.G)
        SIVarInput(params['temperature'], params['y_min'], params['y_max'], 
                   params['z_length'], params['y_floor'])
        TCVarInput(params['z_length'], beta, params['y_floor'], alpha, 
                   params['y_min'], params['y_max'])
        LRVarInput(self.P_0, self.K_b, params['temperature'], self.m, 
                   params['gravity'], self.n_0, self.d)
    
    def generate_run_id(self, designation: str, num_particles: int) -> str:
        """Generate standardized run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_designation = designation.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        return f"{clean_designation}_{num_particles}particles_{timestamp}"
    
    def run_single_simulation(self, row: pd.Series, override_params: dict = None):
        """Run simulation for a single ringworld configuration."""
        designation = row.get('Designation', 'Unknown')
        
        # Convert row to parameters
        params = self.convert_row_to_params(row, override_params)
        
        # Generate run ID
        run_id = self.generate_run_id(designation, params['num_particles'])
        
        # Set up logging
        log_dir = self.results_dir / "logs"
        sim_logger = SimulationLogger(str(log_dir), run_id)
        
        # Log all information
        sim_logger.log_system_info()
        sim_logger.log_code_versions()
        sim_logger.log_ringworld_params(row)
        sim_logger.log_simulation_params(params)
        sim_logger.log_optimization_params(params)
        
        print(f"\n{'='*70}")
        print(f"Running simulation: {designation}")
        print(f"Run ID: {run_id}")
        print(f"{'='*70}")
        
        try:
            # Initialize modules
            self.initialize_modules(params)
            
            # Run simulation
            start_time = datetime.now()
            
            results = run_optimized_simulation(
                radius=params['radius'],
                gravity=params['gravity'],
                t_max=params['t_max'],
                dt=params['dt'],
                is_rotating=params['is_rotating'],
                num_particles=params['num_particles'],
                save_results=params['save_results'],
                show_plots=params['show_plots'],
                find_leak_rate=params['find_leak_rate'],
                comp_list=params['comp_list'],
                num_processes=params.get('num_processes'),
                batch_size=params.get('batch_size'),
                output_dir=str(self.results_dir),
                output_filename=run_id
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log results
            sim_logger.log_results(results, duration)
            
            print(f"\n✅ Simulation completed successfully!")
            print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"Results saved in: {self.results_dir}")
            print(f"Log file: {log_dir / f'{run_id}.log'}")
            
            return {
                'success': True,
                'run_id': run_id,
                'designation': designation,
                'duration': duration,
                'results': results
            }
            
        except Exception as e:
            sim_logger.log_error(e)
            print(f"\n❌ Simulation failed: {str(e)}")
            print(f"See log file for details: {log_dir / f'{run_id}.log'}")
            
            return {
                'success': False,
                'run_id': run_id,
                'designation': designation,
                'error': str(e)
            }
    
    def run_all_simulations(self, priority_filter: int = None, 
                           designation_filter: str = None,
                           row_index: int = None,
                           max_runs: int = None,
                           override_params: dict = None):
        """Run simulations for multiple configurations."""
        df = self.load_csv()
        
        # Apply filters
        if row_index is not None:
            if row_index < 0 or row_index >= len(df):
                raise ValueError(f"Row index {row_index} out of range (0-{len(df)-1})")
            df = df.iloc[[row_index]]
            print(f"🎯 Running simulation for row {row_index}: {df.iloc[0]['Designation']}")
        elif designation_filter:
            df = df[df['Designation'].str.contains(designation_filter, case=False, na=False)]
            print(f"🎯 Filtered to {len(df)} configurations matching '{designation_filter}'")
        elif priority_filter is not None:
            df = df[df['Priority'] == priority_filter]
            print(f"🎯 Filtered to {len(df)} configurations with priority {priority_filter}")
        
        if max_runs and len(df) > max_runs:
            df = df.head(max_runs)
            print(f"🎯 Limited to first {max_runs} configurations")
        
        if len(df) == 0:
            print("❌ No configurations match the filter criteria")
            return []
        
        # Run simulations
        results = []
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            print(f"\n{'='*70}")
            print(f"Simulation {idx}/{len(df)}")
            print(f"{'='*70}")
            
            result = self.run_single_simulation(row, override_params)
            results.append(result)
        
        # Summary
        print(f"\n{'='*70}")
        print("SIMULATION SUMMARY")
        print(f"{'='*70}")
        successful = sum(1 for r in results if r['success'])
        print(f"Total simulations: {len(results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {len(results) - successful}")
        print(f"📁 Results directory: {self.results_dir}")
        
        return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run ringworld atmospheric simulations from CSV parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation_v2.py                              # Run all simulations
  python run_simulation_v2.py --priority 1                 # Run priority 1 only
  python run_simulation_v2.py --designation "Bishop Ring"  # Run Bishop Ring
  python run_simulation_v2.py --row 0                      # Run first row
  python run_simulation_v2.py --max-runs 3                 # Run first 3
  python run_simulation_v2.py --particles 10000            # Override particle count
        """
    )
    
    parser.add_argument('--csv', type=str, 
                       default='ringworld_parameters.csv',
                       help='Path to CSV file with parameters (default: ringworld_parameters.csv)')
    
    parser.add_argument('--results-dir', type=str,
                       default='simulation_results',
                       help='Directory to save results (default: simulation_results)')
    
    parser.add_argument('--priority', type=int,
                       help='Run only simulations with this priority')
    
    parser.add_argument('--designation', type=str,
                       help='Run only simulations matching this designation (case-insensitive)')
    
    parser.add_argument('--row', type=int,
                       help='Run simulation for specific row index (0-based)')
    
    parser.add_argument('--max-runs', type=int,
                       help='Maximum number of simulations to run')
    
    parser.add_argument('--particles', type=int,
                       help='Override number of particles')
    
    parser.add_argument('--t-max', type=float,
                       help='Override simulation time (seconds)')
    
    parser.add_argument('--dt', type=float,
                       help='Override time step (seconds)')
    
    parser.add_argument('--temperature', type=float,
                       help='Override temperature (Kelvin)')
    
    parser.add_argument('--num-orbits', type=float,
                       help='Number of orbits to simulate (overrides t-max, uses orbital period)')
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not Path(args.csv).exists():
        print(f"❌ Error: CSV file '{args.csv}' not found")
        sys.exit(1)
    
    # Build override parameters
    override_params = {}
    if args.particles:
        override_params['num_particles'] = args.particles
    if args.t_max:
        override_params['t_max'] = args.t_max
    if args.dt:
        override_params['dt'] = args.dt
    if args.temperature:
        override_params['temperature'] = args.temperature
    
    # Create runner and run simulations
    runner = RingworldSimulationRunner(
        csv_file=args.csv,
        results_dir=args.results_dir
    )
    
    print("=" * 70)
    print("RINGWORLD ATMOSPHERIC SIMULATION RUNNER V2.0")
    print("=" * 70)
    print(f"CSV file: {args.csv}")
    print(f"Results directory: {args.results_dir}")
    if override_params:
        print(f"Parameter overrides: {override_params}")
    print("=" * 70)
    
    results = runner.run_all_simulations(
        priority_filter=args.priority,
        designation_filter=args.designation,
        row_index=args.row,
        max_runs=args.max_runs,
        override_params=override_params if override_params else None
    )
    
    print(f"\n{'='*70}")
    print("ALL SIMULATIONS COMPLETED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

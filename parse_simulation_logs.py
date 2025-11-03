"""
Parse simulation log files and compile information into a CSV file.

This script reads all log files from the simulation_results/logs directory
and extracts key simulation parameters and results into a structured CSV format.
"""

import os
import re
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def parse_log_file(log_path: str) -> Optional[Dict[str, str]]:
    """
    Parse a single log file and extract relevant information.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Dictionary containing extracted information, or None if parsing fails
    """
    data = {
        'log_filename': os.path.basename(log_path),
        'timestamp': '',
        'git_commit': '',
        'designation': '',
        'priority': '',
        'width_km': '',
        'radius_km': '',
        'gravity_ms2': '',
        'angular_velocity_rads': '',
        'central_mass': '',
        'num_particles': '',
        'orbital_period_s': '',
        'orbital_period_min': '',
        'simulation_time_s': '',
        'simulation_time_hours': '',
        'num_orbits': '',
        'time_step_s': '',
        'actual_steps': '',
        'max_allowed_steps': '',
        'temperature_K': '',
        'rotating_frame': '',
        'find_leak_rate': '',
        'atmosphere_thickness_m': '',
        'spawn_range_m': '',
        'spawn_altitude_min_m': '',
        'spawn_altitude_max_m': '',
        'atmospheric_composition': '',
        'solver': '',
        'physics': '',
        'classification': '',
        'initial_conditions': '',
        'leak_rate_module': '',
        'vectorized_solver': '',
        'numpy_classification': '',
        'parallel_processes': '',
        'batch_size': '',
        'duration_s': '',
        'duration_min': '',
        'total_particles': '',
        'escaped': '',
        'escaped_pct': '',
        'recaptured': '',
        'recaptured_pct': '',
        'resimulate': '',
        'resimulate_pct': '',
        'beta_crossings': ''
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract timestamp
        match = re.search(r'Timestamp: (.+)', content)
        if match:
            data['timestamp'] = match.group(1)
            
        # Extract git commit
        match = re.search(r'Git commit: (.+)', content)
        if match:
            data['git_commit'] = match.group(1)
            
        # Extract ringworld parameters
        match = re.search(r'Designation: (.+)', content)
        if match:
            data['designation'] = match.group(1)
            
        match = re.search(r'Priority: (.+)', content)
        if match:
            data['priority'] = match.group(1)
            
        match = re.search(r'Width: (.+?) km', content)
        if match:
            data['width_km'] = match.group(1)
            
        match = re.search(r'Radius: (.+?) km', content)
        if match:
            data['radius_km'] = match.group(1)
            
        match = re.search(r'Gravity: (.+?) m/s²', content)
        if match:
            data['gravity_ms2'] = match.group(1)
            
        match = re.search(r'Angular velocity: (.+?) rad/s', content)
        if match:
            data['angular_velocity_rads'] = match.group(1)
            
        match = re.search(r'Central mass: (.+)', content)
        if match:
            data['central_mass'] = match.group(1)
            
        # Extract simulation parameters
        match = re.search(r'Number of particles: (.+)', content)
        if match:
            data['num_particles'] = match.group(1)
            
        match = re.search(r'Orbital period: .+? \((.+?) s\)', content)
        if match:
            data['orbital_period_s'] = match.group(1)
            
        match = re.search(r'Orbital period: (.+?) minutes', content)
        if match:
            data['orbital_period_min'] = match.group(1)
            
        match = re.search(r'Simulation time: (.+?) s', content)
        if match:
            data['simulation_time_s'] = match.group(1)
            
        match = re.search(r'Simulation time: .+? \((.+?) hours\)', content)
        if match:
            data['simulation_time_hours'] = match.group(1)
            
        match = re.search(r'Simulation time: .+? = (.+?) orbits', content)
        if match:
            data['num_orbits'] = match.group(1)
            
        match = re.search(r'Time step: (.+?) s', content)
        if match:
            data['time_step_s'] = match.group(1)
            
        match = re.search(r'Actual steps for simulation: (.+)', content)
        if match:
            data['actual_steps'] = match.group(1).replace(',', '')
            
        match = re.search(r'Maximum allowed steps: (.+)', content)
        if match:
            data['max_allowed_steps'] = match.group(1).replace(',', '')
            
        match = re.search(r'Temperature: (.+?) K', content)
        if match:
            data['temperature_K'] = match.group(1)
            
        match = re.search(r'Rotating frame: (.+)', content)
        if match:
            data['rotating_frame'] = match.group(1)
            
        match = re.search(r'Find leak rate: (.+)', content)
        if match:
            data['find_leak_rate'] = match.group(1)
            
        # Extract geometric parameters
        match = re.search(r'Atmosphere thickness: (.+?) m', content)
        if match:
            data['atmosphere_thickness_m'] = match.group(1)
            
        match = re.search(r'Spawn range: (.+?) m', content)
        if match:
            data['spawn_range_m'] = match.group(1)
            
        match = re.search(r'Spawn altitude min \(y_min\): (.+?) m', content)
        if match:
            data['spawn_altitude_min_m'] = match.group(1)
            
        match = re.search(r'Spawn altitude max \(y_max\): (.+?) m', content)
        if match:
            data['spawn_altitude_max_m'] = match.group(1)
            
        # Extract atmospheric composition (simplified - just get the first line)
        match = re.search(r'Atmospheric composition:\n.+?INFO - (.+)', content)
        if match:
            data['atmospheric_composition'] = match.group(1).strip()
            
        # Extract code versions
        match = re.search(r'Solver: (.+)', content)
        if match:
            data['solver'] = match.group(1)
            
        match = re.search(r'Physics: (.+)', content)
        if match:
            data['physics'] = match.group(1)
            
        match = re.search(r'Classification: (.+)', content)
        if match:
            data['classification'] = match.group(1)
            
        match = re.search(r'Initial conditions: (.+)', content)
        if match:
            data['initial_conditions'] = match.group(1)
            
        match = re.search(r'Leak rate: (.+)', content)
        if match:
            data['leak_rate_module'] = match.group(1)
            
        # Extract optimization parameters
        match = re.search(r'Using vectorized solver: (.+)', content)
        if match:
            data['vectorized_solver'] = match.group(1)
            
        match = re.search(r'Using NumPy classification: (.+)', content)
        if match:
            data['numpy_classification'] = match.group(1)
            
        match = re.search(r'Parallel processes: (.+)', content)
        if match:
            data['parallel_processes'] = match.group(1)
            
        match = re.search(r'Batch size: (.+)', content)
        if match:
            data['batch_size'] = match.group(1)
            
        # Extract results
        match = re.search(r'Duration: (.+?) seconds', content)
        if match:
            data['duration_s'] = match.group(1)
            
        match = re.search(r'Duration: .+? \((.+?) minutes\)', content)
        if match:
            data['duration_min'] = match.group(1)
            
        match = re.search(r'Total particles: (.+)', content)
        if match:
            data['total_particles'] = match.group(1)
            
        match = re.search(r'Escaped: (\d+) \((.+?)%\)', content)
        if match:
            data['escaped'] = match.group(1)
            data['escaped_pct'] = match.group(2)
            
        match = re.search(r'Recaptured: (\d+) \((.+?)%\)', content)
        if match:
            data['recaptured'] = match.group(1)
            data['recaptured_pct'] = match.group(2)
            
        match = re.search(r'Resimulate: (\d+) \((.+?)%\)', content)
        if match:
            data['resimulate'] = match.group(1)
            data['resimulate_pct'] = match.group(2)
            
        match = re.search(r'Total beta crossings: (.+)', content)
        if match:
            data['beta_crossings'] = match.group(1)
            
        return data
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def parse_all_logs(logs_dir: str, output_csv: str):
    """
    Parse all log files in the directory and save to CSV.
    
    Args:
        logs_dir: Directory containing log files
        output_csv: Path to output CSV file
    """
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"Error: Directory {logs_dir} does not exist")
        return
        
    # Find all log files
    log_files = sorted(logs_path.glob('*.log'))
    
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return
        
    print(f"Found {len(log_files)} log file(s)")
    
    # Parse all log files
    all_data = []
    for log_file in log_files:
        print(f"Parsing: {log_file.name}")
        data = parse_log_file(str(log_file))
        if data:
            all_data.append(data)
            
    if not all_data:
        print("No data extracted from log files")
        return
        
    # Write to CSV
    fieldnames = list(all_data[0].keys())
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
        
    print(f"\nSuccessfully compiled {len(all_data)} log file(s) into: {output_csv}")
    print(f"Total columns: {len(fieldnames)}")


def main():
    """Main entry point."""
    # Set paths relative to script location
    script_dir = Path(__file__).parent
    logs_dir = script_dir / 'simulation_results' / 'logs'
    output_dir = script_dir / 'simulation_results' / 'compiled_results'
    output_csv = output_dir / 'compiled_simulation_logs.csv'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Simulation Log Parser")
    print("=" * 70)
    print(f"Logs directory: {logs_dir}")
    print(f"Output CSV: {output_csv}")
    print("=" * 70)
    print()
    
    parse_all_logs(str(logs_dir), str(output_csv))
    
    print("\nDone!")


if __name__ == '__main__':
    main()

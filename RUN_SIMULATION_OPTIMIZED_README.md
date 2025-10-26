# run_simulation_optimized.py - Comprehensive Simulation Runner

## Overview

`run_simulation_optimized.py` is an enhanced simulation runner that integrates the optimized LGST simulation code with CSV parameter management and comprehensive logging for reproducibility.

## Key Features

### ✅ CSV Parameter Integration
- Reads ringworld parameters from `ringworld_parameters.csv`
- Supports all ringworld configurations (Bishop Ring to Seyfert Ringworlds)
- Automatic unit conversion (km to meters)
- Central mass parsing (supports formats like "1xSun", "0.05*Jupiter", "2 Mars")

### ✅ Optimized Simulation Stack
- Uses `LGST_Simulation_Wrapper.py`
- Vectorized solver (`StochasticInputRK45Solver_Vectorized.py`)
- Optimized physics (`SolverSharedCodePlusSolar_Optimized.py`)
- Fast NumPy classification (`TrajectoryClassification_numpy.py`)

### ✅ Comprehensive Logging
Each simulation creates a detailed log file containing:
- **System Information**: Python version, NumPy version, git commit, timestamp
- **Code Versions**: All module files being used
- **Ringworld Parameters**: Designation, radius, width, gravity, angular velocity, central mass
- **Simulation Parameters**: Particles, time, temperature, geometric parameters, atmospheric composition
- **Optimization Parameters**: Parallel processes, batch size, solver type
- **Results**: Duration, escape/recapture fractions, leak rate analysis
- **Error Information**: Full traceback if simulation fails

### ✅ Organized Output Structure
```
simulation_results/
├── logs/
│   ├── Bishop_Ring_10000particles_20251025_160230.log
│   ├── T5_class_Ringworld_10000particles_20251025_160245.log
│   └── ...
└── [Excel files from simulation output]
├── Bishop_Ring_10000particles_20251025_160230.xlsx
├── T5_class_Ringworld_10000particles_20251025_160245.xlsx
├── ...
```

## Usage

### Basic Usage

```bash
# Run all simulations from CSV
python run_simulation_optimized.py

# Run specific priority
python run_simulation_optimized.py --priority 1

# Run specific ringworld by designation
python run_simulation_optimized.py --designation "Bishop Ring"

# Run specific row (0-indexed)
python run_simulation_optimized.py --row 0

# Limit number of runs
python run_simulation_optimized.py --max-runs 3
```

### Parameter Overrides

```bash
# Override particle count
python run_simulation_optimized.py --row 0 --particles 10000

# Override simulation time (seconds)
python run_simulation_optimized.py --row 0 --t-max 10000

# Override time step (seconds)
python run_simulation_optimized.py --row 0 --dt 0.5

# Override temperature (Kelvin)
python run_simulation_optimized.py --row 0 --temperature 500

# Combine multiple overrides
python run_simulation_optimized.py --row 0 --particles 50000 --t-max 10000 --temperature 300
```

### Advanced Usage

```bash
# Use custom CSV file
python run_simulation_optimized.py --csv my_parameters.csv

# Save to custom directory
python run_simulation_optimized.py --results-dir my_results

# Run first 5 simulations with custom parameters
python run_simulation_v2.py --max-runs 5 --particles 5000 --t-max 8000
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv` | string | `ringworld_parameters.csv` | Path to CSV file with parameters |
| `--results-dir` | string | `simulation_results` | Directory to save results |
| `--priority` | int | None | Run only simulations with this priority |
| `--designation` | string | None | Run simulations matching this designation |
| `--row` | int | None | Run simulation for specific row index (0-based) |
| `--max-runs` | int | None | Maximum number of simulations to run |
| `--particles` | int | None | Override number of particles |
| `--t-max` | float | None | Override simulation time (seconds) |
| `--dt` | float | None | Override time step (seconds) |
| `--temperature` | float | None | Override temperature (Kelvin) |

## CSV Format

The CSV file should have the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Priority` | Simulation priority | 1 |
| `Designation` | Ringworld name | "Bishop Ring" |
| `Width (km)` | Ringworld width in km | 200 |
| `Radius (km)` | Ringworld radius in km | 1000 |
| `Gravity (m/s^2)` | Surface gravity | 9.81 |
| `Ringworld angular velocity (rad/s)` | Angular velocity | 0.003132092 |
| `Central mass` | Central body mass | "1xSun", "0.05*Jupiter", "None" |

## Default Simulation Parameters

When not overridden, the following defaults are used:

- **Particles**: 10,000
- **Simulation time**: 5,000 seconds
- **Time step**: 0.1 seconds
- **Temperature**: 289 K
- **Atmospheric composition**: Diatomic oxygen (O2)
- **Spawn altitude**: 218-228 km above surface
- **Parallel processing**: Auto-detect optimal settings

## Log File Contents

Each log file contains complete information for reproducibility:

```
======================================================================
SYSTEM INFORMATION
======================================================================
Python version: 3.x.x
NumPy version: 1.x.x
Pandas version: 2.x.x
Working directory: /path/to/ringworld
Timestamp: 2025-10-25T16:02:30.123456
Git commit: abc123def456...

======================================================================
CODE VERSIONS
======================================================================
Main simulation: LGST_Simulation_Example_Code_Usman_Updated_Optimized.py
Solver: StochasticInputRK45Solver_Vectorized.py
Physics: SolverSharedCodePlusSolar_Optimized.py
Classification: TrajectoryClassification_numpy.py
Initial conditions: StochasticInput.py
Leak rate: LeakRate.py

======================================================================
RINGWORLD PARAMETERS
======================================================================
Designation: Bishop Ring
Priority: 1
Width: 200 km
Radius: 1000 km
Gravity: 9.81 m/s²
Angular velocity: 0.003132092 rad/s
Central mass: None

======================================================================
SIMULATION PARAMETERS
======================================================================
Number of particles: 10000
Simulation time: 5000 s (1.39 hours)
Time step: 0.1 s
Temperature: 289 K
Rotating frame: False
Find leak rate: True

Geometric parameters:
  Radius (y_floor): 1000000.0 m
  Width (z_length): 200000.0 m
  Spawn altitude min (y_min): 772000.0 m
  Spawn altitude max (y_max): 782000.0 m

Atmospheric composition:
  O2: mass=5.3133924e-26 kg, charge=0, density=100 particles/m³

======================================================================
OPTIMIZATION PARAMETERS
======================================================================
Parallel processes: Auto-detect
Batch size: Auto-detect
Using vectorized solver: Yes
Using NumPy classification: Yes

======================================================================
SIMULATION RESULTS
======================================================================
Duration: 45.23 seconds (0.75 minutes)
Total particles: 10000
Escaped: 234 (2.3400%)
Recaptured: 9766 (97.6600%)
Resimulate: 0 (0.0000%)

Leak rate analysis:
  Leak rate: 1.23e-5 kg/s
  Atmospheric lifetime: 2.34e8 years
```

## Testing

Run the test script to validate the installation:

```bash
python test_run_simulation_v2.py
```

This will run a quick test with Bishop Ring using 100 particles (~1-3 seconds).

## Comparison with Original run_simulation.py

| Feature | Original | run_simulation_v2.py |
|---------|----------|---------------------|
| CSV Integration | ❌ No | ✅ Yes |
| Comprehensive Logging | ❌ No | ✅ Yes |
| Reproducibility Info | ❌ No | ✅ Yes |
| Command Line Interface | ❌ Limited | ✅ Full |
| Parameter Overrides | ❌ No | ✅ Yes |
| Organized Output | ❌ No | ✅ Yes |
| Optimized Solver | ⚠️ Partial | ✅ Full |
| Error Handling | ⚠️ Basic | ✅ Comprehensive |

## Integration with Existing Tools

`run_simulation_v2.py` works seamlessly with existing tools:

- **csv_parameter_runner.py**: Similar functionality but with different interface
- **run_quick_test.py**: Quick testing with predefined modes
- **parameter_sweep.py**: Parameter space exploration

Choose the tool based on your needs:
- Use `run_simulation_v2.py` for production runs with full logging
- Use `run_quick_test.py` for rapid testing and development
- Use `parameter_sweep.py` for systematic parameter studies

## Troubleshooting

### CSV File Not Found
```
❌ Error: CSV file 'ringworld_parameters.csv' not found
```
**Solution**: Ensure the CSV file exists in the current directory or specify the full path with `--csv`

### Import Errors
```
ModuleNotFoundError: No module named 'LGST_Simulation_Example_Code_Usman_Updated_Optimized'
```
**Solution**: Ensure you're running from the LGST_Ringworld directory

### Simulation Fails
Check the log file in `simulation_results/logs/` for detailed error information including full traceback.

## Performance Tips

1. **Start Small**: Test with `--particles 100` before running large simulations
2. **Use Row Index**: `--row 0` is faster than filtering by designation
3. **Parallel Processing**: The script auto-detects optimal CPU usage
4. **Monitor Logs**: Check log files for performance bottlenecks

## Future Enhancements

Potential improvements for future versions:
- Resume interrupted simulations
- Parallel execution of multiple ringworld configurations
- Real-time progress monitoring
- Automatic result comparison with previous runs
- Integration with analysis tools

## Support

For issues or questions:
1. Check the log file for detailed error information
2. Run the test script: `python test_run_simulation_v2.py`
3. Verify CSV format matches expected structure
4. Ensure all required modules are available

## Version History

- **v1.0** (October 2025): Initial release with comprehensive logging and CSV integration
- Based on optimized solver stack with vectorized processing and NumPy classification

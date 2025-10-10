# Parameter Sweep Tools for Ringworld Atmospheric Simulations

This directory contains tools for running parameter sweeps and analyzing results from the ringworld atmospheric simulation codebase.

## Files Overview

- **`parameter_sweep.py`** - Main wrapper script for running parameter sweeps
- **`analyze_sweep_results.py`** - Analysis and visualization tools for sweep results
- **`example_parameter_config.json`** - Example configuration file for parameter sweeps
- **`run_simulation.py`** - Original single simulation runner (unchanged)

## Quick Start

### 1. Run a Quick Test
```bash
python parameter_sweep.py --quick-test
```
This runs a small parameter sweep to verify everything is working.

### 2. Run a Gravity Parameter Sweep
```bash
python parameter_sweep.py --gravity-sweep
```
This sweeps over different gravity values to study their effect on atmospheric retention.

### 3. Run a Custom Parameter Sweep
```bash
# First create an example config file
python parameter_sweep.py --create-example

# Edit the example_parameter_config.json file as needed, then run:
python parameter_sweep.py --config example_parameter_config.json
```

### 4. Control Output Format
```bash
# Save only CSV files (faster, smaller file size)
python parameter_sweep.py --quick-test --save-format csv

# Save only Excel files (for compatibility)
python parameter_sweep.py --quick-test --save-format excel

# Save both formats (default)
python parameter_sweep.py --quick-test --save-format both
```

### 5. Analyze Results
```bash
# Analyze the most recent results
python analyze_sweep_results.py --results-dir parameter_sweep_results

# Or analyze a specific summary file
python analyze_sweep_results.py --summary-file parameter_sweep_results/gravity_sweep_summary_20250926_220000.csv
```

## Parameter Sweep Configuration

The parameter sweep system accepts any parameters that the main simulation function recognizes. Key parameters include:

### Physical Parameters
- **`gravity`** - Surface gravity (m/s²). Default: 2.743176313
- **`temperature`** - Atmospheric temperature (K). Default: 289
- **`radius`** - Ringworld radius (m). Default: 8.19381e+15
- **`z_length`** - Ringworld width (m). Default: 81938128337000

### Simulation Parameters
- **`num_particles`** - Number of particles to simulate. Default: 100
- **`t_max`** - Maximum simulation time (s). Default: 1e6
- **`dt`** - Time step (s). Default: 100
- **`is_rotating`** - Include solar gravity effects. Default: True

### Atmospheric Parameters
- **`y_min`** - Minimum spawn altitude (m)
- **`y_max`** - Maximum spawn altitude (m)
- **`comp_list`** - List of atmospheric components

## Configuration File Format

```json
{
  "sweep_name": "my_parameter_study",
  "description": "Description of the study",
  "parameter_ranges": {
    "gravity": [1.0, 2.743176313, 5.0, 10.0],
    "temperature": [200, 289, 400],
    "num_particles": [50, 100],
    "is_rotating": [true, false]
  },
  "output_settings": {
    "log_level": "INFO",
    "save_format": "csv"
  }
}
```

### Save Format Options

- **`"csv"`** - Save only CSV files (faster, smaller file size, easier to process)
- **`"excel"`** - Save only Excel files (for compatibility with existing tools)
- **`"both"`** - Save both CSV and Excel files (default if not specified)

## Output Structure

Parameter sweeps create organized output directories:

```
parameter_sweep_results/
├── logs/
│   └── parameter_sweep_20250926_220000.log
├── detailed_results/
│   ├── sweep_name_0001/
│   │   ├── sweep_name_0001_particles.csv
│   │   ├── sweep_name_0001_particles.xlsx
│   │   └── sweep_name_0001_metadata.json
│   └── sweep_name_0002/
│       └── ...
├── sweep_name_summary_20250926_220000.csv
└── sweep_name_summary_20250926_220000.xlsx
```

## Analysis Features

The analysis script provides:

1. **Summary Statistics** - Overview of escape fractions, recapture rates, etc.
2. **Parameter vs Results Plots** - Visualize how parameters affect outcomes
3. **Correlation Matrix** - Understand parameter interactions
4. **Result Distributions** - Histograms of simulation outcomes
5. **Optimal Parameter Finding** - Identify best parameter combinations

## Example Workflows

### Study Gravity Effects
```bash
# Create a gravity-focused configuration
cat > gravity_study.json << EOF
{
  "sweep_name": "gravity_study",
  "parameter_ranges": {
    "gravity": [0.5, 1.0, 2.743176313, 5.0, 9.81, 15.0, 20.0],
    "num_particles": [100]
  }
}
EOF

# Run the sweep
python parameter_sweep.py --config gravity_study.json

# Analyze results
python analyze_sweep_results.py --results-dir parameter_sweep_results
```

### Study Temperature and Rotation Effects
```bash
# Create a temperature/rotation configuration
cat > temp_rotation_study.json << EOF
{
  "sweep_name": "temperature_rotation_study",
  "parameter_ranges": {
    "temperature": [150, 200, 250, 289, 350, 400, 500],
    "is_rotating": [true, false],
    "num_particles": [100]
  }
}
EOF

# Run the sweep
python parameter_sweep.py --config temp_rotation_study.json
```

### Large-Scale Comprehensive Study
```bash
# Use the provided example configuration for a comprehensive study
python parameter_sweep.py --config example_parameter_config.json --output-dir comprehensive_study_results
```

## Performance Notes

- Each simulation can take several minutes depending on `num_particles` and `t_max`
- Start with small `num_particles` (25-50) for initial exploration
- Use larger `num_particles` (100-200) for final production runs
- The system uses the optimized NumPy trajectory classification for ~150x speedup
- Logs track progress and can help identify failed simulations

## Troubleshooting

1. **Import Errors**: Ensure you're running from the LGST_Ringworld directory
2. **Memory Issues**: Reduce `num_particles` or run fewer parameter combinations
3. **Failed Simulations**: Check logs for specific error messages
4. **Analysis Issues**: Ensure summary files contain successful simulations

## Integration with Existing Workflow

These tools are designed to work alongside your existing simulation workflow:

- The original `run_simulation.py` remains unchanged
- Results are saved in the same format (Excel files with particle data)
- The optimized trajectory classification from the memories is used automatically
- All existing analysis scripts should work with the detailed results

## Next Steps

1. Run a quick test to verify the setup
2. Explore parameter effects with focused sweeps
3. Use analysis tools to identify interesting parameter regions
4. Run comprehensive studies with optimal parameters
5. Integrate findings into your atmospheric retention models

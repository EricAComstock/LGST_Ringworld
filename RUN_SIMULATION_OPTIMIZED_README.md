# run_simulation_optimized.py – Production Simulation Runner

## Overview

`run_simulation_optimized.py` is the production entry point for running atmospheric retention simulations across the full catalog of ringworld configurations. It couples CSV-driven parameter management with the optimized solver stack, orbital-aware timing, and comprehensive logging so that every run is reproducible and auditable.@LGST_Ringworld/run_simulation_optimized.py#1-310

## Quick Start

```bash
# Run every configuration in ringworld_parameters.csv
python run_simulation_optimized.py

# Target a single configuration by row (0-indexed)
python run_simulation_optimized.py --row 0

# Filter by designation or priority
python run_simulation_optimized.py --designation "Bishop Ring"
python run_simulation_optimized.py --priority 1

# Limit how many rows execute
python run_simulation_optimized.py --max-runs 3
```

Logs and Excel output are saved into `simulation_results/`, grouped by timestamped run IDs.

## Feature Summary

1. **CSV-driven parameter pipeline** – Loads all configuration columns, converts kilometres to metres, and parses free-form central-mass descriptors before building the simulation payload.@LGST_Ringworld/run_simulation_optimized.py#319-509
2. **Orbital-aware timing with memory safeguards** – Simulation horizon (`t_max`) and step size (`dt`) are derived from the ringworld’s angular velocity, automatically reducing orbit count for very long periods and capping the total number of steps at 200,000 to avoid memory exhaustion.@LGST_Ringworld/run_simulation_optimized.py#349-395; @MEMORY[b6a9f6bc-7309-4b31-be13-f744f71ebdde]
3. **Optimized solver stack** – Delegates execution to `LGST_Simulation_Wrapper.run_simulation`, which wraps the vectorized solver, optimized physics engine, NumPy trajectory classification, and leak-rate analysis.@LGST_Ringworld/run_simulation_optimized.py#41-47; @LGST_Ringworld/LGST_Simulation_Wrapper.py#1-168; @MEMORY[94b46c62-5bb8-4182-a31b-385042c8956d]
4. **Full reproducibility logging** – Each run produces a timestamped `.log` capturing system info, code versions, input parameters (including derived orbital metrics), solver inputs, parallelization settings, and end-state statistics.@LGST_Ringworld/run_simulation_optimized.py#49-300
5. **Batch execution supervisor** – Filters by priority, designation, or row and collates results/summary statistics across all requested simulations.@LGST_Ringworld/run_simulation_optimized.py#618-667

## Architecture & Module Dependencies

```
run_simulation_optimized.py
  └── LGST_Simulation_Wrapper.run_simulation()
        └── StochasticInputRK45Solver_Vectorized.main_vectorized()
              ├── SolverSharedCodePlusSolar_Optimized.compute_motion()
              ├── StochasticInput.stochastic_initial_conditions()
              ├── TrajectoryClassification_numpy.classify_trajectory()
              └── LeakRate.*
```

- **LGST_Simulation_Wrapper.py** – Pass-through wrapper ensuring all geometric and solver parameters arrive explicitly, preventing hidden defaults.@LGST_Ringworld/LGST_Simulation_Wrapper.py#24-168
- **StochasticInputRK45Solver_Vectorized.py** – Parallel/vectorized engine with intelligent batching and CPU detection for high particle counts.@LGST_Ringworld/StochasticInputRK45Solver_Vectorized.py#1-400
- **SolverSharedCodePlusSolar_Optimized.py** – Float64-safe physics integrator with cached angular velocity and overflow-protected solar gravity calculations.@LGST_Ringworld/SolverSharedCodePlusSolar_Optimized.py#1-240
- **TrajectoryClassification_numpy.py** – NumPy-based classifier that avoids integer overflow and mirrors the original logic, fixing the historical radial-distance bug.@MEMORY[890e1591-1e54-4e80-8269-84b8468d2a49]; @MEMORY[bab19fcc-cc8e-445e-b5ee-a43aaca121f2]
- **LeakRate.py** – Computes leak rate and atmospheric lifetime using the final particle classifications.

All modules are initialized per-run via `SSCPSVarInput`, `SIVarInput`, `TCVarInput`, and `LRVarInput` before the solver executes.@LGST_Ringworld/run_simulation_optimized.py#511-525

## Parameter Handling

### Required CSV Columns

| Column | Description | Notes |
| --- | --- | --- |
| `Priority` | Scheduling priority | Integer filterable via `--priority` |
| `Designation` | Ringworld name | Used in log/run ID generation |
| `Width (km)` | Ring plane width | Converted to `z_length` (metres) |
| `Radius (km)` | Radial distance to floor | Converted to `radius`, `y_floor` |
| `Gravity (m/s^2)` | Surface gravity | Passed to solver and leak rate |
| `Ringworld angular velocity (rad/s)` | Rotation rate | Drives orbital timing logic |
| `Central mass` | e.g. `1xSun`, `0.05*Jupiter`, `None` | Parsed into `solar_mu` and toggles rotating frame |
| `Atmosphere Thickness (km)` *(optional)* | Overrides default 218 km | Determines spawn region |
| `Spawn Range (km)` *(optional)* | Overrides default 10 km | Controls `y_min` buffer |

### Derived Simulation Inputs

- `radius` / `y_floor` – Base altitude for the floor (metres).
- `z_length` – Ring width (metres).
- `y_min`, `y_max` – Spawn band inside the atmosphere (`floor - thickness ± range`).
- `alpha`, `beta` – Passed to classification during module initialization.
- `is_rotating` – Enabled automatically when a central mass is supplied so the solver includes solar gravity.

### Command Line Arguments

| Argument | Type | Default | Effect |
| --- | --- | --- | --- |
| `--csv` | str | `ringworld_parameters.csv` | Override CSV source |
| `--results-dir` | str | `simulation_results` | Change output root |
| `--priority` | int | `None` | Filter by priority |
| `--designation` | str | `None` | Case-insensitive substring match |
| `--row` | int | `None` | Run a single row (0-index) |
| `--max-runs` | int | `None` | Limit number of rows processed |
| `--particles` | int | `None` | Override particle count |
| `--t-max` | float | `None` | Force simulation horizon |
| `--dt` | float | `None` | Force time step |
| `--temperature` | float | `None` | Override Maxwell-Boltzmann temperature |
| `--num-orbits` | float | `None` | Recalculate `t_max`/`dt` using orbital-period logic |

Parameter overrides are merged last so you can blend CSV data with ad-hoc experiments (e.g. `--row 0 --particles 50000 --num-orbits 2`).@LGST_Ringworld/run_simulation_optimized.py#706-759

## Output & Logging

- **Excel results** – Particle trajectories, velocities, and classifications saved to `<results_dir>/<run_id>.xlsx` when `save_results=True` (default).@LGST_Ringworld/run_simulation_optimized.py#566-585; @LGST_Ringworld/StochasticInputRK45Solver_Vectorized.py#373-399
- **Logs** – `<results_dir>/logs/<run_id>.log` capturing:
  - System / environment snapshot
  - Code versions (module list)
  - Source CSV data for the ringworld row
  - Simulation parameters (including orbital-period breakdown and derived spawn geometry)
  - Exact arguments passed to the solver
  - Parallelization strategy (cores, batch size)
  - Aggregate results and leak-rate metrics
  - Full tracebacks for any exception

If particles land in the “resimulate” bucket, the log explicitly warns that `t_max` might need to increase.@LGST_Ringworld/run_simulation_optimized.py#248-276

### Log Analysis Utility (`parse_simulation_logs.py`)

- Run `python parse_simulation_logs.py` to aggregate every log in `simulation_results/logs/` into a single CSV report stored at `simulation_results/compiled_results/compiled_simulation_logs.csv`.@LGST_Ringworld/parse_simulation_logs.py#313-334
- The parser extracts timestamps, git commit hashes, configuration metadata (designation, priority, geometric parameters), orbital timing details, solver stack versions, parallelization settings, and outcome statistics (escaped/recaptured/resimulate counts and percentages).@LGST_Ringworld/parse_simulation_logs.py#16-259
- Missing directories or log files are handled gracefully, and the script reports how many files were processed plus the number of columns written so you can validate coverage quickly.@LGST_Ringworld/parse_simulation_logs.py#266-311

## Recommended Workflow

1. **Dry run:** start with `--row <index> --particles 100` to verify the configuration and inspect the log.
2. **Scale up:** bump `--particles` and optionally `--num-orbits` to ensure the bulk of particles finish without resimulation.
3. **Full sweep:** remove filters or use `--max-runs` for staged execution across the catalog.
4. **Review outputs:** compare `.log` summaries and Excel results; feed them into downstream analysis scripts such as `parameter_sweep.py` or custom notebooks.

## Operational Safeguards

- **Memory protection** – Automatic cap at 200,000 steps per run limits per-particle memory use to ~12 MB while still allowing multi-orbit simulations; overrides (`--t-max`, `--dt`) respect the cap by increasing step size when necessary.@LGST_Ringworld/run_simulation_optimized.py#380-394
- **Numerical stability** – All physics calculations run in float64 across the solver stack, fixing historic overflow issues in both orbital dynamics and trajectory classification.@LGST_Ringworld/SolverSharedCodePlusSolar_Optimized.py#68-240; @MEMORY[890e1591-1e54-4e80-8269-84b8468d2a49]
- **Parameter sanity** – Central-mass parsing ignores malformed strings rather than crashing, keeping runs resilient to partial CSV data.@LGST_Ringworld/run_simulation_optimized.py#397-430

## Troubleshooting

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `CSV file ... not found` | Wrong working directory | Launch commands from `LGST_Ringworld/` or pass `--csv` with an absolute path.@LGST_Ringworld/run_simulation_optimized.py#723-726 |
| All particles show `resimulate` | Simulation horizon too short | Increase `--num-orbits` or `--t-max`; review log warnings for capped step counts.@LGST_Ringworld/run_simulation_optimized.py#166-276 |
| `ModuleNotFoundError` for optimized modules | Script executed outside project root | Ensure `PYTHONPATH` includes the repository root or run from the project directory.
| Zero escapes at extreme temperatures | Indicates earlier dictionary parameter bug – confirmed resolved in the optimized stack; ensure you are running this script (not legacy versions).@MEMORY[f73a3faa-d709-4f66-a63f-e18edfd96dd0]
| Long runtimes on ultra-large rings | dt expanded to honor step cap | Explicitly set `--dt` or reduce `--num-orbits`; logs show the capped value used.

## Integration with Other Tools

- **`parameter_sweep.py`** – Uses the same optimized stack for systematic studies, so logs and outputs remain consistent.@MEMORY[82d81894-a4cd-4fc7-9d83-92f12edca9cf]
- **`run_quick_test.py` / `csv_parameter_runner.py`** – Compatible wrappers for targeted or scripted workflows.@MEMORY[94b46c62-5bb8-4182-a31b-385042c8956d]
- **Legacy scripts** – `run_simulation.py` and other exploratory utilities remain available but do not benefit from the vectorized optimizations.

## Version History

- **v1.0 (Oct 2025)** – First optimized release featuring CSV integration, orbital-based timing, solver stack upgrades, float64 trajectory classification, and reproducibility logging.


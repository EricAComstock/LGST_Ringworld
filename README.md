# Ringworld Particle Dynamics Simulation
**Version 1.1**  
**Last Updated: August 25, 2025**

===============================================================================

## Overview

This computational framework simulates the atmospheric retention dynamics
of a rotating ringworld structure. It models the phase-space evolution
of molecular constituents in the exospheric region to determine whether
they exceed escape velocity or are recaptured by the ringworld's
gravitational potential well.

The simulation implements a collisionless approach where particles evolve
independently without mutual interactions. The system operates in a
non-inertial rotating reference frame where Coriolis and centrifugal
pseudoforces dominate the particle kinematics.

===============================================================================

## Coordinate System

- **Origin**: Heliocentric (sun-centered)
- **X-Y plane**: Equatorial plane of the ringworld structure  
- **Z-axis**: Orthogonal to the ringworld plane
- **Reference**: Non-inertial rotating frame with angular velocity ω about z-axis

===============================================================================

## Component Modules

### StochasticInput.py
**Purpose**: Generates initial conditions via stochastic sampling  
**Features**:
- Position vectors within defined spatial domains
- Velocity vectors from Maxwell-Boltzmann distribution
- Configurable molecular properties (species, wieght, charge)

### StochasticInputRK45Solver.py
**Purpose**: Primary execution module  
**Features**:
- Adaptive Runge-Kutta particle trajectory integration
- Classification of escape and recapture events
- Statistical analysis of ensemble behavior
- Data visualization and persistence

### SolverSharedCodePlusSolar.py
**Purpose**: Physics engine  
**Features**:
- Dynamical equations in the rotating reference frame
- Runge-Kutta 45 numerical integration
- Coriolis and centrifugal force calculations
- Optional heliocentric gravitational field
- Lorentz force calculation

### TrajectoryClassification.py
**Purpose**: Analyzes trajectories against boundary conditions  
**Features**:
- Escape threshold detection
- Recapture event identification
- Boundary crossing quantification

### LeakRate.py
**Purpose**: Analyzes simulation results to determine atmospheric loss  
**Features**:
- Processes particle classification data (escaped/recaptured)
- Calculates molecular escape fraction
- Computes mass flux rate using kinetic theory
- Estimates total atmospheric lifetime in years

### tester-mullis-NEW.py
**Purpose**: Verification module  
**Features**:
- Reference frame transformations
- Analytical vs. numerical solution comparison
- Error quantification and validation

### tester-chambers.py ###
**Purpose**: Additional Verification module
**Features**:
- Preliminary testing of Electromagnetic effects in the ringworld via Lorentz force

### Validation_error_plot.py ###
**Purpose**: Error Visualization
**Features**:
- Plot showing error of integration method over time

### Ringworld_gravity.py ###
**Purpose**: Visualizes gravity caused by ringworld
**Features**:
 - Heatmap of RW gravity

===============================================================================

## Execution Instructions

### Primary Simulation

1. **Run command**:
   ```
   python StochasticInputRK45Solver.py
   ```

2. **Expected outcomes**:
   - Progress updates displayed in console (particle processing status)
   - Summary statistics shown after completion (escape/recapture percentages)
   - Trajectory plots generated (if show_plots=True)
   - Results file created with timestamp (particle_data_YYYYMMDD_HHMMSS.xlsx)

3. **Output files**:
   - Excel spreadsheet containing:
     * Initial positions and velocities
     * Final positions and velocities
     * Boundary crossing counts
     * Trajectory classification results

4. **Visualization**:
   - X-Y Plot: Horizontal particle trajectories with atmosphere boundary
   - Z-Y Plot: Vertical trajectories with all boundary lines
   - Color-coded paths to distinguish multiple particles

5. **Parameter adjustment**:
   Edit configuration parameters in the `__main__` block:
   - `t_max`: Simulation duration
   - `dt`: Time step size
   - `num_particles`: Number of particles to simulate
   - `comp_list`: Composition of relivant atmosphere

### Atmospheric Leak Rate Analysis

The leak rate analysis is integrated into the main simulation and runs
automatically after particle processing completes.

### Verification Tools ###
**Mullis**
**Run command**:
```
python tester-mullis-NEW.py
```

**Inputs**:
- Initial positions and velocities in inertial frames
- Initial and final simulation time
- Angular velocity
- Gravitational acceleration constant

**Outputs**:
- Initial/final positions in inertial and rotating frames
- Calculated vs. numerical position comparisons
- Error metrics showing solver accuracy
- Angular velocity and rotation period information

**Chambers**
**Run command**:
```
python tester-chambers.py
```

**Inputs**:
 - E & B fields as vectors or vector functions at a location/time
 - Particle velocity
 - Mass & Charge

**Outputs**:
 - Lorentz acceleration as a 3D vector

===============================================================================

## Configuration Parameters

### StochasticInputRK45Solver.py
| Parameter | Description | Units |
|-----------|-------------|-------|
| t_max | Simulation time | s |
| dt | Integration time step | s |
| num_particles | Number of particles | - |
| is_rotating | Reference frame selection | bool |
| radius | Ringworld radius | m |
| gravity | Gravitational acceleration | m/s² |
| save_results | Data persistence control | bool |
| show_plots | Visualization toggle | bool |
| comp_list | Atmoshere Composition | [(String,kg,C,n/m^3)] |
| find_leak_rate | Use Leak Rate code toggle | bool |

### TrajectoryClassification.py
| Parameter | Description | Units |
|-----------|-------------|-------|
| z_length | Total length of ringworld | m |
| beta | Lateral boundary (z_length/2) | m |
| y_floor | Lower vertical boundary (1 AU) | m |
| alpha | Atmospheric threshold | m |
| y_min | Minimum particle spawn height | m |
| y_max | Maximum particle spawn height | m |

### StochasticInput.py
| Parameter | Description | Units |
|-----------|-------------|-------|
| T | Temperature | K |
| k_B | Boltzmann constant | J/K |
| scale | Velocity scale parameter | m/s |

### SolverSharedCodePlusSolar.py
| Parameter | Description | Units |
|-----------|-------------|-------|
| G_UNIVERSAL | Gravitational constant | m³/kg·s² |
| solar_mu | Heliocentric parameter | m³/s² |
| rtol, atol | Integration tolerance | - |
| omega | Angular velocity vector | rad/s |

### LeakRate.py
| Parameter | Description | Units |
|-----------|-------------|-------|
| P_0 | Atmospheric pressure | Pa |
| K_b | Boltzmann constant | J/K |
| T_0 | Standard temperature | K |
| m | Molecular mass | kg |
| g | Gravitational acceleration | m/s² |
| n_0 | Molecular density | 1/m³ |
| d | Molecular diameter | m |

===============================================================================

## Physical Models

1. **Coriolis Effect**: F = -2m(ω × v)

2. **Centrifugal Effect**: F = -m(ω × (ω × r))

3. **Maxwell-Boltzmann Distribution**: f(v) ∝ v²exp(-mv²/2kT)

4. **Boundary Dynamics**: Classification based on geometric thresholds

5. **Atmospheric Leak Rate Model**:
   - Escape Fraction: f_escape = N_escaped / N_recaptured
   - Mean Free Path: λ = (σ·n₀)⁻¹, where σ = π·d²/4
   - Scale Height: h_s = k·T₀/(m·g)
   - Exobase Altitude: alt = h_s·ln(h_s/λ)
   - Molecular Density: n = n₀·e^(-alt/h_s)
   - Mass Flux: φ_m = n·m·v_thermal·f_escape
   - Atmospheric Lifetime: t_life = (P₀/g)/φ_m (years)
  
6. **Lorentz Force**: F = q(E + v × B)

===============================================================================

## Parameter Modification

To modify simulation parameters:

1. Edit values in their respective files as detailed in the
   "Configuration Parameters" section

2. For quick parameter adjustments, modify values in the
   `if __name__ == "__main__"` block at the bottom of 
   StochasticInputRK45Solver.py

===============================================================================

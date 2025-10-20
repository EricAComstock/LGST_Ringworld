"""
LGST_Simulation_Example_Code_Usman_Updated_Optimized.py
Optimized version using vectorized solver for improved performance.

Uses:
- StochasticInputRK45Solver_Vectorized.py (parallel processing)
- SolverSharedCodePlusSolar_Optimized.py (optimized physics)
- TrajectoryClassification_numpy.py (fast NumPy classification)

V1.1 Optimized, October 2025
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Import optimized components
from SolverSharedCodePlusSolar_Optimized import compute_motion, SSCPSVarInput
from StochasticInput import stochastic_initial_conditions, SIVarInput
from TrajectoryClassification_numpy import classify_trajectory, TCVarInput
from LeakRate import LRVarInput
from StochasticInputRK45Solver_Vectorized import main_vectorized


def main(radius, gravity, t_max, dt, is_rotating=False, num_particles=100,
         save_results=True, show_plots=False, find_leak_rate=True, comp_list=None,
         num_processes=None, batch_size=None):
    """
    Optimized main simulation function using vectorized processing.
    
    This function wraps the vectorized solver to maintain compatibility with
    the original interface while providing significant performance improvements
    through parallel processing and optimized algorithms.

    Inputs:
    radius          Radius for calculating omega [m]
    gravity         Gravity for calculating omega [m/s²]
    t_max           Maximum simulation time [s]
    dt              Time step [s]
    is_rotating     Whether reference frame is rotating [bool]
    num_particles   Number of particles to simulate [int]
    save_results    Whether to save results to Excel [bool]
    show_plots      Whether to display trajectory plots [bool]
    find_leak_rate  Whether to calculate leak rate [bool]
    comp_list       Composition of the atmosphere [list of tuples] (Name, mass, charge, number density)
    num_processes   Number of parallel processes (None = auto-detect) [int]
    batch_size      Particles per batch for parallel processing (None = auto-detect) [int]

    Outputs:
    results  DataFrame containing all particle simulation results
    """
    
    # Call the vectorized solver with all parameters
    return main_vectorized(
        radius=radius,
        gravity=gravity,
        t_max=t_max,
        dt=dt,
        is_rotating=is_rotating,
        num_particles=num_particles,
        save_results=save_results,
        show_plots=show_plots,
        find_leak_rate=find_leak_rate,
        comp_list=comp_list,
        num_processes=num_processes,
        batch_size=batch_size
    )


# Main execution
if __name__ == "__main__":
    # Simulation parameters
    t_max = 5000  # Total simulation time [s]
    dt = 0.1  # Time step [s]
    num_particles = 100  # Number of particles to simulate

    # Physical parameters - modify gravity for different simulations
    g = 9.81                   # 1.0g - Standard gravity [m/s²]
    # g = 19.62                  # 2.0g - Double gravity [m/s²]
    # g = 4.905                  # 0.5g - Half gravity [m/s²]
    # g = 14.715                 # 1.5g [m/s²]
    # g = 2.4525                 # 0.25g [m/s²]
    # g = 29.43                  # 3.0g [m/s²]
    # g = 0.01269381              # Custom Gravity
    G = 6.6743e-11               # Universal gravitational constant [m^3/kg/s^2]

    # Atmospheric parameters
    T = 289  # Temperature [K]
    P_0 = 101325  # Atmospheric pressure at sea level [Pa]
    K_b = 1.380649e-23  # Boltzmann constant [J/K]
    m = 2.6566962e-26 * 2  # Mass of diatomic molecule [kg]
    n_0 = 2.687e25  # Molecular density [1/m³]
    d = 3.59e-10  # Molecular diameter [m]

    # Geometric parameters
    z_length = 200 * 1000  # Ringworld width [m]
    y_floor = 1000 * 1000  # Floor value [m]
    beta = z_length / 2  # Lateral boundary [m]
    alpha = y_floor - (218 * 1000)  # Atmosphere boundary [m]
    y_min = alpha - 10000  # Min spawn height [m]
    y_max = alpha  # Max spawn height [m]

    # Atmospheric composition
    diatomic_oxygen = ("O2", 2.6566962e-26 * 2, 0, 100)  # diatomic oxygen
    comp_list = [diatomic_oxygen]  # collection of all species at desired altitude

    # Initialize all modules with parameters
    SSCPSVarInput(G)
    SIVarInput(T, y_min, y_max, z_length, y_floor)
    TCVarInput(z_length, beta, y_floor, alpha, y_min, y_max)
    LRVarInput(P_0, K_b, T, m, g, n_0, d)

    # Performance optimization settings
    # Set num_processes=None for auto-detection (recommended)
    # Set num_processes=1 for single-threaded (useful for debugging)
    # Set num_processes=N for specific number of parallel processes
    num_processes = None  # Auto-detect optimal number of processes
    batch_size = None     # Auto-detect optimal batch size

    print("=" * 70)
    print("OPTIMIZED RINGWORLD ATMOSPHERIC SIMULATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Particles: {num_particles}")
    print(f"  Simulation time: {t_max} s")
    print(f"  Time step: {dt} s")
    print(f"  Gravity: {g} m/s²")
    print(f"  Temperature: {T} K")
    print(f"  Ringworld width: {z_length/1000} km")
    print(f"  Parallel processing: {'Auto-detect' if num_processes is None else f'{num_processes} processes'}")
    print("=" * 70)

    # Run optimized simulation
    start_time = datetime.now()
    
    results = main(
        radius=y_min,                   # Use y_min as radius
        gravity=g,                      # Selected gravity value
        t_max=t_max,
        dt=dt,
        is_rotating=False,              # Solar gravity disabled
        num_particles=num_particles,
        find_leak_rate=True,            # Calculate atmospheric lifetime
        comp_list=comp_list,
        num_processes=num_processes,    # Parallel processing
        batch_size=batch_size           # Batch size
    )
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("=" * 70)
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Average time per particle: {elapsed_time/num_particles*1000:.2f} ms")
    print("=" * 70)

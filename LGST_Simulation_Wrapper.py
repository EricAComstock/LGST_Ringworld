"""
LGST_Simulation_Wrapper.py
Clean wrapper for optimized simulation without hardcoded parameters.

This module provides a parameter-free interface to the optimized simulation stack.
All parameters must be passed explicitly to prevent silent failures from hardcoded values.

Uses:
- StochasticInputRK45Solver_Vectorized.py (parallel processing)
- SolverSharedCodePlusSolar_Optimized.py (optimized physics)
- TrajectoryClassification_numpy.py (fast NumPy classification)

Version: 2.0 - Parameter-free wrapper
Date: October 2025
"""

import numpy as np
import pandas as pd

# Import optimized components
from StochasticInputRK45Solver_Vectorized import main_vectorized


def run_simulation(radius, gravity, t_max, dt, y_min, y_max, y_floor, z_length,
                   is_rotating=False, num_particles=100,
                   save_results=True, show_plots=False, find_leak_rate=True, comp_list=None,
                   num_processes=None, batch_size=None, output_dir=None, output_filename=None):
    """
    Run optimized ringworld atmospheric simulation.
    
    This is a clean wrapper that passes all parameters directly to the vectorized solver
    without any hardcoded defaults. All parameters must be provided by the caller.
    
    Parameters
    ----------
    radius : float
        Radius for calculating omega [m]
        Note: This is typically set to y_min (spawn altitude minimum)
    gravity : float
        Surface gravity [m/s²]
    t_max : float
        Maximum simulation time [s]
    dt : float
        Time step [s]
    y_min : float
        Minimum y-coordinate for particle spawning [m]
    y_max : float
        Maximum y-coordinate for particle spawning [m]
    y_floor : float
        Ringworld floor/surface y-coordinate [m]
    z_length : float
        Total ringworld width in z-direction [m]
    is_rotating : bool, optional
        Whether reference frame is rotating (default: False)
        Set to True when central mass is present
    num_particles : int, optional
        Number of particles to simulate (default: 100)
    save_results : bool, optional
        Whether to save results to Excel (default: True)
    show_plots : bool, optional
        Whether to display trajectory plots (default: False)
    find_leak_rate : bool, optional
        Whether to calculate leak rate (default: True)
    comp_list : list of tuples, optional
        Atmospheric composition as list of (Name, mass, charge, number_density)
        Example: [("O2", 2.6566962e-26 * 2, 0, 100)]
    num_processes : int or None, optional
        Number of parallel processes (None = auto-detect, default: None)
    batch_size : int or None, optional
        Particles per batch for parallel processing (None = auto-detect, default: None)
    output_dir : str or None, optional
        Directory to save output files (default: current directory)
    output_filename : str or None, optional
        Base filename for output (without extension, default: auto-generated)
    
    Returns
    -------
    pandas.DataFrame
        Simulation results containing particle trajectories and classifications
        
    Notes
    -----
    - All module initialization (SSCPSVarInput, SIVarInput, TCVarInput, LRVarInput)
      must be done by the caller BEFORE calling this function
    - This function does NOT initialize any modules to avoid parameter conflicts
    - The caller is responsible for ensuring all modules are properly initialized
    
    Examples
    --------
    >>> # Initialize modules first
    >>> from SolverSharedCodePlusSolar_Optimized import SSCPSVarInput
    >>> from StochasticInput import SIVarInput
    >>> from TrajectoryClassification_numpy import TCVarInput
    >>> from LeakRate import LRVarInput
    >>> 
    >>> # Physical constants
    >>> G = 6.6743e-11
    >>> T = 289
    >>> P_0 = 101325
    >>> K_b = 1.380649e-23
    >>> m = 2.6566962e-26 * 2
    >>> n_0 = 2.687e25
    >>> d = 3.59e-10
    >>> 
    >>> # Geometric parameters
    >>> radius_m = 1000 * 1000
    >>> width_m = 200 * 1000
    >>> y_floor = radius_m
    >>> y_min = y_floor - 218000 - 10000
    >>> y_max = y_floor - 218000
    >>> alpha = y_floor - 218000
    >>> beta = width_m / 2
    >>> 
    >>> # Initialize all modules
    >>> SSCPSVarInput(G)
    >>> SIVarInput(T, y_min, y_max, width_m, y_floor)
    >>> TCVarInput(width_m, beta, y_floor, alpha, y_min, y_max)
    >>> LRVarInput(P_0, K_b, T, m, 9.81, n_0, d)
    >>> 
    >>> # Run simulation
    >>> comp_list = [("O2", 2.6566962e-26 * 2, 0, 100)]
    >>> results = run_simulation(
    ...     radius=y_min,
    ...     gravity=9.81,
    ...     t_max=5000,
    ...     dt=0.1,
    ...     num_particles=1000,
    ...     comp_list=comp_list
    ... )
    """
    
    # Validate required parameters
    required_params = {
        'radius': radius, 'gravity': gravity, 't_max': t_max, 'dt': dt,
        'y_min': y_min, 'y_max': y_max, 'y_floor': y_floor, 'z_length': z_length
    }
    missing = [k for k, v in required_params.items() if v is None]
    if missing:
        raise ValueError(f"Required parameters missing: {', '.join(missing)}")
    
    # Build sim_params dict with all geometric parameters
    sim_params = {
        'y_min': y_min,
        'y_max': y_max,
        'y_floor': y_floor,
        'z_length': z_length,
    }
    
    # Call the vectorized solver with all parameters
    # Geometric parameters passed via sim_params to override defaults
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
        batch_size=batch_size,
        output_dir=output_dir,
        output_filename=output_filename,
        sim_params=sim_params
    )


# No __main__ block - this is a pure library module
# All parameters must come from the caller

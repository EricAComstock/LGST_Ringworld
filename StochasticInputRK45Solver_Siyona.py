"""
StochasticInputRK45Solver.py
Main simulation driver that runs particle trajectory simulations using RK45
integration and classifies atmospheric escape/recapture outcomes.

V1.0, Edwin Ontiveros, April 29, 2025
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Path operations
from SolverSharedCodePlusSolar import compute_motion, SSCPSVarInput
from StochasticInput import stochastic_initial_conditions, SIVarInput
from TrajectoryClassification_numpy import classify_trajectory, TCVarInput
from LeakRate import LRVarInput

# Simulation parameters
SIMULATION_PARAMS = {
    # Core simulation parameters
    'radius': 1.495978707e11,  # 1 AU in meters
    'gravity': 9.81,           # Standard gravity [m/s²]
    't_max': 100,              # Total simulation time [s]
    'dt': 0.1,                 # Time step [s]
    'is_rotating': False,       # Solar gravity disabled by default
    'num_particles': 100,      # Number of particles to simulate
    'save_results': True,      # Save results to file
    'show_plots': False,       # Show trajectory plots
    'find_leak_rate': True,    # Calculate atmospheric lifetime
    'temperature': 289,        # Temperature [K]
}

# Ringworld physical parameters
RINGWORLD_PARAMS = {
    'y_floor': 149597870691,   # Floor value (1 AU) [m]
    'y_min': 149597870691 - 10000,  # Min spawn height [m]
    'y_max': 149597870691,     # Max spawn height [m]
    'z_length': 1000 * 1000,   # Ringworld width [m]
}

# Atmospheric parameters
ATMOSPHERE_PARAMS = {
    'comp_list': [
        ("O2", 2.6566962e-26 * 2, 0, 100)  # (name, mass, charge, density)
    ]
}

# Leak rate calculation parameters
LEAK_RATE_PARAMS = {
    'P_0': 101325,             # Atmospheric pressure at sea level [Pa]
    'K_b': 1.380649e-23,       # Boltzmann constant [J/K]
    'molecular_mass': 2.6566962e-26 * 2,  # Mass of diatomic molecule [kg]
    'n_0': 2.687e25,           # Molecular density [1/m³]
    'molecular_diameter': 3.59e-10,  # Molecular diameter [m]
}

# Combine all parameters into DEFAULT_PARAMS
DEFAULT_PARAMS = {}
DEFAULT_PARAMS.update(SIMULATION_PARAMS)
DEFAULT_PARAMS.update(RINGWORLD_PARAMS)
DEFAULT_PARAMS.update(ATMOSPHERE_PARAMS)
DEFAULT_PARAMS.update(LEAK_RATE_PARAMS)

def get_simulation_params(override_params=None):
    """
    Get simulation parameters with optional overrides.
    
    Args:
        override_params (dict, optional): Parameters to override defaults.
            Can include parameters from SIMULATION_PARAMS, RINGWORLD_PARAMS,
            ATMOSPHERE_PARAMS, or LEAK_RATE_PARAMS.
        
    Returns:
        dict: Complete set of simulation parameters with derived values
    """
    # Start with default parameters
    params = DEFAULT_PARAMS.copy()
    
    # Apply any overrides
    if override_params:
        params.update(override_params)
    
    # Calculate derived parameters
    params['beta'] = params['z_length'] / 2  # Lateral boundary [m]
    params['alpha'] = params['y_floor'] - (218 * 1000)  # Atmosphere boundary [m]
    
    return params



def main(radius=None, gravity=None, t_max=None, dt=None, is_rotating=None, 
         num_particles=None, save_results=None, show_plots=None, 
         find_leak_rate=None, comp_list=None, sim_params=None):
    """
    Main simulation function for particle trajectory analysis.

    Args:
        All parameters are optional and will use defaults if not provided.
        Pass either individual parameters or a sim_params dictionary.
        
    Returns:
        DataFrame: Simulation results
    """
    # Handle parameters
    if sim_params is None:
        sim_params = {}
    
    # Update with any explicitly passed parameters
    local_params = {
        'radius': radius,
        'gravity': gravity,
        't_max': t_max,
        'dt': dt,
        'is_rotating': is_rotating,
        'num_particles': num_particles,
        'save_results': save_results,
        'show_plots': show_plots,
        'find_leak_rate': find_leak_rate,
        'comp_list': comp_list
    }
    
    # Remove None values (parameters that weren't passed)
    local_params = {k: v for k, v in local_params.items() if v is not None}
    sim_params.update(local_params)
    
    # Get final parameter set with defaults
    params = get_simulation_params(sim_params)
    
    all_data = []
    print(f"Processing {params['num_particles']} particles...")

    # Create plots if requested
    if params['show_plots']:
        plt.figure(1, figsize=(10, 8))
        plt.title("X-Y Trajectories")
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")

        plt.figure(2, figsize=(10, 8))
        plt.title("Z-Y Trajectories")
        plt.xlabel("Z Position [m]")
        plt.ylabel("Y Position [m]")

    # Process each particle
    for i in range(params['num_particles']):
        # Progress indicator - every 5%
        fiveP = num_particles / 20
        if (i % fiveP == 0):
            percent = 100 * i / num_particles
            print(str(percent) + "% percent done")

        # Generate initial conditions using parameters from params
        initial_state = stochastic_initial_conditions(
            T=params['temperature'],
            y_min=params['y_min'],
            y_max=params['y_max'],
            z_length=params['z_length'],
            comp_list=params['comp_list']
        )
        initial_position = initial_state[0:3]
        initial_velocity = initial_state[4:7]

        try:
            # Compute trajectory
            final_position, final_velocity, solution = compute_motion(
                initial_position, initial_velocity, radius, gravity, t_max, dt, None
            )

            # Extract trajectory data
            trajectory = solution.y[:3, :].T  # Shape: (n_timesteps, 3)

            # Convert to DataFrame for classification
            solution_df = pd.DataFrame(trajectory, columns=[0, 1, 2])

            # Get trajectory classification parameters with defaults
            y_floor = sim_params.get('y_floor', 149597870691) if sim_params else 149597870691
            z_length = sim_params.get('z_length', 1000 * 1000) if sim_params else 1000 * 1000
            alpha = y_floor - (218 * 1000)
            beta = z_length / 2
            
            # Classify trajectory
            beta_crossings, result = classify_trajectory(
                alpha=alpha,
                beta=beta,
                y_floor=y_floor,
                trajectories=solution_df
            )

            # Store particle data
            particle_data = {
                'Particle #': i + 1,
                'Initial x': initial_position[0],
                'Initial y': initial_position[1],
                'Initial z': initial_position[2],
                'Initial vx': initial_velocity[0],
                'Initial vy': initial_velocity[1],
                'Initial vz': initial_velocity[2],
                'Final x': final_position[0],
                'Final y': final_position[1],
                'Final z': final_position[2],
                'Final vx': final_velocity[0],
                'Final vy': final_velocity[1],
                'Final vz': final_velocity[2],
                'Beta crossings': beta_crossings,
                'Result': result
            }

            all_data.append(particle_data)

            # Plot trajectory if requested
            if show_plots:
                plt.figure(1)
                plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7)

                plt.figure(2)
                plt.plot(trajectory[:, 2], trajectory[:, 1], alpha=0.7)

        except Exception as e:
            print(f"Error processing particle {i + 1}: {e}")
            traceback.print_exc()

    # Create results DataFrame
    df = pd.DataFrame(all_data)

    # Add reference lines to plots
    if params['show_plots']:
        # Configure plot 1 (X-Y)
        plt.figure(1)
        plt.axhline(y=params['alpha'], color='r', linestyle='--', label='Alpha (atmosphere top)')
        plt.axhline(y=params['y_min'], color='g', linestyle='--', label='Y min (spawn floor)')
        plt.legend()
        plt.xlim(-75000, 75000)

        # Configure plot 2 (Z-Y)
        plt.figure(2)
        plt.axhline(y=params['alpha'], color='r', linestyle='--', label='Alpha (atmosphere top)')
        plt.axhline(y=params['y_min'], color='g', linestyle='--', label='Y min (spawn floor)')
        plt.axvline(x=params['beta'], color='b', linestyle='--', label='Beta (z boundary)')
        plt.axvline(x=-params['beta'], color='b', linestyle='--')
        plt.legend()

        plt.show()

    # Save results if requested
    if params['save_results'] and not df.empty:
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'particle_data_{timestamp}.xlsx'

        try:
            # Save to Excel
            df.to_excel(filename, sheet_name='Particles', index=False)
            print(f"\nResults saved to: {filename}")

            # Calculate summary statistics
            escaped_count = df[df['Result'] == 'escaped'].shape[0]
            recaptured_count = df[df['Result'] == 'recaptured'].shape[0]
            resimulate_count = df[df['Result'] == 'resimulate'].shape[0]

            print(f"\nSummary Statistics:")
            print(f"Total particles: {len(df)}")
            print(f"Escaped: {escaped_count} ({escaped_count / len(df) * 100:.1f}%)")
            print(f"Recaptured: {recaptured_count} ({recaptured_count / len(df) * 100:.1f}%)")
            print(f"Need resimulation: {resimulate_count} ({resimulate_count / len(df) * 100:.1f}%)")

            # Calculate leak rate if requested
            if params['find_leak_rate']:
                from LeakRate import find_lifetime, LRVarInput  # Import here to avoid circular import

                # Initialize leak rate module with parameters from params
                LRVarInput(
                    params['P_0'],
                    params['K_b'],
                    params['temperature'],
                    params['molecular_mass'],
                    params['gravity'],
                    params['n_0'],
                    params['molecular_diameter']
                )

                print("\n" + "=" * 60)
                print("ATMOSPHERIC LEAK RATE ANALYSIS")
                print("=" * 60)
                try:
                    lifetime_result = find_lifetime(filename)
                    print(f"\n{lifetime_result}")
                except Exception as e:
                    print(f"Error calculating leak rate: {e}")
                print("=" * 60)

        except Exception as e:
            print(f"Error saving file: {e}")
            # Fallback to current directory
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                backup_filename = os.path.join(current_dir, filename)
                df.to_excel(backup_filename, sheet_name='Particles', index=False)
                print(f"Saved to current directory instead: {backup_filename}")
            except Exception as e2:
                print(f"Failed to save file: {e2}")
                # If all else fails, print the data to console
                print("\nData preview:")
                print(df.head())

    return df


# Main execution
if __name__ == "__main__":
    # Example of running with custom parameters
    custom_params = {
        'gravity': 9.81,           # Standard gravity [m/s²]
        't_max': 100,              # Total simulation time [s]
        'dt': 0.1,                 # Time step [s]
        'num_particles': 100,      # Number of particles to simulate
        'show_plots': True,        # Show trajectory plots
        'find_leak_rate': True,    # Calculate atmospheric lifetime
    }
    
    # Get complete parameter set with defaults
    params = get_simulation_params(custom_params)
    
    # Initialize all modules with parameters
    G = 6.6743e-11  # Universal gravitational constant [m^3/kg/s^2]
    SSCPSVarInput(G)
    
    # Atmospheric parameters
    P_0 = 101325  # Atmospheric pressure at sea level [Pa]
    K_b = 1.380649e-23  # Boltzmann constant [J/K]
    m = 2.6566962e-26 * 2  # Mass of diatomic molecule [kg]
    n_0 = 2.687e25  # Molecular density [1/m³]
    d = 3.59e-10  # Molecular diameter [m]
    
    # Initialize other modules
    SIVarInput(params['temperature'], params['y_min'], params['y_max'], 
              params['z_length'], params['y_floor'])
    TCVarInput(params['z_length'], params['beta'], params['y_floor'], 
              params['alpha'], params['y_min'], params['y_max'])
    LRVarInput(P_0, K_b, params['temperature'], m, params['gravity'], n_0, d)
    
    # Run simulation with all parameters
    results = main(sim_params=params)
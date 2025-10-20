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
from SolverSharedCodePlusSolar_Optimized import compute_motion, SSCPSVarInput
from StochasticInput import stochastic_initial_conditions, SIVarInput
from TrajectoryClassification_numpy import classify_trajectory, TCVarInput
from LeakRate import LRVarInput



def main(radius, gravity, t_max, dt, is_rotating=False, num_particles=100,
         save_results=True, show_plots=False, find_leak_rate=True, comp_list = None):
    """
    Main simulation function for particle trajectory analysis.

    results = main(radius, gravity, t_max, dt, is_rotating, num_particles,
                   save_results, show_plots, find_leak_rate)

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
    comp_list       Composition of the atmosphere [list of tuples] (Name, mass, charge, number density) [String, kg, C, n/m^3]

    Outputs:
    results  DataFrame containing all particle simulation results
    """
    all_data = []

    print(f"Processing {num_particles} particles...")

    # Create plots if requested
    if show_plots:
        plt.figure(1, figsize=(10, 8))
        plt.title("X-Y Trajectories")
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")

        plt.figure(2, figsize=(10, 8))
        plt.title("Z-Y Trajectories")
        plt.xlabel("Z Position [m]")
        plt.ylabel("Y Position [m]")

    # Process each particle
    for i in range(num_particles):
        # Progress indicator - every 5%
        fiveP = num_particles / 20
        if (i % fiveP == 0):
            percent = 100 * i / num_particles
            print(str(percent) + "% percent done")

        # Generate initial conditions
        initial_state = stochastic_initial_conditions(T, y_min, y_max, z_length, comp_list)
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

            # Classify trajectory
            beta_crossings, result = classify_trajectory(alpha, beta, y_floor, solution_df)

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
    if show_plots:
        plt.figure(1)
        plt.axhline(y=alpha, color='r', linestyle='--', label='Alpha (atmosphere top)')
        plt.axhline(y=y_min, color='g', linestyle='--', label='Y max (spawn ceiling)')
        plt.legend()
        plt.xlim(-75000, 75000)

        plt.figure(2)
        plt.axhline(y=alpha, color='r', linestyle='--', label='Alpha (atmosphere top)')
        plt.axhline(y=y_min, color='g', linestyle='--', label='Y max (spawn ceiling)')
        plt.axvline(x=beta, color='b', linestyle='--', label='Beta (z boundary)')
        plt.axvline(x=-beta, color='b', linestyle='--')
        plt.legend()

        plt.show()

    # Save results if requested
    if save_results and not df.empty:
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
            if find_leak_rate:
                from LeakRate import find_lifetime  # Import here to avoid circular import

                print("\n" + "=" * 60)
                print("ATMOSPHERIC LEAK RATE ANALYSIS")
                print("=" * 60)
                lifetime_result = find_lifetime(filename)
                print(f"\n{lifetime_result}")
                print("=" * 60)

        except Exception as e:
            print(f"Error saving file: {e}")
            # Fallback to desktop
            desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
            backup_filename = os.path.join(desktop, filename)
            df.to_excel(backup_filename, sheet_name='Particles', index=False)
            print(f"Saved to desktop instead: {backup_filename}")

    return df


# Main execution
if __name__ == "__main__":
    # Simulation parameters
    t_max = 5000 # Total simulation time [s]
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
    G = 6.6743e-11               # Univeral gravitational constant [m^3/kg/s^2]

    # Atmospheric parameters
    T = 289  # Temperature [K]
    P_0 = 101325  # Atmospheric pressure at sea level [Pa]
    K_b = 1.380649e-23  # Boltzmann constant [J/K]
    m = 2.6566962e-26 * 2  # Mass of diatomic molecule [kg]
    n_0 = 2.687e25  # Molecular density [1/m³]
    d = 3.59e-10  # Molecular diameter [m]

    # Geometric parameters
    z_length = 200 * 1000  # Ringworld width [m]
    y_floor = 1000 * 1000  # Floor value (1 AU) [m]
    beta = z_length / 2  # Lateral boundary [m]
    alpha = y_floor - (218 * 1000)  # Atmosphere boundary [m]
    y_min = alpha - 10000  # Min spawn height [m]
    y_max = alpha  # Max spawn height [m]

    #atmospheric composition
    diatomic_oxygen = ("O2", 2.6566962e-26 * 2, 0,100)  #diatomic oxygen
    comp_list = [diatomic_oxygen]                       #collection of all species at desired altitude

    # Initialize all modules with parameters
    SSCPSVarInput(G)
    SIVarInput(T, y_min, y_max, z_length, y_floor)
    TCVarInput(z_length, beta, y_floor, alpha, y_min, y_max)
    LRVarInput(P_0, K_b, T, m, g, n_0, d)

    # Run simulation
    results = main(
        radius=y_min,                   # Use y_min as radius (1 AU)
        gravity=g,                      # Selected gravity value
        t_max=t_max,
        dt=dt,
        is_rotating=False,              # Solar gravity disabled
        num_particles=num_particles,
        find_leak_rate=True,            # Calculate atmospheric lifetime
        comp_list=comp_list
    )
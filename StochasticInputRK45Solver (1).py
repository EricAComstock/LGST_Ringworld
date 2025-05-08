"""
StochasticInputRK45Solver.py

Runs a particle trajectory simulation using randomly generated initial
conditions, propagates them using RK45 integration, and classifies outcomes.

Version: 1.0
Author: Edwin Ontivoros
Date: April 29, 2025
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import from other modules
from SolverSharedCodePlusSolar import compute_motion, G
from StochasticInput import stochastic_initial_conditions, T, m
from TrajectoryClassification import classify_trajectory

# Import boundary conditions from TrajectoryClassification
from TrajectoryClassification import y_min, y_max, z_length, beta, alpha, y_floor


def main(radius, gravity, t_max, dt, is_rotating=False, num_particles=100, save_results=True, show_plots=True):
    """
    Main function to run the particle simulation.

    Parameters:
    radius (float): Radius for calculating omega (m)
    gravity (float): Gravity for calculating omega (m/s²)
    t_max (float): Maximum simulation time (s)
    dt (float): Time step (s)
    is_rotating (bool): Whether the reference frame is rotating
    num_particles (int): Number of particles to simulate
    save_results (bool): Whether to save results to Excel
    show_plots (bool): Whether to display trajectory plots
    
    Returns:
    pd.DataFrame: DataFrame containing all particle simulation results
    """
    all_data = []

    print(f"Processing {num_particles} particles...")

    # Create plots if needed
    if show_plots:
        plt.figure(1, figsize=(10, 8))  # X-Y plot
        plt.title("X-Y Trajectories")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")

        plt.figure(2, figsize=(10, 8))  # Z-Y plot
        plt.title("Z-Y Trajectories")
        plt.xlabel("Z Position (m)")
        plt.ylabel("Y Position (m)")

    # Process each particle
    for i in range(num_particles):
        # Get initial conditions for this particle
        initial_state    = stochastic_initial_conditions(m, T, y_min, y_max, z_length)
        initial_position = initial_state[:3]
        initial_velocity = initial_state[3:]

        # Compute trajectory
        try:
            # Call compute_motion to get final position, velocity and full trajectory
            final_position, final_velocity, solution = compute_motion(
                initial_position, initial_velocity, radius, gravity, t_max, dt, None
            )

            # Extract trajectory data from solution
            trajectory  = solution.y[:3, :].T  # Transpose for (n_timesteps, 3) shape

            # Convert trajectory to DataFrame for classification
            solution_df = pd.DataFrame(trajectory, columns=[0, 1, 2])

            # Classify trajectory using imported boundary values
            beta_crossings, result = classify_trajectory(alpha, beta, y_floor, solution_df)

            # Create data row for this particle
            particle_data = {
                'Particle #': i + 1,
                'Initial x' : initial_position[0],
                'Initial y' : initial_position[1],
                'Initial z' : initial_position[2],
                'Initial vx': initial_velocity[0],
                'Initial vy': initial_velocity[1],
                'Initial vz': initial_velocity[2],
                'Final x'   : final_position[0],
                'Final y'   : final_position[1],
                'Final z'   : final_position[2],
                'Final vx'  : final_velocity[0],
                'Final vy'  : final_velocity[1],
                'Final vz'  : final_velocity[2],
                'Beta crossings': beta_crossings,
                'Result'        : result
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

    # Create DataFrame with all results
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

        # Show plots if requested
        if show_plots:
            plt.show()

    # Save results if requested
    if save_results and not df.empty:
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f'particle_data_{timestamp}.xlsx'
        
        try:
            # Save to Excel
            df.to_excel(filename, sheet_name='Particles', index=False)
            print(f"\nResults saved to: {filename}")

            # Additional summary statistics
            escaped_count    = df[df['Result'] == 'escaped'].shape[0]
            recaptured_count = df[df['Result'] == 'recaptured'].shape[0]
            resimulate_count = df[df['Result'] == 'resimulate'].shape[0]

            print(f"\nSummary Statistics:")
            print(f"Total particles: {len(df)}")
            print(f"Escaped: {escaped_count} ({escaped_count / len(df) * 100:.1f}%)")
            print(f"Recaptured: {recaptured_count} ({recaptured_count / len(df) * 100:.1f}%)")
            print(f"Need resimulation: {resimulate_count} ({resimulate_count / len(df) * 100:.1f}%)")

        except Exception as e:
            print(f"Error saving file: {e}")
            # Try saving to desktop as fallback
            desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
            backup_filename = os.path.join(desktop, filename)
            df.to_excel(backup_filename, sheet_name='Particles', index=False)
            print(f"Saved to desktop instead: {backup_filename}")

    return df

if __name__ == "__main__":
    # Simulation parameters
    t_max         = 100  # Total simulation time (s)
    dt            = 0.1  # Time step (s)
    num_particles = 100  # Number of particles to simulate

    # Run simulation
    results = main(
        radius    = y_min,  # Use y_min as radius
        gravity   = G,     # Use Earth gravity
        t_max     = t_max,
        dt        = dt,
        is_rotating=False,  # Solar gravity is disabled
        num_particles=num_particles
    )

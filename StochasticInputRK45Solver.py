from SolverSharedCode import compute_motion, G
from StochasticInput import stochastic_initial_conditions, T, m, y_min, y_max, z_length
from TrajectoryClassification import classify_trajectory
import pandas as pd
import os
from datetime import datetime


def main(radius, gravity, t_max, dt, is_rotating, num_particles=100):  # Added num_particles parameter
    # Create empty lists to store multiple particles
    all_data = []

    print(f"Processing {num_particles} particles...")

    for i in range(num_particles):
        # Get initial conditions for each particle
        initial_state = stochastic_initial_conditions(m, T, y_min, y_max, z_length)
        initial_position, initial_velocity = initial_state[:3], initial_state[3:]

        # Compute final state
        final_position, final_velocity, solution = compute_motion(
            initial_position, initial_velocity, radius, gravity, t_max, dt, is_rotating
        )

        #Determine result of simulation
        beta = z_length / 2
        alpha = y_min
        solution_df = pd.DataFrame(solution.T)
        [beta_crossings,result] = classify_trajectory(alpha,beta,solution_df)

        # Create data row for this particle
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

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'particle_data_{timestamp}.xlsx'

    try:
        # Save to Excel
        df.to_excel(filename, sheet_name='Particles', index=False)
        print(f"\nResults saved to: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")
        # Try saving to desktop as fallback
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        backup_filename = os.path.join(desktop, filename)
        df.to_excel(backup_filename, sheet_name='Particles', index=False)
        print(f"Saved to desktop instead: {backup_filename}")


if __name__ == "__main__":
    t_max = 10  # Total simulation time
    dt = 0.1  # Time step
    num_particles = 100  # Set number of particles
    main(y_min, G, t_max, dt, "yes", num_particles)
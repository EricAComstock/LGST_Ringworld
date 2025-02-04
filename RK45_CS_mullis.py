import numpy as np
from scipy.integrate import solve_ivp

# Constants
AU = 1.496e11  # 1 astronomical unit in meters
G = 9.81       # Gravity on Earth in m/s²

# Core Functions
def calculate_omega(radius, gravity):
    """
    Calculate the angular velocity of the rotating frame.
    ω = sqrt(gravity / radius)
    """
    return np.sqrt(gravity / radius)

def equations_of_motion(t, state, omega):
    """
    Compute derivatives of position and velocity in the rotating frame.
    state = [x, y, z, vx, vy, vz]
    omega = angular velocity vector
    """
    r = np.array(state[:3])  # Position vector
    v = np.array(state[3:])  # Velocity vector

    # Calculate pseudoforces in the rotating frame
    coriolis_force = -2 * np.cross(omega, v)
    centrifugal_force = -np.cross(omega, np.cross(omega, r))
    acceleration = coriolis_force + centrifugal_force

    # Return derivatives of position and velocity
    return [*v, *acceleration]

def solve_dynamics(initial_conditions, t_max, dt, omega):
    """
    Solve the equations of motion using RK45.
    Returns the time vector and the solution matrix.
    """
    t_span = (0, t_max)  # Time span for integration
    t_eval = np.arange(0, t_max, dt)  # Time points for evaluation

    # Solve the equations of motion
    solution = solve_ivp(
        equations_of_motion,
        t_span,
        initial_conditions,
        t_eval=t_eval,
        args=(omega,),  # Pass angular velocity as an argument
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    return solution.t, solution.y

# Main Function
def main(radius, gravity, t_max, dt, is_rotating, x, y, z, vx, vy, vz):
    
    # Combine position and velocity into the initial conditions array
    initial_conditions = np.array([x, y, z, vx, vy, vz])

    # Angular velocity based on user input
    if is_rotating == "yes":
        omega = np.array([0, 0, calculate_omega(radius, gravity)])
        print("\nFrame is rotating.")
    else:
        omega = np.array([0, 0, 0])  # No rotation
        print("\nFrame is not rotating.")

    # Solve dynamics
    print("\nSolving dynamics...")
    t, solution = solve_dynamics(initial_conditions, t_max, dt, omega)

    # Output the final position and velocity
    final_position = solution[:3, -1]  # Final [x, y, z]
    final_velocity = solution[3:, -1]  # Final [vx, vy, vz]

    print("\nFinal Results:")
    print(f"  Final Position: {final_position}")
    print(f"  Final Velocity: {final_velocity}")
    print("---")

    return final_position, final_velocity

    # Optional: Output all results over time (uncomment to view)
    # print("\nResults Over Time:")
    # for i in range(len(t)):
    #     print(f"Time: {t[i]:.2f} s")
    #     print(f"  Position: {solution[:3, i]}")
    #     print(f"  Velocity: {solution[3:, i]}")
    #     print("---")



# Run the main function
if __name__ == "__main__":
    main()

import numpy as np
from SolverSharedCode import compute_motion
from math import sqrt

def inertial_translation(pi, vi, radius, gravity, t_max, dt, is_rotating="yes"):
    #Computes the final position using correct equations of motion with Coriolis & centrifugal forces.
    from scipy.integrate import solve_ivp

    if is_rotating == "yes":
        omega = np.array([0, 0, sqrt(gravity / radius)])  # Only along z-axis
    else:
        omega = np.array([0, 0, 0])  # No rotation

    def equations_of_motion(t, state):
        r = np.array(state[:3])  # Position [x, y, z]
        v = np.array(state[3:])  # Velocity [vx, vy, vz]

        # Compute rotational effects
        omega_cross_v = np.cross(omega, v)  # Coriolis term
        omega_cross_r = np.cross(omega, r)  # Centrifugal term
        coriolis_force = -2 * omega_cross_v
        centrifugal_force = -np.cross(omega, omega_cross_r)

        acceleration = coriolis_force + centrifugal_force
        return [*v, *acceleration]  # Returns position derivatives and acceleration

    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    initial_conditions = pi + vi  # Combine position and velocity

    solution = solve_ivp(equations_of_motion, t_span, initial_conditions, t_eval=t_eval, method="RK45")

    final_position = solution.y[:3, -1].tolist()
    return final_position

def test_translation():
    #Tests the final position from compute_motion against inertial_translation.

    # Define test parameters
    #THIS IS WHERE TO INPUT TEST VALUES
    pi = [1, 1, 1]  # Initial position (x, y, z) in meters
    vi = [1, 1, 1]  # Initial velocity (vx, vy, vz) in meters/second
    r = 6371000  # Earth's radius in meters
    g = 9.81  # Gravitational acceleration in m/s²
    t = 100  # Total simulation time in seconds
    dt = 0.1  # Time step
    is_rotating = "yes"

    # Compute final position using SolverSharedCode
    final_position_solver, final_velocity_solver = compute_motion(pi, vi, r, g, t, dt, is_rotating)

    # Compute final position using corrected inertial translation
    final_position_inertial = inertial_translation(pi, vi, r, g, t, dt, is_rotating)

    # Compare results
    final_position_solver = np.array(final_position_solver)
    final_position_inertial = np.array(final_position_inertial)

    # Check if positions match within tolerance
    tolerance = 1e-4  # Acceptable difference in meters
    error = np.linalg.norm(final_position_solver - final_position_inertial)

    print("\nFinal Position (compute_motion):", [f"{coord:.4f}" for coord in final_position_solver], "m")
    print("Final Position (inertial_translation):", [f"{coord:.4f}" for coord in final_position_inertial], "m")

    if error < tolerance:
        print("\n✅ TEST PASSED: Final positions match within tolerance.")
    else:
        print("\n❌ TEST FAILED: Difference in final positions is", error, "m")

# Run the test
test_translation()

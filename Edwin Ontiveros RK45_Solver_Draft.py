import numpy as np
from scipy.integrate import solve_ivp

# Input Initial Conditions and Parameters (Later to be Replaced with Stochastic Initial Conditions)
def get_inputs():
    print("=== Collisionless Particle Simulation ===")

    # Simulation Parameters
    radius = float(input("Enter ringworld radius (m, default 1 AU = 1.496e11): ") or 1.496e11)
    gravity = float(input("Enter artificial gravity (m/s^2, default 9.8): ") or 9.8)
    t_max = float(input("Enter maximum simulation time (s, default 1000): ") or 1000)
    dt = float(input("Enter time step (s, default 0.1): ") or 0.1)

    # Initial Conditions
    print("\nEnter initial conditions for the particle:")
    r = float(input(f"  Initial radial position (m, default {radius}): ") or radius)
    theta_degrees = float(input("  Initial azimuthal angle (degrees, default 0): ") or 0)  # Angle in degrees
    theta = np.radians(theta_degrees)  # Convert to radians
    z = float(input("  Initial vertical position (m, default 0): ") or 0)
    vr = float(input("  Initial radial velocity (m/s, default 0): ") or 0)
    vtheta = float(input("  Initial azimuthal velocity (m/s, default 0): ") or 0)
    vz = float(input("  Initial vertical velocity (m/s, default 0): ") or 0)

    # Combine inputs
    initial_conditions = [r, theta, z, vr, vtheta, vz]
    return radius, gravity, t_max, dt, initial_conditions


# Cylindrical Coord Motion Equations for Particle in Rotating RF
def equations_of_motion(t, y, radius, gravity, omega):
    """
    Compute the derivatives of position and velocity in cylindrical coordinates.
    """
    r, theta, z, vr, vtheta, vz = y
    r_effective = radius - z  # Effective radius decreases with inward height
    return [
        vr,  # dr/dt
        vtheta / r_effective,  # dtheta/dt
        vz,  # dz/dt
        r_effective * omega ** 2 - 2 * omega * vtheta,  # dvr/dt
        2 * omega * vr,  # dvtheta/dt
        -gravity  # dvz/dt
    ]


# Solver
def solver(initial_conditions, t_max, dt, radius, gravity):
    """
    Solve the equations of motion for a particle in a rotating reference frame.
    """
    omega = np.sqrt(gravity / radius)  # Angular velocity
    sol = solve_ivp(
        equations_of_motion, (0, t_max), initial_conditions,
        t_eval=np.arange(0, t_max, dt), args=(radius, gravity, omega), method="RK45"
    )
    return sol.t, sol.y

# Main Execution
if __name__ == "__main__":
    # Get all inputs (parameters and initial conditions)
    radius, gravity, t_max, dt, initial_conditions = get_inputs()
    # Run the solver
    t, y = solver(initial_conditions, t_max, dt, radius, gravity)
    # Display results
    print(f"Final position: (r={y[0, -1]:.2e}, theta={y[1, -1]:.2e}, z={y[2, -1]:.2e})")
    print(f"Final velocity: (vr={y[3, -1]:.2e}, vtheta={y[4, -1]:.2e}, vz={y[5, -1]:.2e})")
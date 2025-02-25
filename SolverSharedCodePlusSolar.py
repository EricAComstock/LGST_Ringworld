import numpy as np
from scipy.integrate import solve_ivp

# Constants
G = 9.81  # Gravity on Earth in m/s²

def calculate_omega(radius, gravity):
    return np.sqrt(gravity / radius)

def calculate_solar_gravity(r, solar_mu):
    """
    Compute the gravitational acceleration due to a central solar body.
    """
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.zeros(3)
    return -solar_mu * r / (r_mag ** 3)

def equations_of_motion(t, state, omega, solar_mu=None):
    """
    Equations of motion in the rotating frame.
    Now modified to print position at each time step.
    """
    r, v = state[:3], state[3:]

    omega_vector = np.array([0, 0, omega])

    # Coriolis force: -2 * (Ω × v)
    coriolis_force = -2 * np.cross(omega_vector, v)

    # Centrifugal force: -Ω × (Ω × r)
    centrifugal_force = -np.cross(omega_vector, np.cross(omega_vector, r))

    # Sum accelerations
    total_force = coriolis_force + centrifugal_force
    if solar_mu is not None:
        solar_acceleration = calculate_solar_gravity(r, solar_mu)
        total_force += solar_acceleration

    # Debugging output for position at each time step
    print(f"t={t:.6f}, Position: {r}")

    return np.concatenate((v, total_force))

def solve_dynamics(initial_conditions, t_max, dt, omega, solar_mu=None):
    """
    Solve equations of motion using RK45 (Runge-Kutta) with high precision.
    """
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max + dt / 2, dt)

    solution = solve_ivp(
        equations_of_motion,
        t_span,
        initial_conditions,
        t_eval=t_eval,
        args=(omega, solar_mu),
        method='RK45',
        rtol=1e-12,  # Increased precision
        atol=1e-12
    )

    return solution.t, solution.y

def compute_motion(initial_position, initial_velocity, radius, gravity, t_max, dt, solar_mu=None):
    """
    Computes the motion strictly in the rotating frame.
    """
    initial_position = np.array(initial_position, dtype=float)
    initial_velocity = np.array(initial_velocity, dtype=float)

    omega = calculate_omega(radius, gravity)
    initial_state = np.concatenate([initial_position, initial_velocity])

    t, solution = solve_dynamics(initial_state, t_max, dt, omega, solar_mu)

    final_position = solution[:3, -1].tolist()
    final_velocity = solution[3:, -1].tolist()

    return final_position, final_velocity

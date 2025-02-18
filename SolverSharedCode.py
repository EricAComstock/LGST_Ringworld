import numpy as np
from scipy.integrate import solve_ivp

# Constants
G = 9.81  # Gravity on Earth in m/s²

def calculate_omega(radius, gravity):
    return np.sqrt(gravity / radius)

def equations_of_motion(t, state, omega):
    r, v = state[:3], state[3:]
    omega_cross_v = np.cross(omega, v)
    omega_cross_r = np.cross(omega, r)
    coriolis_force = -2 * omega_cross_v
    centrifugal_force = -np.cross(omega, omega_cross_r)
    acceleration = coriolis_force + centrifugal_force
    return [*v, *acceleration]

def solve_dynamics(initial_conditions, t_max, dt, omega):
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    solution = solve_ivp(
        equations_of_motion,
        t_span,
        initial_conditions,
        t_eval=t_eval,
        args=(omega,),
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    return solution.t, solution.y

def compute_motion(initial_position, initial_velocity, radius, gravity, t_max, dt, is_rotating="yes"):
    omega = np.array([0, 0, calculate_omega(radius, gravity)]) if is_rotating == "yes" else np.zeros(3)
    t, solution = solve_dynamics(initial_position + initial_velocity, t_max, dt, omega)
    final_position = solution[:3, -1].tolist()
    final_velocity = solution[3:, -1].tolist()
    return final_position, final_velocity
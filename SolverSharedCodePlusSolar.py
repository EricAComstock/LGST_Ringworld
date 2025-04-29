"""
SolverSharedCodePlusSolar.py

Provides physics utility functions for rotating reference frames, including
angular velocity computation, solar gravity, and RK45-based motion integration.

Version: 1.0
Author: Edwin Ontivoros
Date: April 29, 2025
"""


import numpy as np
from scipy.integrate import solve_ivp

# Constants
G = 9.81  # Gravity on Earth in m/s²

def calculate_omega(radius, gravity):
    """
    Calculate angular velocity for a rotating reference frame.

    Parameters:
    radius (float): Radius of rotation (m)
    gravity (float): Desired artificial gravity (m/s²)

    Returns:
    float: Angular velocity (rad/s)
    """
    return -np.sqrt(gravity / radius)  # Negative sign for clockwise rotation

def calculate_solar_gravity(r, solar_mu):
    """
    Compute gravitational acceleration due to a central body.

    Parameters:
    r (np.array): Position vector [x, y, z]
    solar_mu (float): Gravitational parameter (G*M)

    Returns:
    np.array: Gravitational acceleration vector [ax, ay, az]
    """
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.zeros(3)  # Avoid division by zero
    return -solar_mu * r / (r_mag ** 3)

def equations_of_motion_rotating(t, state, omega, solar_mu=None):
    """
    Equations of motion for a particle in a rotating frame.

    Parameters:
    t (float): Time (s)
    state (list or np.array): State vector [x, y, z, vx, vy, vz]
    omega (float): Angular velocity (rad/s)
    solar_mu (float, optional): Solar gravity parameter

    Returns:
    np.array: Derivatives of state vector [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    r, v            = state[:3], state[3:]

    # Position derivatives are simply the velocities
    dr_dt           = v

    omega_vector    = np.array([0, 0, omega])

    # Coriolis acceleration: -2(ω × v)
    coriolis_acc    = -2 * np.cross(omega_vector, v)

    # Centrifugal acceleration: -ω × (ω × r)
    centrifugal_acc = -np.cross(omega_vector, np.cross(omega_vector, r))

    # Combine all accelerations
    dv_dt = coriolis_acc + centrifugal_acc

    # Add solar gravity if enabled
    if solar_mu is not None:
        dv_dt += calculate_solar_gravity(r, solar_mu)

    # Return derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    return np.concatenate((dr_dt, dv_dt))

def compute_motion(initial_position, initial_velocity, radius, gravity, t_max, dt, solar_mu=None):
    """
    Computes particle motion in the rotating frame using RK45 integration.

    Parameters:
    initial_position (list): Initial position vector [x, y, z]
    initial_velocity (list): Initial velocity vector [vx, vy, vz]
    radius (float): Radius for calculating omega (m)
    gravity (float): Gravity for calculating omega (m/s²)
    t_max (float): Maximum simulation time (s)
    dt (float): Time step (s)
    solar_mu (float, optional): Solar gravity parameter

    Returns:
    tuple: (final_position, final_velocity, solution)
        - final_position (list): Final position vector [x, y, z]
        - final_velocity (list): Final velocity vector [vx, vy, vz]
        - solution (OdeResult): Full integration result
    """
    initial_position = np.array(initial_position, dtype=float)
    initial_velocity = np.array(initial_velocity, dtype=float)

    # Calculate angular velocity from radius and gravity
    omega            = calculate_omega(radius, gravity)

    # Create initial state vector [x, y, z, vx, vy, vz]
    initial_state    = np.concatenate((initial_position, initial_velocity))

    t_span  = (0, t_max)
    t_eval  = np.arange(0, t_max + dt / 2, dt)

    # Solve the equations of motion using RK45
    solution = solve_ivp(
        equations_of_motion_rotating,
        t_span,
        initial_state,
        t_eval=t_eval,
        args=(omega, solar_mu),
        method='RK45',
        rtol=1e-12,
        atol=1e-12
    )

    # Extract the final state
    final_position = solution.y[:3, -1]
    final_velocity = solution.y[3:, -1]

    # Return the final position, final velocity, and the full solution
    return final_position.tolist(), final_velocity.tolist(), solution

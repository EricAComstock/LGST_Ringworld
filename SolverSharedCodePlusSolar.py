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
g = 9.81  # Gravity on Earth in m/s²

def SSCPSVarInput(g_i):
    global g
    g = g_i

def calculate_omega(radius, gravity):
    """
    Calculate angular velocity for a rotating reference frame where
    centripetal acceleration equals desired gravity.

    For a ringworld, ω² = g/r

    The sign is negative, indicating clockwise rotation
    when viewed from the positive z-axis.

    Parameters:
    radius: Radius of rotation (m)
    gravity: Desired artificial gravity (m/s²)

    Returns:
    omega: Angular velocity (rad/s)
    """
    return -np.sqrt(gravity / radius)  # Negative sign for clockwise rotation


def calculate_solar_gravity(r, solar_mu):
    """
    Compute the gravitational acceleration due to a central body.

    F = -G·M·m/r² in direction of r
    a = -G·M/r² · r̂ = -μ · r̂/r²

    Parameters:
    r: Position vector [x, y, z]
    solar_mu: Gravitational parameter μ = G·M

    Returns:
    Gravitational acceleration vector [ax, ay, az]

    """
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.zeros(3)  # Avoid division by zero
    return -solar_mu * r / (r_mag ** 3)

def equations_of_motion_rotating(t, state, omega, solar_mu=None):
    """
    Equations of motion for a particle in a rotating frame.


    In a rotating frame, a free particle experiences Coriolis and centrifugal forces
    even when there are no external forces in the inertial frame.

    For negative omega (clockwise rotation):
    - Coriolis acceleration: -2(ω × v)
    - Centrifugal acceleration: -ω × (ω × r)

    Parameters:
    t: Time (s)
    state: State vector [x, y, z, vx, vy, vz]
    omega: Angular velocity magnitude (rad/s)
    solar_mu: Solar gravity parameter (None to disable)

    Returns:
    Derivatives of state vector [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    r, v = state[:3], state[3:]

    # Position derivatives are simply the velocities
    dr_dt = v

    # For a particle in a rotating frame, we need to include the Coriolis and centrifugal forces
    omega_vector = np.array([0, 0, omega])  # omega is now negative

    # Coriolis acceleration: -2(ω × v)
    coriolis_acc = -2 * np.cross(omega_vector, v)

    # Centrifugal acceleration: -ω × (ω × r)
    centrifugal_acc = -np.cross(omega_vector, np.cross(omega_vector, r))

    # Combine all accelerations
    dv_dt = coriolis_acc + centrifugal_acc

    # Add solar gravity if enabled
    if solar_mu is not None:
        solar_acceleration = calculate_solar_gravity(r, solar_mu)
        dv_dt += solar_acceleration

    # Return derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    return np.concatenate((dr_dt, dv_dt))

def compute_motion(initial_position, initial_velocity, radius, gravity, t_max, dt, solar_mu=None):
    """
    Computes particle motion in the rotating frame using RK45 and returns the final state.

    This function stays entirely within the rotating frame and numerically solves
    the equations of motion for a particle using the RK45 method.

    Parameters:
    initial_position: Initial position vector in rotating frame
    initial_velocity: Initial velocity vector in rotating frame
    radius: Radius for calculating omega
    gravity: Gravity for calculating omega
    t_max: Maximum simulation time (s)
    dt: Time step (s)
    solar_mu: Solar gravity parameter (None to disable)

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


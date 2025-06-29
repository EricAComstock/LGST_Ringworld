"""
SolverSharedCodePlusSolar.py
Provides physics utility functions for rotating reference frames, including
angular velocity computation, solar gravity, and RK45-based motion integration.

V1.1, James Stewart, June 2, 2025
V1.0, Edwin Ontiveros, April 29, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants
G_UNIVERSAL = 6.6743e-11  # Universal Gravitational Constant [m³/kg·s²]
G_UNIVERSAL = G_UNIVERSAL / 1e9  # Scaled version for numerical stability

# Ringworld parameters
g_ringworld = 9.81  # Default ringworld gravitational acceleration [m/s²]


def SSCPSVarInput(g_i):
    """
    Set ringworld's gravitational acceleration from main module.

    SSCPSVarInput(g_i)

    Inputs:
    g_i  Ringworld's gravitational acceleration [m/s²]

    Outputs:
    None (sets global variable)

    Note: g_i is the artificial gravity, NOT the universal constant
    """
    global g_ringworld
    g_ringworld = g_i


def calculate_omega(radius, gravity):
    """
    Calculate angular velocity for rotating reference frame.
    For a ringworld, ω² = g/r with negative sign for clockwise rotation.

    omega = calculate_omega(radius, gravity)

    Inputs:
    radius   Radius of rotation [m]
    gravity  Desired artificial gravity [m/s²]

    Outputs:
    omega  Angular velocity [rad/s]
    """
    return -np.sqrt(gravity / radius)  # Negative for clockwise rotation


def calculate_solar_gravity(r, solar_mu):
    """
    Compute gravitational acceleration due to central body.
    Uses gravitational parameter μ = G·M for efficiency.

    a = calculate_solar_gravity(r, solar_mu)

    Inputs:
    r         Position vector [x, y, z] [m]
    solar_mu  Gravitational parameter μ = G·M [m³/s²]

    Outputs:
    a  Gravitational acceleration vector [ax, ay, az] [m/s²]
    """
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.zeros(3)  # Avoid division by zero
    return -solar_mu * r / (r_mag ** 3)


def equations_of_motion_rotating(t, state, omega, solar_mu=None):
    """
    Equations of motion for particle in rotating frame.
    Includes Coriolis and centrifugal pseudoforces.

    state_dot = equations_of_motion_rotating(t, state, omega, solar_mu)

    Inputs:
    t         Time [s]
    state     State vector [x, y, z, vx, vy, vz] [m, m/s]
    omega     Angular velocity magnitude [rad/s]
    solar_mu  Solar gravity parameter (None to disable) [m³/s²]

    Outputs:
    state_dot  Derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    r, v = state[:3], state[3:]  # Position and velocity

    # Position derivatives are velocities
    dr_dt = v

    # Angular velocity vector (rotation about z-axis)
    omega_vector = np.array([0, 0, omega])

    # Coriolis acceleration: -2(ω × v)
    coriolis_acc = -2 * np.cross(omega_vector, v)

    # Centrifugal acceleration: -ω × (ω × r)
    centrifugal_acc = -np.cross(omega_vector, np.cross(omega_vector, r))

    # Total acceleration
    dv_dt = coriolis_acc + centrifugal_acc

    # Add solar gravity if enabled
    if solar_mu is not None:
        solar_acceleration = calculate_solar_gravity(r, solar_mu)
        dv_dt += solar_acceleration

    return np.concatenate((dr_dt, dv_dt))


def inertial_to_rotating(i_position, i_velocity, omega, theta):
    """
    Transform position and velocity from inertial to rotating frame.

    r_position, r_velocity = inertial_to_rotating(i_position, i_velocity, omega, theta)

    Inputs:
    i_position  Position vector in inertial frame [x, y, z] [m]
    i_velocity  Velocity vector in inertial frame [vx, vy, vz] [m/s]
    omega       Angular velocity magnitude [rad/s]
    theta       Current rotation angle [rad]

    Outputs:
    r_position  Position in rotating frame [m]
    r_velocity  Velocity in rotating frame [m/s]
    """
    # Rotation matrix for clockwise rotation
    R_i2r = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Transform position
    r_position = R_i2r @ i_position

    # Transform velocity: v_rot = R·v_inertial - (ω × r_rot)
    omega_vector = np.array([0, 0, omega])
    r_velocity = R_i2r @ i_velocity - np.cross(omega_vector, r_position)

    return r_position, r_velocity


def compute_gravity(i_position, i_velocity, omega, theta, mass, rw_position):
    """
    Calculate gravitational acceleration on Ringworld from third-body objects.

    a_rw = compute_gravity(i_position, i_velocity, omega, theta, mass, rw_position)

    Inputs:
    i_position   List of position vectors in inertial frame [m]
    i_velocity   List of velocity vectors in inertial frame [m/s]
    omega        Angular velocity magnitude [rad/s]
    theta        Current rotation angle [rad]
    mass         List of third-body masses [kg]
    rw_position  Ringworld position in inertial frame [m]

    Outputs:
    a_rw  Acceleration of Ringworld [m/s²]
    """
    acceleration_ringworld = np.array([0., 0., 0.])  # Initialize

    # Loop through all third-bodies
    for i in range(len(i_position)):
        # Convert to rotating frame
        r_position = inertial_to_rotating(i_position[i], i_velocity[i], omega, theta)[0]

        # Gravitational acceleration from body i
        r_vec = r_position - rw_position  # Separation vector
        r_mag = np.linalg.norm(r_vec)  # Distance
        a_direct = G_UNIVERSAL * mass[i] / (r_mag ** 3) * r_vec  # Direct term
        a_indirect = G_UNIVERSAL * mass[i] / (np.linalg.norm(r_position) ** 3) * r_position

        acceleration_ringworld += (a_direct - a_indirect)

    return acceleration_ringworld


def save_fig(i_position, i_velocity, omega, mass, rw_position):
    """
    Plot gravitational acceleration vs rotation angle for visualization.

    save_fig(i_position, i_velocity, omega, mass, rw_position)

    Inputs:
    i_position   List of position vectors in inertial frame [m]
    i_velocity   List of velocity vectors in inertial frame [m/s]
    omega        Angular velocity magnitude [rad/s]
    mass         List of third-body masses [kg]
    rw_position  Ringworld position in inertial frame [m]

    Outputs:
    fig.png  Saved plot file
    """
    plt.figure()

    # Sample angles
    angles = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) * np.pi / 6

    # Calculate gravity at each angle
    for theta in angles:
        acceleration_ringworld = compute_gravity(i_position, i_velocity, omega, theta, mass, rw_position)
        gravity_norm = np.linalg.norm(acceleration_ringworld)

        plt.plot(theta, gravity_norm, 'o')  # Plot as points

    # Format plot
    plt.xlabel('Angle [radians]')
    plt.ylabel('Gravity Norm [m/s²]')
    plt.title('Gravitational Acceleration vs Rotation Angle')
    plt.grid(True)
    plt.savefig('fig.png')
    plt.close()  # Free memory


def compute_motion(initial_position, initial_velocity, radius, gravity, t_max, dt, solar_mu=None):
    """
    Compute particle motion in rotating frame using RK45 integration.

    final_pos, final_vel, solution = compute_motion(initial_position, initial_velocity,
                                                   radius, gravity, t_max, dt, solar_mu)

    Inputs:
    initial_position  Initial position vector in rotating frame [m]
    initial_velocity  Initial velocity vector in rotating frame [m/s]
    radius            Radius for calculating omega [m]
    gravity           Gravity for calculating omega [m/s²]
    t_max             Maximum simulation time [s]
    dt                Time step [s]
    solar_mu          Solar gravity parameter (None to disable) [m³/s²]

    Outputs:
    final_position  Final position vector [x, y, z] [m]
    final_velocity  Final velocity vector [vx, vy, vz] [m/s]
    solution        Full integration result (OdeResult object)
    """
    # Convert to numpy arrays
    initial_position = np.array(initial_position, dtype=float)
    initial_velocity = np.array(initial_velocity, dtype=float)

    # Calculate angular velocity
    omega = calculate_omega(radius, gravity)

    # Create initial state vector
    initial_state = np.concatenate((initial_position, initial_velocity))

    # Time parameters
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max + dt / 2, dt)

    # Solve equations of motion using RK45
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

    # Extract final state
    final_position = solution.y[:3, -1]
    final_velocity = solution.y[3:, -1]

    return final_position.tolist(), final_velocity.tolist(), solution


# Testing code - only runs when this file is executed directly
if __name__ == "__main__":
    # Test compute_gravity function
    test_result = compute_gravity([[2e8, 0., 0.]], [[0., 0., 0.]], 1e-6, 0., [1e13], [1e8, 1e8, 0.])
    print(f"Test gravity calculation result: {test_result}")
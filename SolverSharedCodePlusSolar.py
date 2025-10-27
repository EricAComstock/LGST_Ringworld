"""
SolverSharedCodePlusSolar.py

Provides physics utility functions for rotating reference frames, including
angular velocity computation, solar gravity, and RK45-based motion integration.

1.0, Edwin Ontivoros, April 29, 2025
1.1, James Stewart, June 2, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.6743e-11 # Universal Gravitational Constant
radius = 0     # Radius of RW

def SSCPSVarInput(G_i, r_0_i, B_0_i, gamma_i):
    """
    Set global parameters for solver code.
    Called by StochasticInputRK45Solver.py to pass simulation parameters.

    SSCPSVarInput(G_i)

    Inputs:
    G_i         Universal Gravitational Constant [Nm^2/kg^2]       
    r_0 _i        Distance From Earth to Sun (1AU) [m]
    B_0_i         Solar Magnetic Field at Earth's Location [T]
    v_r_i         Radial Solar Wind Speed [m/s] (Placeholder) 
    gamma_i       Angle From the Sun's Equator to the Ringworld plane [Degrees] (Placeholder)      
    Outputs:
    None (sets global variables)
    """
    global G, r_0, B_0, gamma
    G = G_i
    r_0 = r_0_i
    B_0 = B_0_i
    gamma = gamma_i

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

    F = -G·M·m/r² direction of r
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
    Derivatives of state vector [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dq/dt, dm/dt]
    """
    r, v = state[:3], state[3:6]

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

    # Return derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dq/dt, dm/dt]
    derivs = np.concatenate((dr_dt,dv_dt, [0,0]))
    return derivs

"""
Implement gravity from 3rd bodies (planets and other stars) as a function that outputs gravitational acceleration, and inputs a structure containing a list of third body positions
Make sure to convert the 3rd body positions from the inertial to the rotating reference frame (tester-Mullis-new.py has a lot of the code you will need)
"""

def inertial_to_rotating(i_position, i_velocity, omega, theta):
    """
    Transform position and velocity from inertial to rotating frame.

    Parameters:
    i_position (np.array): Position vector in inertial frame [x, y, z]
    i_velocity (np.array): Velocity vector in inertial frame [vx, vy, vz]
    omega (float): Angular velocity magnitude (rad/s)
    theta (float): Current rotation angle (rad)

    Returns:
    tuple: (r_position, r_velocity) in rotating frame
    """
    # For clockwise rotation (negative omega), the rotation matrix is:
    print("Theta: ", theta)
    theta = theta[0]
    omega = omega[0]
    print("Theta: ", theta)

    R_i2r = np.array([
        [np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Transform position
    r_position = R_i2r @ i_position

    # Transform velocity: v_rot = R·v_inertial - (ω × r_rot)
    omega_vector = np.array([0, 0, omega])
    r_velocity   = R_i2r @ i_velocity - np.cross(omega_vector, r_position)

    return r_position, r_velocity

def compute_gravity(i_position, i_velocity, omega, theta, mass, rw_position):
    """
    Calculates gravitational acceleration of the Ringworld under influence of third-body objects.

    Parameters: 
    i_position (list of np.array): Position vector in inertial frame [x, y, z] of third-body objects
    i_velocity (list of np.array): Velocity vector in inertial frame [vx, vy, vz] of third-body objects
    omega (number): Angular velocity magnitude (rad/s) 
    theta (number): Current rotation angle (rad)
    mass (list of float): Mass of the third-body objects (kg)
    rw_position (np.array): Position vector in inertial frame [x, y, z] of Ringworld

    Returns:
    acceleration_ringworld (np.array): Acceleration of the Ringworld. 
    """

    acceleration_ringworld = np.array([0.,0.,0.]) # Initialize acceleration array

    for i in range(len(i_position)): # Loop through all the third-bodies
        
        # Convert from inertial to rotating reference frame
        r_position = inertial_to_rotating(i_position[i], i_velocity[i], omega, theta)[0]

        # Add to acceleration vector for Ringworld
        acceleration_ringworld += (G * mass[i] / (np.linalg.norm(r_position - rw_position) ** 3) * (r_position - rw_position) - G * mass[i] / (np.linalg.norm(r_position) ** 3) * r_position)

    return acceleration_ringworld

def save_fig(i_position, i_velocity, omega, mass, rw_position, N):
    """
    Calculates gravitational acceleration of the Ringworld under influence of third-body objects.

    Parameters: 
    i_position (list of np.array): Position vector in inertial frame [x, y, z] of third-body objects
    i_velocity (list of np.array): Velocity vector in inertial frame [vx, vy, vz] of third-body objects
    omega (number): Angular velocity magnitude (rad/s) 
    mass (list of float): Mass of the third-body objects (kg)
    rw_position (np.array): Position vector in inertial frame [x, y, z] of Ringworld

    Returns:
    fig.png (plot): Plot of gravity norm vs. angle for many angles
    """
    plt.figure()
    normal = []
    tangential = []
    forces = []
    angles = []
    for n in range(N):
        angles.append(n*2*np.pi/N)
    for theta in angles:
        acceleration_ringworld = compute_gravity(i_position, i_velocity, omega, theta, mass, rw_position)
        normal.append(acceleration_ringworld[0])
        tangential.append(acceleration_ringworld[1])
        forces.append(np.linalg.norm(acceleration_ringworld))
    print(normal)
    print(tangential)
    print(forces)
    print(angles)


#save_fig([[2e8, 0., 0.]], [[0., 0., 0.]], 1e-6, [1e13], [149597871, 0., 0.], 6)

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
    initial_state    = np.concatenate((initial_position, initial_velocity, [0,0]))

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
    final_velocity = solution.y[3:6, -1]

    # Return the final position, final velocity, and the full solution
    return final_position.tolist(), final_velocity.tolist(), solution

def calculate_radial_solar_wind_speed(gamma):
    """
    Finds the radial solar wind speed at a particular point

    Parameters:
    gamma: angle between solar equator and the plane of the Ringworld

    Returns:
    scalar value for solar wind speed in radial direction (m/s)
    """
    if gamma < 28.7:
        v_r = 66176
    elif gamma > 28.7:
        v_r = 546568
    else:
        v_r = 161765
    return v_r
        
def calculate_magnetic_field(radius, omega, v_r):
    """
    Finds the interplanetary magnetic field induced by the Parker Spiral
    This function uses the reference magnetic field for Earth (B_0) and
    the distance between the Sun and Earth (r_0)

    Parameters:
    radius: Radius of rotation (m)
    omega: Angular velocity magnitude (rad/s) 
    v_r: Radial solar wind speed (m/s)

    Returns:
    magnetic_field: 3D vector repesenting the B field experienced by the particle at a particular time and place (T = N*s/C/m)
    """
    B_r = B_0 * (r_0 / radius) ** 2
    B_phi = -omega * radius * B_r / v_r 
    magnetic_field = np.array([B_r, 0, B_phi])
    return magnetic_field

def calculate_electric_field(magnetic_field, radius, omega, v_r):
    """
    Calculates the electric field from solar-wind convection, induced by the magnetic field

    Parameters:
    radius: Radius of rotation (m)
    omega: Angular velocity magnitude (rad/s)
    magnetic_field: Magnetic field vector experienced by particle (T)
    v_r: Radial solar wind speed (m/s)

    Returns:
    electric_field: 3D vector representing the E field experienced by the particle at a particular time and place (N/C)
    """
    v_combined = - (v_r - omega * radius)
    v_vector = np.array([v_combined, 0, 0])
    electric_field = np.cross(v_vector, magnetic_field)
    return electric_field


def calculate_acceleration_from_lorentz_force(particle_charge: float, particle_velocity,particle_mass:float,magnetic_field, electric_field):
    """
    Finds the acceleration a particle expiriences under electric and magnetic forces

    Parameters:
    particle_charge: charge of the particle in coulombs (C)
    particle_velocity: 3D vector representing the particle's current velocity (m/s)
    particle_mass: mass of the particle (kg)
    magnetic_field: 3D vector representing the B field experienced by the particle at a particular time and place (T = N*s/C/m)
    electric_field: 3D vector representing the E field experienced by the particle at a particular time and place (N/C)

    Returns:
    acceleration: 3D vector representing how the lorenz force affects the particle (m/s^2)
    """
    Q = particle_charge
    V = particle_velocity
    M = particle_mass
    B = magnetic_field
    E = electric_field
    force = Q * (E+np.cross(V,B))
    acceleration = force/M
    return acceleration



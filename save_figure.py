import numpy as np
import matplotlib.pyplot as plt

g = 6.6743e-11 # Universal Gravitational Constant
g = g/1e9 

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
        acceleration_ringworld += (g * mass[i] / (np.linalg.norm(r_position - rw_position) ** 3) * (r_position - rw_position) - g * mass[i] / (np.linalg.norm(r_position) ** 3) * r_position)

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
    plt.figure()
    plt.plot(angles, tangential)
    plt.savefig("ringworld forces")
    print(normal)
    print(tangential)
    print(forces)
    print(angles)

save_fig([[2e8, 0., 0.]], [[0., 0., 0.]], 1e-6, [1e13], [149597871, 0., 0.], 1000)

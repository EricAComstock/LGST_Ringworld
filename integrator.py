"""
integrator.py
V1.0, James Stewart, September 24, 2025
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt

G = 6.6743e-11 # Universal Gravitational Constant

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
        acceleration_ringworld += (G * mass[i] / (np.linalg.norm(r_position - rw_position) ** 3) * (r_position - rw_position)) - (G * mass[i] / (np.linalg.norm(r_position) ** 3) * r_position)


    return acceleration_ringworld

def save_fig(i_position, i_velocity, omega, mass, rw_position, N):
    """
    Calculates gravitational acceleration of the Ringworld under influence of third-body objects.

    Parameters: 
    i_position (list of np.array): Position vector in inertial frame [x, y, z] of third-body objects
    i_velocity (list of np.array): Velocity vector in inertial frame [vx, vy, vz] of third-body objects
    omega (number): Angular velocity magnitude (rad/s) 
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
    return [np.array(angles), np.array(tangential)]

from scipy.integrate import cumulative_trapezoid as cumtrapz

def integrate_ringworld_velocity(y_force, angles, radius, ringworld_velocity, normalize_to_first=True):
    """
    Integrate tangential forces with respect to arc-length to produce velocity map.
    
    Parameters:
    -----------
    y_force : numpy.ndarray
        Array of tangential forces (y-component forces)
    angles : numpy.ndarray  
        Array of angles corresponding to positions on the ringworld (in radians)
    radius : float
        Radius of the ringworld
    ringworld_velocity : float
        Orbital velocity of the ringworld (for unit conversion)
    normalize_to_first : bool, default=True
        Whether to normalize the velocity map so the first value is 0
        
    Returns:
    --------
    velocity_map : numpy.ndarray
        Integrated velocity values at each angle position (zero average)
    arc_lengths : numpy.ndarray
        Arc-length coordinates for reference
        
    Notes:
    ------
    - Integration is performed using trapezoidal rule
    - Arc-length is calculated as angle * radius
    - Result is multiplied by 1/ringworld_velocity for correct units
    - Average is subtracted so the velocity map averages to zero
    - Assumes mass = 1 for simplicity (force = acceleration)
    """
    
    # Validate inputs
    if len(y_force) != len(angles):
        raise ValueError("y_force and angles arrays must have the same length")
    
    if len(angles) < 2:
        raise ValueError("Need at least 2 data points for integration")
        
    # Calculate arc-length coordinates
    arc_lengths = angles * radius
    
    # Integrate force with respect to arc-length using trapezoidal rule
    # This gives us the change in velocity (assuming unit mass)
    velocity_changes = cumtrapz(y_force, arc_lengths, initial=0)
    
    # Apply unit conversion factor
    velocity_changes = velocity_changes / ringworld_velocity
    
    # Subtract the average so the result averages to zero
    velocity_map = velocity_changes - np.mean(velocity_changes)
    
    # Normalize to first point if requested
    if normalize_to_first:
        velocity_map = velocity_map - velocity_map[0]
    
    return velocity_map, arc_lengths

def plot_velocity_results(angles, y_force, velocity_map, radius, ringworld_velocity):
    """
    Utility function to plot the force input and velocity output.
    
    Parameters:
    -----------
    angles : numpy.ndarray
        Angle array
    y_force : numpy.ndarray  
        Force array
    velocity_map : numpy.ndarray
        Computed velocity map
    radius : float
        Ringworld radius
    ringworld_velocity : float
        Ringworld orbital velocity
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fontsize = 16
    label_fontsize = 18
    tick_fontsize = 14
    ax1.plot(np.degrees(angles), y_force, 'b-', linewidth=2)
    ax1.set_xlabel('Angle (degrees)', fontsize=label_fontsize)
    ax1.set_ylabel('Tangential Force', fontsize=label_fontsize)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=fontsize, fontweight='bold', va='top', ha='right')
    ax2.plot(np.degrees(angles), velocity_map, 'r-', linewidth=2)
    ax2.set_xlabel('Angle (degrees)', fontsize=label_fontsize)
    ax2.set_ylabel('Velocity Perturbation', fontsize=label_fontsize)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero average')
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.legend(fontsize=fontsize)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=fontsize, fontweight='bold', va='top', ha='right')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__": # omega = 8.53586332E-6, i_position, i_velocity, omega, mass, rw_position, N
    x1 = random.random()
    y1 = random.random()
    x2 = random.random()
    y2 = random.random()
    x3 = random.random()
    y3 = random.random()
    x4 = random.random()
    y4 = random.random()
    x5 = random.random()
    y5 = random.random()
    x6 = random.random()
    y6 = random.random()
    x7 = random.random()
    y7 = random.random()
    x8 = random.random()
    y8 = random.random()
    x9 = random.random()
    y9 = random.random()
    x10 = random.random()
    y10 = random.random()
    x11 = random.random()
    y11 = random.random()
    x12 = random.random()
    y12 = random.random()
    x13 = random.random()
    y13 = random.random()
    x14 = random.random()
    y14 = random.random()
    angles, y_force = save_fig([np.array([1.72*10**9*x1, 1.72*10**9*math.sqrt(1-x1**2)*y1, 1.72*10**9*math.sqrt(1-x1**2)*math.sqrt(1-y1**2)]),
                                np.array([2.36*10**9*x2, 2.36*10**9*math.sqrt(1-x2**2)*y2, 2.36*10**9*math.sqrt(1-x2**2)*math.sqrt(1-y2**2)]),
                                np.array([3.34*10**9*x3, 3.34*10**9*math.sqrt(1-x3**2)*y3, 3.34*10**9*math.sqrt(1-x3**2)*math.sqrt(1-y3**2)]),
                                np.array([4.37*10**9*x4, 4.37*10**9*math.sqrt(1-x4**2)*y4, 4.37*10**9*math.sqrt(1-x4**2)*math.sqrt(1-y4**2)]), 
                                np.array([5.76*10**9*x5, 5.76*10**9*math.sqrt(1-x5**2)*y5, 5.76*10**9*math.sqrt(1-x5**2)*math.sqrt(1-y5**2)]), 
                                np.array([7.02*10**9*x6, 7.02*10**9*math.sqrt(1-x6**2)*y6, 7.02*10**9*math.sqrt(1-x6**2)*math.sqrt(1-y6**2)]), 
                                np.array([9.26*10**9*x7, 9.26*10**9*math.sqrt(1-x7**2)*y7, 9.26*10**9*math.sqrt(1-x7**2)*math.sqrt(1-y7**2)])], 

                               [np.array([8.3*10**4*x8, 8.3*10**4*math.sqrt(1-x8**2)*y8, 8.3*10**4*math.sqrt(1-x8**2)*math.sqrt(1-y8**2)]),
                                np.array([7.1*10**4*x9, 7.1*10**4*math.sqrt(1-x9**2)*y9, 7.1*10**4*math.sqrt(1-x9**2)*math.sqrt(1-y9**2)]),
                                np.array([5.9*10**4*x10, 5.9*10**4*math.sqrt(1-x10**2)*y10, 5.9*10**4*math.sqrt(1-x10**2)*math.sqrt(1-y10**2)]),
                                np.array([5.2*10**4*x11, 5.2*10**4*math.sqrt(1-x11**2)*y11, 5.2*10**4*math.sqrt(1-x11**2)*math.sqrt(1-y11**2)]),
                                np.array([4.5*10**4*x12, 4.5*10**4*math.sqrt(1-x12**2)*y12, 4.5*10**4*math.sqrt(1-x12**2)*math.sqrt(1-y12**2)]),
                                np.array([4.1*10**4*x13, 4.1*10**4*math.sqrt(1-x13**2)*y13, 4.1*10**4*math.sqrt(1-x13**2)*math.sqrt(1-y13**2)]),
                                np.array([3.6*10**4*x14, 3.6*10**4*math.sqrt(1-x14**2)*y14, 3.6*10**4*math.sqrt(1-x14**2)*math.sqrt(1-y14**2)])], 2.2E-5, np.array([8.18E24, 7.83E24, 2.33E24, 4.12E24, 6.21E24, 7.88E24, 1.97E24]), [3.4E+9, 0., 0.], 1000)
    radius = 149597871000*1.1
    ringworld_velocity = np.sqrt(1.1*149597871000*9.81)
    # Integrate to get velocity map
    print(angles)
    print(radius)
    velocity_map, arc_lengths = integrate_ringworld_velocity(y_force, angles, radius, ringworld_velocity)
    
    print(f"Integration complete!")
    print(f"Angle range: {np.degrees(angles[0]):.1f}° to {np.degrees(angles[-1]):.1f}°")
    print(f"Arc-length range: {arc_lengths[0]:.1f} to {arc_lengths[-1]:.1f}")
    print(f"Velocity range: {velocity_map.min():.3f} to {velocity_map.max():.3f}")
    print(f"Final velocity: {velocity_map[-1]:.3f}")
    # Uncomment to plot results
    plot_velocity_results(angles, y_force, velocity_map, radius, ringworld_velocity)

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

def plot_velocity_results(angles1, angles2, angles3, angles4, y_force1, y_force2, y_force3, y_force4, velocity_map1, velocity_map2, velocity_map3, velocity_map4, radius, ringworld_velocity):
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
    # 1 is TRAPPIST-1, 2 is Alpha Centauri, 3 is Sol, 4 is Proxima Centauri
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fontsize = 16
    label_fontsize = 18
    tick_fontsize = 14
    ax1.plot(np.degrees(angles3), y_force3, 'b-', label="Sol", linewidth=2)
    ax1.set_xlabel('Angle (degrees)', fontsize=label_fontsize)
    ax1.set_ylabel('Tangential Force (m/s^2)', fontsize=label_fontsize)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=fontsize, fontweight='bold', va='top', ha='right')
    ax2.plot(np.degrees(angles3), velocity_map3, 'b-', label="Sol", linewidth=2)
    ax2.set_xlabel('Angle (degrees)', fontsize=label_fontsize)
    ax2.set_ylabel('Velocity Perturbation (m/s)', fontsize=label_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.legend(fontsize=fontsize)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=fontsize, fontweight='bold', va='top', ha='right')
    ax1.plot(np.degrees(angles2), y_force2, 'k-', label = "Alpha Centauri", linewidth=2)
    ax2.plot(np.degrees(angles2), velocity_map2, 'k-', label = "Alpha Centauri", linewidth=2)
    ax1.plot(np.degrees(angles4), y_force4, 'm-', label="Proxima Centauri", linewidth=2)
    ax2.plot(np.degrees(angles4), velocity_map4, 'm-', label="Proxima Centauri", linewidth=2)
    ax1.plot(np.degrees(angles1), y_force1, 'y-', label="TRAPPIST-1", linewidth=2)
    ax2.plot(np.degrees(angles1), velocity_map1, 'y-', label="TRAPPIST-1", linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero average')

    ax1.set_yscale('symlog', linthresh=1e-7)
    ax2.set_yscale('symlog', linthresh=4e-3)
    ax1.legend(loc = 'upper right', fontsize = 'medium')
    ax2.legend(loc = 'upper right', fontsize = 'medium')


    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__": # omega = 8.53586332E-6, i_position, i_velocity, omega, mass, rw_position, N
    # 1 is TRAPPIST-1, 2 is Alpha Centauri, 3 is Sol, 4 is Proxima Centauri
    x1 = random.random()
    y1 = random.random()
    x2 = random.random()
    y2 = random.random()
    x3 = random.random()
    y3 = random.random()
    x4 = random.random()``
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
    angles1, y_force1 = save_fig([np.array([1.72*10**9*x1, 1.72*10**9*math.sqrt(1-x1**2)*y1, 1.72*10**9*math.sqrt(1-x1**2)*math.sqrt(1-y1**2)]),
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
    x1 = random.random()
    y1 = random.random()
    x2 = random.random()
    y2 = random.random()
    x3 = random.random()
    y3 = random.random()
    x4 = random.random()
    y4 = random.random()
    angles2, y_force2 = save_fig([np.array([3.25*10**12*x1, 3.25*10**12*math.sqrt(1-x1**2)*y1, 3.25*10**12*math.sqrt(1-x1**2)*math.sqrt(1-y1**2)]),
                                np.array([1.945*10**15*x2, 1.945*10**15*math.sqrt(1-x2**2)*y2, 1.945*10**15*math.sqrt(1-x2**2)*math.sqrt(1-y2**2)])], 

                               [np.array([5200*x3, 5200*math.sqrt(1-x3**2)*y3, 5200*math.sqrt(1-x3**2)*math.sqrt(1-y3**2)]),
                                np.array([309*x4, 309*math.sqrt(1-x4**2)*y4, 309*math.sqrt(1-x4**2)*math.sqrt(1-y4**2)])], 2E-6, np.array([2.19E30, 2.43E29]), [1.06E11, 0., 0.], 1000)
    angles3, y_force3 = save_fig([np.array([-1.690367255730857E+10, -6.761862842133108E+10, -3.975545184741601E+09]),np.array([-8.657290834062675E+10, 6.344380291912919E+10, 5.866856370652959E+09]),np.array([1.431795025515026E+11, 4.265690992143974E+10, -3.681308152420446E+06]), np.array([-1.222736918585425E+11, -1.936980852099699E+11, -1.060706768727213E+09]), np.array([-1.620220057328354E+11, 7.580427870121850E+11, 4.761201117259860E+08]), np.array([1.426350320271618E+12, -3.056008428888809E+10, -5.624474951631455E+10]), np.array([1.520413537345293E+12,2.490416894102039E+12,-1.046532352987754E+10]), np.array([4.469651652396883E+12, 3.842412215808669E+10, -1.037907155122914E+11])], 
                            [np.array([3.748040901281054E+04, -9.368235025154132E+03, -4.203286286629456E+03]),np.array([-2.083840702699918E+04,-2.842395145065711E+04,8.118746856672274E+02]),np.array([-8.978544418765617E+03,2.842892665702806E+04,-2.107282487841644]), np.array([2.140282398794097E+04,-1.085567571073013E+04, -7.523319408541229E+02]), np.array([-1.293951528469888E+04, -2.120838557000888E+03, 2.983633291932282E+02]), np.array([-3.354495346125045E+02, 9.637840053460994E+03, -1.545121165018060E+02]), np.array([-5.875598373949423E+03, 3.232193423003962E+03, 8.804866934798827E+01]), np.array([-9.396658697775445E+01, 5.467997908101429E+03, -1.111796140349235E+02])], 8.53586332E-06, np.array([3.28E+23, 4.87E+24, 5.97E+24, 6.419E+23, 1.899E+27, 5.68E+26, 8.685E+25, 1.024E+26]), [149597871000*1.1, 0., 0.], 1000)
    x1 = random.random()
    y1 = random.random()
    x2 = random.random()
    y2 = random.random()
    x3 = random.random()
    y3 = random.random()
    x4 = random.random()
    y4 = random.random()
    angles4, y_force4 = save_fig([np.array([7255496728.5*x1, 7255496728.5*math.sqrt(1-x1**2)*y1, 7255496728.5*math.sqrt(1-x1**2)*math.sqrt(1-y1**2)]),
                                np.array([4309914654.6*x2, 4309914654.6*math.sqrt(1-x2**2)*y2, 4309914654.6*math.sqrt(1-x2**2)*math.sqrt(1-y2**2)])], 

                               [np.array([4.7*10**4*x3, 4.7*10**4*math.sqrt(1-x3**2)*y3, 4.7*10**4*math.sqrt(1-x3**2)*math.sqrt(1-y3**2)]),
                                np.array([6.2*10**4*x4, 6.2*10**4*math.sqrt(1-x4**2)*y4, 6.2*10**4*math.sqrt(1-x4**2)*math.sqrt(1-y4**2)])], 9E-7, np.array([6.3E24,1.6E24]), [6.2E9, 0., 0.], 1000)

    radius = 149597871000*1.1
    ringworld_velocity = np.sqrt(1.1*149597871000*9.81)
    # Integrate to get velocity map
    velocity_map1, arc_lengths1 = integrate_ringworld_velocity(y_force1, angles1, radius, ringworld_velocity)
    print(f"Integration complete!")
    print(f"Angle range: {np.degrees(angles1[0]):.1f}° to {np.degrees(angles1[-1]):.1f}°")
    print(f"Arc-length range: {arc_lengths1[0]:.1f} to {arc_lengths1[-1]:.1f}")
    print(f"Velocity range: {velocity_map1.min():.3f} to {velocity_map1.max():.3f}")
    print(f"Final velocity: {velocity_map1[-1]:.3f}")

    velocity_map2, arc_lengths2 = integrate_ringworld_velocity(y_force2, angles2, radius, ringworld_velocity)
    
    print(f"Integration complete!")
    print(f"Angle range: {np.degrees(angles2[0]):.1f}° to {np.degrees(angles2[-1]):.1f}°")
    print(f"Arc-length range: {arc_lengths2[0]:.1f} to {arc_lengths2[-1]:.1f}")
    print(f"Velocity range: {velocity_map2.min():.3f} to {velocity_map2.max():.3f}")
    print(f"Final velocity: {velocity_map2[-1]:.3f}")

    velocity_map3, arc_lengths3 = integrate_ringworld_velocity(y_force3, angles3, radius, ringworld_velocity)
    
    print(f"Integration complete!")
    print(f"Angle range: {np.degrees(angles3[0]):.1f}° to {np.degrees(angles3[-1]):.1f}°")
    print(f"Arc-length range: {arc_lengths3[0]:.1f} to {arc_lengths3[-1]:.1f}")
    print(f"Velocity range: {velocity_map3.min():.3f} to {velocity_map3.max():.3f}")
    print(f"Final velocity: {velocity_map3[-1]:.3f}")

    velocity_map4, arc_lengths4 = integrate_ringworld_velocity(y_force4, angles4, radius, ringworld_velocity)
    
    print(f"Integration complete!")
    print(f"Angle range: {np.degrees(angles4[0]):.1f}° to {np.degrees(angles4[-1]):.1f}°")
    print(f"Arc-length range: {arc_lengths4[0]:.1f} to {arc_lengths4[-1]:.1f}")
    print(f"Velocity range: {velocity_map4.min():.3f} to {velocity_map4.max():.3f}")
    print(f"Final velocity: {velocity_map4[-1]:.3f}")


    # Uncomment to plot results
    plot_velocity_results(angles1, angles2, angles3, angles4, y_force1, y_force2, y_force3, y_force4, velocity_map1, velocity_map2, velocity_map3, velocity_map4, radius, ringworld_velocity)

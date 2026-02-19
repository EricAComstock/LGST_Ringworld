"""
integrator.py
V1.0, James Stewart, September 24, 2025
"""
import numpy as np
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
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot forces
    ax1.plot(np.degrees(angles), y_force, 'b-', linewidth=2)
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Tangential Force')
    ax1.set_title('Input: Tangential Forces vs Angle')
    ax1.grid(True, alpha=0.3)
    
    # Plot velocity
    ax2.plot(np.degrees(angles), velocity_map, 'r-', linewidth=2)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Velocity Perturbation')
    ax2.set_title(f'Output: Velocity Map (R = {radius}, V = {ringworld_velocity})')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero average')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": # omega = 8.53586332E-6, i_position, i_velocity, omega, mass, rw_position, N
    angles = save_fig([np.array([-1.690367255730857E+10, -6.761862842133108E+10, -3.975545184741601E+09]),np.array([-8.657290834062675E+10, 6.344380291912919E+10, 5.866856370652959E+09]),np.array([1.431795025515026E+11, 4.265690992143974E+10, -3.681308152420446E+06])], [np.array([3.748040901281054E+04, -9.368235025154132E+03, -4.203286286629456E+03]),np.array([-2.083840702699918E+04,-2.842395145065711E+04,8.118746856672274E+02]),np.array([-8.978544418765617E+03,2.842892665702806E+04,-2.107282487841644])], 8.53586332/10**6, np.array([2e30, 2e30, 2e30]), [149597871000*1.1, 0., 0.], 1000)[0]
    radius = 149597871000*1.1
    ringworld_velocity = np.sqrt(1.1*149597871000*9.81)
    y_force = save_fig([np.array([-1.690367255730857E+10, -6.761862842133108E+10, -3.975545184741601E+09]),np.array([-8.657290834062675E+10, 6.344380291912919E+10, 5.866856370652959E+09]),np.array([1.431795025515026E+11, 4.265690992143974E+10, -3.681308152420446E+06])], [np.array([3.748040901281054E+04, -9.368235025154132E+03, -4.203286286629456E+03]),np.array([-2.083840702699918E+04,-2.842395145065711E+04,8.118746856672274E+02]),np.array([-8.978544418765617E+03,2.842892665702806E+04,-2.107282487841644])], 8.53586332/10**6, np.array([2e30, 2e30, 2e30]), [149597871000*1.1, 0., 0.], 1000)[0]
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


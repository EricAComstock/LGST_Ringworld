import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the solver module
import SolverSharedCodePlusSolar

# Define key physical constants and parameters
g = 9.81  # Gravity (m/s²)
r = 149597870691 # Radius = 1 AU (m)
w = SolverSharedCodePlusSolar.calculate_omega(r, g)

dt = 0.001  # Time step (s)
t0 = 0  # Initial time (s)
tf = 100000 + t0  # Final time (s)

# Initial conditions in the inertial frame
xi, yi, zi = (0, 149597939647, 239934282)  # Initial position (m)
xv, yv, zv =  (-101.90575674494171, 244.0304699844323, 260.43477854647216) # Initial velocity (m/s)


# Define helper functions for coordinate transformations
def inertial_to_rotating(i_position, i_velocity, omega, theta):
    """
    Transform position and velocity from inertial to rotating frame.
    For clockwise rotation (negative omega), we use the appropriate matrix.

    Parameters:
    i_position: Position vector in inertial frame [x, y, z]
    i_velocity: Velocity vector in inertial frame [vx, vy, vz]
    omega: Angular velocity magnitude (rad/s), negative for clockwise rotation
    theta: Current rotation angle (rad)

    Returns:
    r_position: Position vector in rotating frame
    r_velocity: Velocity vector in rotating frame
    """
    # For clockwise rotation (negative omega), the rotation matrix is:
    R_i2r = np.array([
        [np.cos(theta), np.sin(theta), 0],   # Note: sign of sin matches theta sign
        [-np.sin(theta), np.cos(theta), 0],  # For clockwise rotation
        [0, 0, 1]
    ])

    # Transform position
    r_position = R_i2r @ i_position

    # Transform velocity: v_rot = R·v_inertial - (ω × r_rot)
    omega_vector = np.array([0, 0, omega])
    r_velocity = R_i2r @ i_velocity - np.cross(omega_vector, r_position)
    return r_position, r_velocity


def rotating_to_inertial(r_position, r_velocity, omega, theta):
    """
    Transform position and velocity from rotating to inertial frame.
    This must be the inverse/transpose of the inertial_to_rotating matrix.

    Parameters:
    r_position: Position vector in rotating frame [x, y, z]
    r_velocity: Velocity vector in rotating frame [vx, vy, vz]
    omega: Angular velocity magnitude (rad/s), negative for clockwise rotation
    theta: Current rotation angle (rad)

    Returns:
    i_position: Position vector in inertial frame
    i_velocity: Velocity vector in inertial frame
    """
    # For clockwise rotation, the inverse rotation matrix is:
    R_r2i = np.array([
        [np.cos(theta), -np.sin(theta), 0],  # Transpose of the R_i2r matrix
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Transform position
    i_position = R_r2i @ r_position

    # Transform velocity: v_inertial = R·(v_rot + ω × r_rot)
    omega_vector = np.array([0, 0, omega])
    rotational_velocity = np.cross(omega_vector, r_position)
    i_velocity = R_r2i @ (r_velocity + rotational_velocity)

    return i_position, i_velocity


# --------------------- INERTIAL FRAME ---------------------#

# Package as vectors for easier manipulation
ri = np.array([xi, yi, zi])  # Initial position vector
vi = np.array([xv, yv, zv])  # Velocity vector in inertial frame

print("Inertial frame coordinates:")
print(f"Initial position: ({xi:.3f}, {yi:.3f}, {zi:.3f})")
print(f"Angular velocity (omega): {w:.6f} rad/s")  # Added to show the value
print(f"Rotation period: {2*np.pi/abs(w):.6f} seconds")  # Show rotation period

# Calculate final position analytically in inertial frame
# For straight-line motion: r = r₀ + v₀·t
rf_inertial = ri + vi * tf
xf, yf, zf = rf_inertial

print(f"Analytical final position (inertial): ({xf:.3f}, {yf:.3f}, {zf:.3f})")

# --------------------- ROTATING FRAME ---------------------#
# Calculate initial rotation angle
theta_initial = w * t0  # This will be 0 if t0=0

# Convert initial conditions from inertial to rotating frame using helper function
r_rotating_initial, v_rotating_initial = inertial_to_rotating(ri, vi, w, theta_initial)

print("\nRotating frame coordinates:")
print(
    f"Initial position (rotating): ({r_rotating_initial[0]:.3f}, {r_rotating_initial[1]:.3f}, {r_rotating_initial[2]:.3f})")
print(
    f"Initial velocity (rotating): ({v_rotating_initial[0]:.3f}, {v_rotating_initial[1]:.3f}, {v_rotating_initial[2]:.3f})")

# Use the solver to compute the final state in the rotating frame
# The solver works entirely in the rotating frame
final_rot_position, final_rot_velocity = SolverSharedCodePlusSolar.compute_motion(
    r_rotating_initial, v_rotating_initial, r, g, tf, dt, solar_mu=None
)

# Convert from list to numpy array if needed
final_rot_position = np.array(final_rot_position)
final_rot_velocity = np.array(final_rot_velocity)

print(
    f"Final position (rotating): ({final_rot_position[0]:.3f}, {final_rot_position[1]:.3f}, {final_rot_position[2]:.3f})")
print(
    f"Final velocity (rotating): ({final_rot_velocity[0]:.3f}, {final_rot_velocity[1]:.3f}, {final_rot_velocity[2]:.3f})")

# Calculate final rotation angle: θ = ω·t
theta_final = w * tf

# Convert final rotating frame results back to inertial frame using helper function
final_inertial_position, final_inertial_velocity = rotating_to_inertial(
    final_rot_position, final_rot_velocity, w, theta_final)

print("\nFinal results converted back to inertial frame:")
print(
    f"Final position (inertial): ({final_inertial_position[0]:.3f}, {final_inertial_position[1]:.3f}, {final_inertial_position[2]:.3f})")
print(
    f"Final velocity (inertial): ({final_inertial_velocity[0]:.3f}, {final_inertial_velocity[1]:.3f}, {final_inertial_velocity[2]:.3f})")

# --------------------- COMPARISON ---------------------#
# Compare analytical solution with the computed solution
print("\nComparison:")
print(f"Analytical final position (inertial): ({rf_inertial[0]:.3f}, {rf_inertial[1]:.3f}, {rf_inertial[2]:.3f})")
print(
    f"Computed final position (inertial): ({final_inertial_position[0]:.3f}, {final_inertial_position[1]:.3f}, {final_inertial_position[2]:.3f})")
print(
    f"Difference: ({final_inertial_position[0] - rf_inertial[0]:.6f}, {final_inertial_position[1] - rf_inertial[1]:.6f}, {final_inertial_position[2] - rf_inertial[2]:.6f})")

# Calculate error
error_vector = final_inertial_position - rf_inertial
error_magnitude = np.linalg.norm(error_vector)
rel_error_percent = (error_magnitude / np.linalg.norm(rf_inertial)) * 100
print(f"Position error magnitude: {error_magnitude:.6f} meters")
print(f"Relative position error: {rel_error_percent:.8f}%")

# Verify final position matches expected analytical solution
print("\nVerification:")
r_analytical_rotating, _ = inertial_to_rotating(rf_inertial, vi, w, theta_final)
print(
    f"Analytical position in rotating frame: ({r_analytical_rotating[0]:.3f}, {r_analytical_rotating[1]:.3f}, {r_analytical_rotating[2]:.3f})")
print(
    f"Numerical position in rotating frame: ({final_rot_position[0]:.3f}, {final_rot_position[1]:.3f}, {final_rot_position[2]:.3f})")

# Calculate rotating frame error
rotating_error = np.linalg.norm(r_analytical_rotating - final_rot_position)
print(f"Rotating frame error: {rotating_error:.6f} meters")
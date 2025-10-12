"""
tester-mullis-NEW.py
Tests analytical vs numerical integration in inertial and rotating frames
using known initial conditions and coordinate transformation utilities.

V1.0, Matthew Mullis, April 29, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

# Path operations
import SolverSharedCodePlusSolar

# Physical constants and parameters
g = 9.81  # Gravity [m/s²]
r = 149597870691  # Radius = 1 AU [m]
w = SolverSharedCodePlusSolar.calculate_omega(r, g)  # Angular velocity [rad/s]

# Time parameters
dt = 0.001  # Time step [s]
t0 = 0  # Initial time [s]
tf = 10000 + t0  # Final time [s]

# Initial conditions in inertial frame
xi = 0  # Initial x position [m]
yi = 149597939647  # Initial y position [m]
zi = 239934282  # Initial z position [m]
xv = -101.90575674494171  # Initial x velocity [m/s]
yv = 244.0304699844323  # Initial y velocity [m/s]
zv = 260.43477854647216  # Initial z velocity [m/s]


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
    # Rotation matrix for clockwise rotation (negative omega)
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


def rotating_to_inertial(r_position, r_velocity, omega, theta):
    """
    Transform position and velocity from rotating to inertial frame.

    i_position, i_velocity = rotating_to_inertial(r_position, r_velocity, omega, theta)

    Inputs:
    r_position  Position vector in rotating frame [x, y, z] [m]
    r_velocity  Velocity vector in rotating frame [vx, vy, vz] [m/s]
    omega       Angular velocity magnitude [rad/s]
    theta       Current rotation angle [rad]

    Outputs:
    i_position  Position in inertial frame [m]
    i_velocity  Velocity in inertial frame [m/s]
    """
    # Inverse rotation matrix
    R_r2i = np.array([
        [np.cos(theta), -np.sin(theta), 0],
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


# Main calculations
if __name__ == "__main__":
    # INERTIAL FRAME ANALYSIS
    print("=" * 60)
    print("INERTIAL FRAME ANALYSIS")
    print("=" * 60)

    # Package initial conditions as vectors
    ri = np.array([xi, yi, zi])  # Initial position vector
    vi = np.array([xv, yv, zv])  # Initial velocity vector

    print("Inertial frame coordinates:")
    print(f"Initial position: ({xi:.3f}, {yi:.3f}, {zi:.3f})")
    print(f"Angular velocity (omega): {w:.6f} rad/s")
    print(f"Rotation period: {2 * np.pi / abs(w):.6f} seconds")

    # Calculate final position analytically in inertial frame
    rf_inertial = ri + vi * tf  # Simple linear motion
    xf, yf, zf = rf_inertial  # Extract components
    print(f"Analytical final position (inertial): ({xf:.3f}, {yf:.3f}, {zf:.3f})")

    # ROTATING FRAME ANALYSIS
    print("\n" + "=" * 60)
    print("ROTATING FRAME ANALYSIS")
    print("=" * 60)

    theta_initial = w * t0  # Initial rotation angle

    # Convert initial conditions to rotating frame
    r_rotating_initial, v_rotating_initial = inertial_to_rotating(ri, vi, w, theta_initial)

    print("Rotating frame coordinates:")
    print(f"Initial position (rotating): ({r_rotating_initial[0]:.3f}, " +
          f"{r_rotating_initial[1]:.3f}, {r_rotating_initial[2]:.3f})")
    print(f"Initial velocity (rotating): ({v_rotating_initial[0]:.3f}, " +
          f"{v_rotating_initial[1]:.3f}, {v_rotating_initial[2]:.3f})")
    # Use solver to compute motion in rotating frame

    final_rot_position, final_rot_velocity, solution = SolverSharedCodePlusSolar.compute_motion(
        r_rotating_initial, v_rotating_initial, r, g, tf, dt, solar_mu=None
    )
    # Convert results to numpy arrays
    final_rot_position = np.array(final_rot_position)
    final_rot_velocity = np.array(final_rot_velocity)

    print(f"Final position (rotating): ({final_rot_position[0]:.3f}, " +
          f"{final_rot_position[1]:.3f}, {final_rot_position[2]:.3f})")
    print(f"Final velocity (rotating): ({final_rot_velocity[0]:.3f}, " +
          f"{final_rot_velocity[1]:.3f}, {final_rot_velocity[2]:.3f})")

    # Convert back to inertial frame for comparison
    theta_final = w * tf
    final_inertial_position, final_inertial_velocity = rotating_to_inertial(
        final_rot_position, final_rot_velocity, w, theta_final
    )

    print("\nFinal results converted back to inertial frame:")
    print(f"Final position (inertial): ({final_inertial_position[0]:.3f}, " +
          f"{final_inertial_position[1]:.3f}, {final_inertial_position[2]:.3f})")
    print(f"Final velocity (inertial): ({final_inertial_velocity[0]:.3f}, " +
          f"{final_inertial_velocity[1]:.3f}, {final_inertial_velocity[2]:.3f})")

    # ERROR ANALYSIS
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print(f"Analytical final position (inertial): ({rf_inertial[0]:.3f}, " +
          f"{rf_inertial[1]:.3f}, {rf_inertial[2]:.3f})")
    print(f"Computed final position (inertial): ({final_inertial_position[0]:.3f}, " +
          f"{final_inertial_position[1]:.3f}, {final_inertial_position[2]:.3f})")

    # Calculate position error
    error_vector = final_inertial_position - rf_inertial
    error_magnitude = np.linalg.norm(error_vector)
    rel_error_percent = (error_magnitude / np.linalg.norm(rf_inertial)) * 100

    print(f"Difference: ({error_vector[0]:.6f}, {error_vector[1]:.6f}, " +
          f"{error_vector[2]:.6f})")
    print(f"Position error magnitude: {error_magnitude:.6f} meters")
    print(f"Relative position error: {rel_error_percent:.8f}%")

    # VERIFICATION
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Compare analytical and numerical results in rotating frame
    r_analytical_rotating, _ = inertial_to_rotating(rf_inertial, vi, w, theta_final)

    print(f"Analytical position in rotating frame: ({r_analytical_rotating[0]:.3f}, " +
          f"{r_analytical_rotating[1]:.3f}, {r_analytical_rotating[2]:.3f})")
    print(f"Numerical position in rotating frame: ({final_rot_position[0]:.3f}, " +
          f"{final_rot_position[1]:.3f}, {final_rot_position[2]:.3f})")

    rotating_error = np.linalg.norm(r_analytical_rotating - final_rot_position)
    print(f"Rotating frame error: {rotating_error:.6f} meters")

    print("\n" + "=" * 60)
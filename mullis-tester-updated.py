from math import sqrt
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


## initial positon (x,y,z)
xi = 1 # initial x position (I make what i want to send to edwin)
yi = 1 # initial y position (I make what i want to send to edwin)
zi = 1 # initial z position (I make what i want to send to edwin)

g = 9.81
r = 1.496e11 #(1 AU)
w = sqrt(g / r)
t = 10 # time duration 
dt = 0.1 # time step

## 3d velocity (x,y,z) (v(t) = v_0) 
xv = 5 # x velocity
yv = -6 # y velocity
zv = 1.111 # z velocity


## final position (x,y,z)
xf = xi + xv*t # final x position
yf = yi + yv*t # final y position
zf = zi + zv*t # final z position

# gives initial and final position
print("Interial frame coordinates:")
print(f"Initial: ({xi:.3f}, {yi:.3f}, {zi:.3f}), Final: ({xf:.3f}, {yf:.3f}, {zf:.3f})")


## rotation
# Define rotation angle 
theta = np.radians(w*t) # theta = (omega*t) 

# Rotation matrix about Z-axis
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

# Original positions in the inertial frame
vi = np.array([xi, yi, zi])


# Transform into the rotating frame
vri = R_z @ vi
#vrf = R_z @ vf

print(f"Initial Rotating Vector: ({vri[0]:.3f}, {vri[1]:.3f}, {vri[2]:.3f})")

print("\nDebug: Calling main() with the following parameters:")
print(f"  radius={r}, gravity={g}, t_max={t}, dt={dt}")
print(f"  is_rotating=yes, initial position=({vri[0]}, {vri[1]}, {vri[2]}), velocity=({xv}, {yv}, {zv})")


import RK45_Cartesian_Solver  
final_position, final_velocity = RK45_Cartesian_Solver.main(r, g, t, dt, "yes", vri[0], vri[1], vri[2], xv, yv, zv)


if final_position is None:
    print("\nError: No valid output received from RK45_Cartesian_Solver.main()")
else:
    print("\nFinal Results from Tester Script:")
    print(f"  Final Position: {final_position}")
    print(f"  Final Velocity: {final_velocity}")

    
def rotating_to_cartesian(x_prime, y_prime, z_prime, theta, axis="z"):
    
    #Converts coordinates from a rotating reference frame back to Cartesian coordinates in 3D.
    #x_prime: x-coordinate in rotating frame
    #y_prime: y-coordinate in rotating frame
    #z_prime: z-coordinate in rotating frame
    #theta: Rotation angle in radians
    #axis: Axis of rotation ('x', 'y', or 'z')
    #return: (x, y, z) in fixed Cartesian coordinates

    # Define rotation matrices for each axis
    if axis == "x":
        R = np.array([
            [1, 0,                          0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    elif axis == "y":
        R = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,              1,             0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == "z":
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Apply transformation
    rotated_coords = R @ np.array([x_prime, y_prime, z_prime])

    return rotated_coords[0], rotated_coords[1], rotated_coords[2]

f_c_cart = rotating_to_cartesian(final_position[0], final_position[1], final_position[2], theta, 'z')

print(f"Final: ({final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f})")

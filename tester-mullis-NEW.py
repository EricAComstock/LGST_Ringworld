from math import sqrt
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


## initial positon (x,y,z)
xi = 1 # initial x position (I make what i want to send to edwin)
yi = 0 # initial y position (I make what i want to send to edwin)
zi = 0 # initial z position (I make what i want to send to edwin)

g = 9.81
r = 1 #(1 AU) 1.496e11
w = sqrt(g / r)
dt = 0.1 # time step
t0 = 0
tf = .1 + t0 # time duration 



## 3d velocity (x,y,z) (v(t) = v_0) 
xv = 1 # x velocity
yv = 0 # y velocity
zv = 0 # z velocity

v = np.array([xv, yv, zv])

## final position (x,y,z)
xf = xi + xv*tf # final x position
yf = yi + yv*tf # final y position
zf = zi + zv*tf # final z position

# gives initial and final position
print("Interial frame coordinates:")
print(f"Initial: ({xi:.3f}, {yi:.3f}, {zi:.3f}), Final: ({xf:.3f}, {yf:.3f}, {zf:.3f})")


## rotation
# Define rotation angle 
theta = np.radians(w*t0) # theta = (omega*t) 

# Rotation matrix about Z-axis
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

# Original positions in the inertial frame
ri = np.array([xi, yi, zi])


# Transform into the rotating frame
vri = R_z @ ri

vri = np.array(vri)
print(f"Initial Rotating Vector: ({vri[0]:.3f}, {vri[1]:.3f}, {vri[2]:.3f})")


import SolverSharedCode
final_position, final_velocity = SolverSharedCode.compute_motion(vri, v, r, g, tf, dt, is_rotating="yes")

    
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

theta = -np.radians(w*tf) # theta = (omega*t) 

print(theta)

f_c_cart = rotating_to_cartesian(final_position[0], final_position[1], final_position[2], theta, 'z')
print(f_c_cart)
print(f"Final: ({final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f})")

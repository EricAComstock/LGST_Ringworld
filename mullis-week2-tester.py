import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## initial positon (x,y,z)
xi = 1 # initial x position
yi = 1 # initial y position
zi = 1 # initial z position


t = 10 # time duration
 

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
print("Initial: (", xi, yi, zi, ")", "Final: (", xf, yf, zf, ")")


## rotation
# Define rotation angle (e.g., 45 degrees in radians)
theta = np.radians(45)

# Rotation matrix about Z-axis
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

# Original positions in the inertial frame
vi = np.array([xi, yi, zi])
vf = np.array([xf, yf, zf])


# Transform into the rotating frame
vri = R_z @ vi
vrf = R_z @ vf
print("Rotated vectors:", vri, vrf)
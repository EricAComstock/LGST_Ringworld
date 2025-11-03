"""
tester-chambers.py

Tests Lorentz force in SSCPS.py

Version: 1.0
Author: James Barritt Chambers
Date: July 30, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

#testing lorentz forces
from SolverSharedCodePlusSolar import calculate_acceleration_from_lorentz_force as lorentz

charge = -2
velocity = np.array([2,-1,0])
mass = 2
B = np.array([-3,1,0])
E = np.array([1,1,0])
omega = .01
r = np.array([100,0,0])

print(lorentz(charge,velocity,mass,B,E, omega,r))

"""
tester-chambers.py

Tests Lorentz force in SSCPS.py

Version: 1.0
Author: James Barritt Chambers
Date: July 30, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from SolverSharedCodePlusSolar import calculate_acceleration_from_lorentz_force as lorentz

charge = -1
velocity = np.array([2,-1,0])
mass = 1
B = np.array([-3,1,0])
E = np.array([1,1,0])

print(lorentz(charge,velocity,mass,B,E))

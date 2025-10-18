# -*- coding: utf-8 -*-
"""
File: wind_table.py
Created on Sun Sep 14 00:27:44 2025
Updated on Sun Sep 28 by Adrian Carrasco
Author: Adrian Carrasco

Description:
    This script prints a table of wind velocities calculated using the Log Law.

Inputs:
    - U_bulk: Wind speed at reference height (m/s)
    - z_ref: Reference height (m)
    - z_values: Heights to calculate wind speed at 
                (can be a tuple [start, stop, points] or an array)
    - z0: Roughness length (m) [for log law]
    - d: Zero-plane displacement height (m) [for log law]

Outputs:
    - A printed table showing height vs. velocity from the model.
"""

import numpy as np # For numerical operations
from wind_equations import log_law_velocity # Import function

# ------------------------------------------------------------
# Function to print table of wind speeds
# ------------------------------------------------------------
def print_table(U_bulk, z_ref, z_values, z0, d):
    """
Prints a table of wind speeds calculated using the
Log Law model.
"""
    # --------------------------------------------------------
    # Convert tuple to array if needed
    # Example: (10, 1000, 10) -> array of values from 10 to 1000
    # --------------------------------------------------------
    if isinstance(z_values, tuple):
        z_values = np.linspace(*z_values)

      # --------------------------------------------------------
    # Calculate velocities using the model
    # --------------------------------------------------------
    velocity_profile1 = log_law_velocity(U_bulk, z_ref, z_values, z0, d)

    # Print header
    print("Height (m)   Log Law (m/s)")
    print("--------------------------")

    # Print each row
    for z, p in zip(z_values, velocity_profile1):
        print(f"{z:8.2f}      {p:8.2f}")
    
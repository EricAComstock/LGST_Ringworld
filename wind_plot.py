# -*- coding: utf-8 -*-
"""
File: wind_plot.py
Created on Sun Sep 14 00:38:36 2025
Updated on Sun Sep 28 by Adrian Carrasco
Author: Adrian Carrasco

Description:
    This script generates a plot comparing wind velocity profiles 
    calculated using the Log Law model.

Inputs:
    - U_bulk: Wind speed at reference height (m/s)
    - z_ref: Reference height (m)
    - z_values: Heights to calculate wind speed at 
                (can be a tuple [start, stop, points] or an array)
    - z0: Roughness length (m) [for log law]
    - d: Zero-plane displacement height (m) [for log law]

Outputs:
    - A visual plot showing wind speed vs. height for the model.
"""

import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting graphs
from wind_equations import log_law_velocity # Velocity equation

# ------------------------------------------------------------
# Function to generate and display the plot
# ------------------------------------------------------------
def plot_profiles(U_bulk, z_ref, z_values, z0, d):
    """
   Creates a plot for Log Law wind profiles.
   """
   
   # --------------------------------------------------------
   # Convert tuple to array if needed
   # Example: (10, 1000, 100) -> array of 100 points between 10 and 1000
   # --------------------------------------------------------
    if isinstance(z_values, tuple):
        z_values = np.linspace(*z_values)
    # --------------------------------------------------------
    # Calculate velocities for the model
    # --------------------------------------------------------
    velocity_profile1 = log_law_velocity(U_bulk, z_ref, z_values, z0, d)

    # Create the plot
    plt.plot(velocity_profile1, z_values, label="Log Law")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Height (m)")
    plt.title("PBL Wind Profile for Log Law")
    plt.grid(True) # Add grid for readability
    plt.legend() # Display legend
    plt.show() # Show the plot
# -*- coding: utf-8 -*-
"""
File: wind_equations.py
Created on Sun Sep 14 00:22:00 2025
Updated on Sun Sep 28 by Adrian Carrasco
Author: Adrian Carrasco

Description:
    This file contains one function used to calculate atmospheric wind velocity 
    profiles in the turbulent boundary layer. The model included is:
    1. Log Law Velocity Profile

Inputs:
    - U_bulk: Wind speed at reference height (m/s)
    - z_ref: Reference height (m)
    - z_values: Heights at which to calculate velocity (can be a tuple or array)
    - z0: Roughness length (m) [for log law only]
    - d: Zero-plane displacement height (m) [for log law only]

Outputs:
    - Arrays containing the calculated velocities at each specified height
"""

import numpy as np   # Library for numerical operations
# ------------------------------------------------------------
# Log Law Velocity Profile
# ------------------------------------------------------------
def log_law_velocity(U_bulk, z_ref, z_values, z0, d):
    """
Calculate wind speed using the Log Law model.

Parameters:
    U_bulk (float): Wind speed at reference height (m/s)
    z_ref (float): Reference height (m)
    z_values (tuple or array): Heights to compute wind speed
    z0 (float): Roughness length (m)
    d (float): Zero-plane displacement height (m)

Returns:
    numpy array: Wind speeds at each height
"""
  # Convert z_values to an array if a tuple is provided
    if isinstance(z_values, tuple):
        z_values = np.linspace(*z_values)
        
    # 1. Check that z0 is physically valid
    if z0 <= 0:
        raise ValueError("Roughness length (z0) must be greater than 0.")

    # 2. Check that reference height is valid
    if z_ref <= d:
        raise ValueError("Reference height (z_ref) must be greater than zero-plane displacement (d).")

    # 3. Check that all z values are valid
    if np.any(z_values <= d):
        raise ValueError("All height values (z) must be greater than zero-plane displacement (d).")

    # Log Law equation
    return U_bulk * (np.log((z_values - d)/z0) / np.log((z_ref - d)/z0))
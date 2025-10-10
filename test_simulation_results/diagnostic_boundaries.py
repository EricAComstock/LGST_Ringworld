#!/usr/bin/env python3
"""
Diagnostic script to analyze boundary conditions and understand simulation results.
"""

import numpy as np

# Parameters from run_simulation.py
ringworld_radius_km = 8.19381e+12  # km
ringworld_width_km = 81938128337  # km
y_floor = 149597870691  # Ringworld floor (1 AU)
y_min = 149597870691 + 218 * 1000  # Minimum spawn altitude (m)
y_max = 149597870691 + 218 * 1000 + 10 * 1000  # Maximum spawn altitude (m)
z_length = ringworld_width_km * 1000  # Convert width to meters

# Default boundaries from TrajectoryClassification_numpy.py
default_z_length = 10000 * 1000  # Total z-length [m]
default_beta = default_z_length / 2  # Lateral boundary [m]
default_y_floor = 149597870691  # Ringworld floor [m]
default_alpha = default_y_floor - (218 * 1000)  # Atmosphere boundary [m]

# Calculated boundaries that should be used
actual_beta = z_length / 2  # Lateral boundary based on actual ringworld width
actual_alpha = y_floor - (218 * 1000)  # Atmosphere boundary

print("=== BOUNDARY CONDITION ANALYSIS ===")
print()

print("Simulation Parameters:")
print(f"  Ringworld radius: {ringworld_radius_km:.2e} km")
print(f"  Ringworld width: {ringworld_width_km:.2e} km")
print(f"  Y floor: {y_floor:.2e} m")
print(f"  Spawn altitude range: {y_min:.2e} to {y_max:.2e} m")
print(f"  Z length: {z_length:.2e} m")
print()

print("Default Boundaries (from TrajectoryClassification_numpy.py):")
print(f"  Default beta (lateral): {default_beta:.2e} m ({default_beta/1000:.0f} km)")
print(f"  Default alpha (atmosphere): {default_alpha:.2e} m")
print(f"  Default z_length: {default_z_length:.2e} m ({default_z_length/1000:.0f} km)")
print()

print("Actual Boundaries (should be used):")
print(f"  Actual beta (lateral): {actual_beta:.2e} m ({actual_beta/1000:.2e} km)")
print(f"  Actual alpha (atmosphere): {actual_alpha:.2e} m")
print(f"  Actual z_length: {z_length:.2e} m ({z_length/1000:.2e} km)")
print()

print("=== PROBLEM ANALYSIS ===")
print()

# Check if boundaries are being set correctly
if default_beta != actual_beta:
    print("❌ CRITICAL ISSUE: Beta boundary mismatch!")
    print(f"   Default beta: {default_beta/1000:.0f} km")
    print(f"   Should be: {actual_beta/1000:.2e} km")
    print(f"   Ratio: {actual_beta/default_beta:.2e}x larger")
    print()

# Analyze spawn conditions
spawn_altitude_above_floor = y_min - y_floor
atmosphere_thickness = 218 * 1000  # 218 km
spawn_above_alpha = y_min - actual_alpha

print("Spawn Condition Analysis:")
print(f"  Particles spawn at: {spawn_altitude_above_floor/1000:.0f} km above floor")
print(f"  Atmosphere boundary at: {atmosphere_thickness/1000:.0f} km above floor")
print(f"  Particles spawn: {spawn_above_alpha/1000:.0f} km above atmosphere boundary")
print()

if spawn_above_alpha > 0:
    print("✅ Particles spawn above atmosphere (in space)")
else:
    print("❌ Particles spawn below atmosphere boundary")

print()

# Classification logic explanation
print("=== CLASSIFICATION LOGIC ===")
print()
print("Classification rules:")
print("- ESCAPED: Goes below alpha while outside beta, OR ends outside beta")
print("- RECAPTURED: Ends below alpha within beta, OR hits beta while below alpha")  
print("- RESIMULATE: Ends above alpha within beta (undetermined fate)")
print()

print("Why you might see 71.4% resimulation:")
print("1. If beta boundary is much larger than expected, most particles stay 'within beta'")
print("2. If particles don't have enough time/energy to fall below alpha")
print("3. If simulation time is too short for particles to reach final state")
print()

print("Why you might see 0% recaptured:")
print("1. If particles never fall below the atmosphere boundary (alpha)")
print("2. If the spawn conditions give particles too much energy")
print("3. If gravity is too weak relative to initial velocities")
print()

print("=== RECOMMENDATIONS ===")
print()
print("1. Check if TCVarInput() is being called to set correct boundaries")
print("2. Verify that beta = z_length/2 is being used (not default 5M km)")
print("3. Consider increasing simulation time if many particles need resimulation")
print("4. Check initial velocity conditions - they might be too high")
print("5. Verify gravity and other physical parameters")

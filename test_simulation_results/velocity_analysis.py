#!/usr/bin/env python3
"""
Analyze the velocity distributions and escape conditions.
"""

import numpy as np
from scipy.stats import maxwell

# Simulation parameters
T = 289  # Temperature [K]
m = 2.6566962e-26 * 2  # Mass of O2 molecule [kg]
gravity = 2.743176313  # m/s²
spawn_height = 218 * 1000  # 218 km above floor [m]
atmosphere_thickness = 218 * 1000  # 218 km [m]

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant [J/K]

print("=== VELOCITY AND ESCAPE ANALYSIS ===")
print()

# Maxwell-Boltzmann distribution parameters
scale = np.sqrt(k_B * T / m)
print(f"Maxwell-Boltzmann scale parameter: {scale:.2f} m/s")

# Calculate typical velocities
mean_speed = maxwell.mean(scale=scale)
median_speed = maxwell.median(scale=scale)
# Maxwell distribution mode is at scale * sqrt(2)
mode_speed = scale * np.sqrt(2)

print(f"Typical speeds from Maxwell-Boltzmann distribution:")
print(f"  Mean speed: {mean_speed:.2f} m/s")
print(f"  Median speed: {median_speed:.2f} m/s")
print(f"  Mode speed: {mode_speed:.2f} m/s")
print()

# Calculate escape velocity from spawn height
# For a particle to fall back down, it needs v < sqrt(2*g*h)
escape_velocity = np.sqrt(2 * gravity * spawn_height)
print(f"Escape velocity from spawn height ({spawn_height/1000:.0f} km): {escape_velocity:.2f} m/s")
print()

# Calculate what fraction of particles have velocities above escape velocity
# This is approximate since we're looking at speed magnitude, not just vertical component
prob_escape = 1 - maxwell.cdf(escape_velocity, scale=scale)
print(f"Probability of particle having speed > escape velocity: {prob_escape*100:.1f}%")
print()

# For more realistic analysis, consider only the vertical component
# The vertical component of velocity is normally distributed with std = scale/sqrt(3)
vertical_std = scale / np.sqrt(3)
print(f"Standard deviation of vertical velocity component: {vertical_std:.2f} m/s")

# Probability that |v_y| > escape_velocity (rough approximation)
from scipy.stats import norm
prob_vertical_escape = 2 * (1 - norm.cdf(escape_velocity, 0, vertical_std))
print(f"Probability of |v_vertical| > escape velocity: {prob_vertical_escape*100:.1f}%")
print()

print("=== RECOMMENDATIONS ===")
print()

if mean_speed > escape_velocity:
    print("❌ PROBLEM: Mean thermal speed > escape velocity!")
    print("Most particles have enough energy to escape.")
    print()
    print("Solutions:")
    print("1. Reduce temperature (currently 289K)")
    print("2. Increase gravity (currently 2.74 m/s²)")
    print("3. Spawn particles closer to the surface")
    print("4. Add a velocity damping factor")
    print()
    
    # Calculate required temperature for reasonable recapture
    required_scale = escape_velocity / 2  # Target mean speed = escape_velocity / 2
    required_temp = (required_scale**2 * m) / k_B
    print(f"For better recapture, reduce temperature to ~{required_temp:.0f}K")
    
    # Or calculate required gravity
    required_gravity = (mean_speed**2) / (2 * spawn_height)
    print(f"Or increase gravity to ~{required_gravity:.1f} m/s²")
    
else:
    print("✅ Thermal speeds are reasonable relative to escape velocity")

print()
print("=== CURRENT SIMULATION ANALYSIS ===")
print(f"Temperature: {T}K")
print(f"Gravity: {gravity} m/s²") 
print(f"Spawn height: {spawn_height/1000} km")
print(f"Mean thermal speed: {mean_speed:.1f} m/s")
print(f"Escape velocity: {escape_velocity:.1f} m/s")
print(f"Ratio (thermal/escape): {mean_speed/escape_velocity:.2f}")

if mean_speed/escape_velocity > 0.5:
    print("⚠️  High ratio suggests many particles will escape")
else:
    print("✅ Reasonable ratio for some recapture")

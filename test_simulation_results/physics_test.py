#!/usr/bin/env python3
"""
Test the physics implementation to see if gravity is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from SolverSharedCodePlusSolar import compute_motion, SSCPSVarInput
from StochasticInput import stochastic_initial_conditions, SIVarInput

print("=== PHYSICS IMPLEMENTATION TEST ===")
print()

# Test parameters
gravity = 9.81
radius = 149597870691  # 1 AU
y_floor = 149597870691
y_min = y_floor + 218 * 1000  # 218 km above floor
y_max = y_min + 10 * 1000    # 228 km above floor
z_length = 1000 * 1000       # 1000 km width for simple test
temperature = 289

print("Test parameters:")
print(f"  Gravity: {gravity} m/s²")
print(f"  Spawn height: {(y_min - y_floor)/1000:.0f} km above floor")
print(f"  Temperature: {temperature} K")
print()

# Initialize modules
SSCPSVarInput(radius, gravity, is_rotating=False)
SIVarInput(temperature, y_min, y_max, z_length, y_floor)

# Generate a test particle with minimal velocity
print("=== TEST PARTICLE ===")

# Create a particle with very small initial velocity
initial_conditions = stochastic_initial_conditions(temperature, y_min, y_max, z_length, [("O2", 2.6566962e-26 * 2, 0, 100)])
x0, y0, z0, vx0, vy0, vz0, m, q = initial_conditions

print(f"Initial position: ({x0:.0f}, {y0:.2e}, {z0:.0f}) m")
print(f"Initial velocity: ({vx0:.1f}, {vy0:.1f}, {vz0:.1f}) m/s")
print(f"Initial radial distance: {np.sqrt(x0**2 + y0**2):.2e} m")
print(f"Distance above floor: {np.sqrt(x0**2 + y0**2) - y_floor:.0f} m")
print()

# Test a simple trajectory with small time steps
print("=== TRAJECTORY TEST ===")
t_max = 1000  # 1000 seconds
dt = 1        # 1 second steps

# Initial state
state = np.array([x0, y0, z0, vx0, vy0, vz0])
times = [0]
positions = [state[:3].copy()]
velocities = [state[3:].copy()]

print("Computing trajectory...")
for step in range(int(t_max/dt)):
    t = step * dt
    
    # Compute motion for one time step
    try:
        new_state = compute_motion(state, t, dt, m, q)
        state = new_state
        
        if step % 100 == 0:  # Print every 100 steps
            r = np.sqrt(state[0]**2 + state[1]**2)
            print(f"  t={t:4.0f}s: r={r:.2e} m, height={r-y_floor:.0f} m")
        
        times.append(t + dt)
        positions.append(state[:3].copy())
        velocities.append(state[3:].copy())
        
    except Exception as e:
        print(f"Error at t={t}: {e}")
        break

# Analyze the trajectory
positions = np.array(positions)
velocities = np.array(velocities)
radial_distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)

print()
print("=== TRAJECTORY ANALYSIS ===")
print(f"Initial radial distance: {radial_distances[0]:.2e} m")
print(f"Final radial distance: {radial_distances[-1]:.2e} m")
print(f"Change in radial distance: {radial_distances[-1] - radial_distances[0]:.0f} m")

if radial_distances[-1] > radial_distances[0]:
    print("❌ Particle moved OUTWARD (away from ringworld)")
    print("This suggests gravity is repulsive or there's a coordinate issue!")
else:
    print("✅ Particle moved INWARD (toward ringworld)")

print(f"Final height above floor: {radial_distances[-1] - y_floor:.0f} m")
print(f"Atmosphere boundary: {y_floor - (218*1000):.2e} m")

if radial_distances[-1] < (y_floor - 218*1000):
    print("✅ Particle fell below atmosphere boundary")
else:
    print("❌ Particle stayed above atmosphere boundary")

# Check velocity changes
initial_speed = np.linalg.norm(velocities[0])
final_speed = np.linalg.norm(velocities[-1])
print(f"Initial speed: {initial_speed:.1f} m/s")
print(f"Final speed: {final_speed:.1f} m/s")

if final_speed > initial_speed:
    print("✅ Particle gained speed (falling toward ringworld)")
else:
    print("❌ Particle lost speed (moving away from ringworld)")

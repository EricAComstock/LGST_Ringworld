#!/usr/bin/env python3
"""
Debug script to verify what parameters are actually being used in the simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StochasticInputRK45Solver import get_simulation_params

# Parameters from run_simulation.py
ringworld_radius_km = 8.19381e+12  # km
ringworld_width_km = 81938128337  # km
gravity = 2.743176313  # m/s²

sim_params = {
    'radius': ringworld_radius_km,
    'gravity': gravity,
    't_max': 1e6,
    'dt': 100,
    'is_rotating': True,
    'num_particles': 50,
    'save_results': True,
    'show_plots': False,
    'find_leak_rate': True,
    'temperature': 289,
    'y_min': 149597870691 + 218 * 1000,  # Minimum spawn altitude (m)
    'y_max': 149597870691 + 218 * 1000 + 10 * 1000,  # Maximum spawn altitude (m)
    'z_length': ringworld_width_km * 1000,  # Convert width to meters
    'y_floor': 149597870691,  # Ringworld floor (1 AU)
    'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]
}

print("=== PARAMETER DEBUG ===")
print()

print("Input parameters from run_simulation.py:")
for key, value in sim_params.items():
    if key in ['y_min', 'y_max', 'y_floor', 'z_length']:
        print(f"  {key}: {value:.2e} m")
    else:
        print(f"  {key}: {value}")
print()

# Get the processed parameters
params = get_simulation_params(sim_params)

print("Processed parameters after get_simulation_params():")
critical_params = ['y_min', 'y_max', 'y_floor', 'z_length', 'alpha', 'beta', 'gravity', 't_max', 'temperature']
for key in critical_params:
    if key in params:
        if key in ['y_min', 'y_max', 'y_floor', 'z_length', 'alpha', 'beta']:
            print(f"  {key}: {params[key]:.2e} m")
        else:
            print(f"  {key}: {params[key]}")
    else:
        print(f"  {key}: NOT FOUND")
print()

# Check spawn height logic
y_floor = params['y_floor']
y_min = params['y_min']
y_max = params['y_max']
alpha = params['alpha']

spawn_height_min = y_min - y_floor
spawn_height_max = y_max - y_floor
atmosphere_boundary = y_floor - alpha

print("=== SPAWN HEIGHT ANALYSIS ===")
print(f"Y floor: {y_floor:.2e} m")
print(f"Alpha (atmosphere boundary): {alpha:.2e} m")
print(f"Y min (spawn): {y_min:.2e} m")
print(f"Y max (spawn): {y_max:.2e} m")
print()
print(f"Spawn height above floor: {spawn_height_min:.0f} to {spawn_height_max:.0f} m")
print(f"Atmosphere thickness: {atmosphere_boundary:.0f} m")
print()

if y_min < y_floor:
    print("❌ CRITICAL ERROR: Particles spawn BELOW the ringworld floor!")
elif y_min < alpha:
    print("❌ ERROR: Particles spawn BELOW the atmosphere boundary!")
elif y_min > alpha:
    print("✅ Particles spawn ABOVE the atmosphere boundary (in space)")
    print(f"   Distance above atmosphere: {y_min - alpha:.0f} m")

print()
print("=== BOUNDARY CONDITIONS ===")
beta = params['beta']
print(f"Beta (lateral boundary): {beta:.2e} m ({beta/1000:.2e} km)")
print(f"Z length: {params['z_length']:.2e} m ({params['z_length']/1000:.2e} km)")
print(f"Beta should be z_length/2: {params['z_length']/2:.2e} m")

if abs(beta - params['z_length']/2) < 1:
    print("✅ Beta calculation is correct")
else:
    print("❌ Beta calculation is wrong!")

print()
print("=== PHYSICS PARAMETERS ===")
print(f"Gravity: {params['gravity']} m/s²")
print(f"Temperature: {params['temperature']} K")
print(f"Simulation time: {params['t_max']:.0f} s ({params['t_max']/3600:.1f} hours)")
print(f"Is rotating: {params.get('is_rotating', 'NOT SET')}")

# Calculate some physics
import numpy as np
fall_time = np.sqrt(2 * spawn_height_min / params['gravity'])
print(f"Free fall time from spawn height: {fall_time:.0f} s ({fall_time/3600:.2f} hours)")

if params['t_max'] > fall_time * 2:
    print("✅ Simulation time should be sufficient")
else:
    print("⚠️  Simulation time might be too short")

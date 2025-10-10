#!/usr/bin/env python3
"""
Test script to verify that boundary conditions are being set correctly.
"""

from run_simulation import run_simulation_with_params
from TrajectoryClassification_numpy import alpha, beta, z_length, y_floor

print("=== TESTING BOUNDARY CONDITIONS ===")
print()

print("Before simulation:")
print(f"  Default alpha: {alpha:.2e} m")
print(f"  Default beta: {beta:.2e} m ({beta/1000:.0f} km)")
print(f"  Default z_length: {z_length:.2e} m ({z_length/1000:.0f} km)")
print(f"  Default y_floor: {y_floor:.2e} m")
print()

# This will trigger the parameter setup but we'll catch any errors
try:
    print("Attempting to run simulation setup...")
    # We'll just import the main function and check parameters
    from StochasticInputRK45Solver_Siyona import get_simulation_params
    
    # Test with the parameters from run_simulation.py
    ringworld_width_km = 81938128337  # km
    test_params = {
        'z_length': ringworld_width_km * 1000,  # Convert to meters
        'y_floor': 149597870691,
        'y_min': 149597870691 + 218 * 1000,
        'y_max': 149597870691 + 218 * 1000 + 10 * 1000,
    }
    
    params = get_simulation_params(test_params)
    
    print("Calculated parameters:")
    print(f"  z_length: {params['z_length']:.2e} m ({params['z_length']/1000:.2e} km)")
    print(f"  beta: {params['beta']:.2e} m ({params['beta']/1000:.2e} km)")
    print(f"  alpha: {params['alpha']:.2e} m")
    print(f"  y_floor: {params['y_floor']:.2e} m")
    print(f"  y_min: {params['y_min']:.2e} m")
    print(f"  y_max: {params['y_max']:.2e} m")
    print()
    
    expected_beta = params['z_length'] / 2
    if abs(params['beta'] - expected_beta) < 1:
        print("✅ Beta calculation is correct!")
    else:
        print("❌ Beta calculation is wrong!")
        
    expected_alpha = params['y_floor'] - (218 * 1000)
    if abs(params['alpha'] - expected_alpha) < 1:
        print("✅ Alpha calculation is correct!")
    else:
        print("❌ Alpha calculation is wrong!")
        
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()

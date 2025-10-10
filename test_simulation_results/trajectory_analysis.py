#!/usr/bin/env python3
"""
Analyze individual particle trajectories to understand why particles don't get recaptured.
"""

import pandas as pd
import numpy as np
from glob import glob
import os

# Find the most recent results file
result_files = glob("particle_data_*.xlsx")
if result_files:
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Analyzing: {latest_file}")
    
    try:
        df = pd.read_excel(latest_file)
        print(f"Loaded {len(df)} particle trajectories")
        print()
        
        # Simulation parameters
        gravity = 2.743176313  # m/s²
        t_max = 1e6  # seconds
        dt = 100  # seconds
        y_floor = 149597870691  # 1 AU
        alpha = y_floor - (218 * 1000)  # Atmosphere boundary
        
        print("=== SIMULATION PARAMETERS ===")
        print(f"Simulation time: {t_max/1000:.0f} ks ({t_max/3600:.1f} hours)")
        print(f"Time step: {dt} s")
        print(f"Gravity: {gravity} m/s²")
        print(f"Alpha boundary: {alpha:.2e} m")
        print()
        
        # Analyze initial vs final positions
        print("=== POSITION ANALYSIS ===")
        
        initial_r = np.sqrt(df['Initial x']**2 + df['Initial y']**2)
        final_r = np.sqrt(df['Final x']**2 + df['Final y']**2)
        
        print(f"Initial radial positions:")
        print(f"  Min: {np.min(initial_r):.2e} m")
        print(f"  Max: {np.max(initial_r):.2e} m")
        print(f"  Mean: {np.mean(initial_r):.2e} m")
        
        print(f"Final radial positions:")
        print(f"  Min: {np.min(final_r):.2e} m")
        print(f"  Max: {np.max(final_r):.2e} m")
        print(f"  Mean: {np.mean(final_r):.2e} m")
        
        # Check if particles are moving inward or outward
        radial_change = final_r - initial_r
        print(f"Radial change (final - initial):")
        print(f"  Min: {np.min(radial_change):.2e} m")
        print(f"  Max: {np.max(radial_change):.2e} m")
        print(f"  Mean: {np.mean(radial_change):.2e} m")
        
        moving_inward = np.sum(radial_change < 0)
        moving_outward = np.sum(radial_change > 0)
        
        print(f"Particles moving inward: {moving_inward} ({moving_inward/len(df)*100:.1f}%)")
        print(f"Particles moving outward: {moving_outward} ({moving_outward/len(df)*100:.1f}%)")
        print()
        
        # Analyze velocities
        print("=== VELOCITY ANALYSIS ===")
        
        initial_v_mag = np.sqrt(df['Initial vx']**2 + df['Initial vy']**2 + df['Initial vz']**2)
        final_v_mag = np.sqrt(df['Final vx']**2 + df['Final vy']**2 + df['Final vz']**2)
        
        print(f"Initial velocity magnitudes:")
        print(f"  Min: {np.min(initial_v_mag):.1f} m/s")
        print(f"  Max: {np.max(initial_v_mag):.1f} m/s")
        print(f"  Mean: {np.mean(initial_v_mag):.1f} m/s")
        
        print(f"Final velocity magnitudes:")
        print(f"  Min: {np.min(final_v_mag):.1f} m/s")
        print(f"  Max: {np.max(final_v_mag):.1f} m/s")
        print(f"  Mean: {np.mean(final_v_mag):.1f} m/s")
        print()
        
        # Check radial velocity components
        # Approximate radial velocity (positive = outward)
        initial_vr = (df['Initial x'] * df['Initial vx'] + df['Initial y'] * df['Initial vy']) / initial_r
        final_vr = (df['Final x'] * df['Final vx'] + df['Final y'] * df['Final vy']) / final_r
        
        print(f"Initial radial velocities:")
        print(f"  Min: {np.min(initial_vr):.1f} m/s")
        print(f"  Max: {np.max(initial_vr):.1f} m/s")
        print(f"  Mean: {np.mean(initial_vr):.1f} m/s")
        
        print(f"Final radial velocities:")
        print(f"  Min: {np.min(final_vr):.1f} m/s")
        print(f"  Max: {np.max(final_vr):.1f} m/s")
        print(f"  Mean: {np.mean(final_vr):.1f} m/s")
        
        initially_outward = np.sum(initial_vr > 0)
        finally_outward = np.sum(final_vr > 0)
        
        print(f"Initially moving outward: {initially_outward} ({initially_outward/len(df)*100:.1f}%)")
        print(f"Finally moving outward: {finally_outward} ({finally_outward/len(df)*100:.1f}%)")
        print()
        
        # Time to fall analysis
        print("=== TIME ANALYSIS ===")
        spawn_height = np.mean(initial_r) - alpha
        fall_time = np.sqrt(2 * spawn_height / gravity)
        
        print(f"Average spawn height above alpha: {spawn_height:.0f} m")
        print(f"Free fall time to alpha: {fall_time:.0f} s ({fall_time/3600:.2f} hours)")
        print(f"Simulation time: {t_max:.0f} s ({t_max/3600:.1f} hours)")
        
        if t_max > fall_time * 3:  # 3x safety factor
            print("✅ Simulation time should be sufficient for particles to fall")
        else:
            print("⚠️  Simulation time might be too short")
            
        print()
        print("=== CONCLUSIONS ===")
        
        if np.mean(final_r) > np.mean(initial_r):
            print("❌ Particles are moving OUTWARD on average")
            print("This suggests:")
            print("  - Centrifugal forces from rotation are dominant")
            print("  - Initial velocities have net outward component")
            print("  - Gravity is too weak to overcome other forces")
        else:
            print("✅ Particles are moving inward on average")
            
        if np.mean(final_vr) > 0:
            print("❌ Particles have net outward velocity at end")
            print("They are still accelerating away from the ringworld")
        else:
            print("✅ Particles have net inward velocity at end")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No result files found!")

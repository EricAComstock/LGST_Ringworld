#!/usr/bin/env python3
"""
Analyze the simulation results to understand why we're getting unusual classifications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        
        # Show column names
        print("Available columns:")
        for col in df.columns:
            print(f"  {col}")
        print()
        
        # Basic statistics
        print("=== CLASSIFICATION ANALYSIS ===")
        if 'result' in df.columns:
            result_counts = df['result'].value_counts()
            total = len(df)
            
            for result, count in result_counts.items():
                percentage = count / total * 100
                print(f"{result}: {count} ({percentage:.1f}%)")
            print()
        
        # Analyze final positions
        if all(col in df.columns for col in ['final_x', 'final_y', 'final_z']):
            print("=== FINAL POSITION ANALYSIS ===")
            
            # Calculate final radial distances and z positions
            final_r = np.sqrt(df['final_x']**2 + df['final_y']**2)
            final_z_abs = np.abs(df['final_z'])
            
            # Boundary conditions (should match the corrected values)
            y_floor = 149597870691  # 1 AU
            alpha = y_floor - (218 * 1000)  # Atmosphere boundary
            ringworld_width_km = 81938128337
            beta = (ringworld_width_km * 1000) / 2  # Lateral boundary
            
            print(f"Boundary conditions:")
            print(f"  Alpha (atmosphere): {alpha:.2e} m")
            print(f"  Beta (lateral): {beta:.2e} m ({beta/1000:.2e} km)")
            print(f"  Y floor: {y_floor:.2e} m")
            print()
            
            # Analyze where particles end up
            above_alpha = final_r > alpha
            below_alpha = final_r < alpha
            inside_beta = final_z_abs <= beta
            outside_beta = final_z_abs > beta
            
            print(f"Final positions:")
            print(f"  Above alpha (atmosphere): {np.sum(above_alpha)} ({np.sum(above_alpha)/len(df)*100:.1f}%)")
            print(f"  Below alpha (atmosphere): {np.sum(below_alpha)} ({np.sum(below_alpha)/len(df)*100:.1f}%)")
            print(f"  Inside beta (lateral): {np.sum(inside_beta)} ({np.sum(inside_beta)/len(df)*100:.1f}%)")
            print(f"  Outside beta (lateral): {np.sum(outside_beta)} ({np.sum(outside_beta)/len(df)*100:.1f}%)")
            print()
            
            # Expected classifications based on final positions
            print("Expected classifications based on final positions:")
            
            # Inside beta and above alpha -> resimulate
            expected_resim = np.sum(inside_beta & above_alpha)
            print(f"  Should be resimulate (inside beta, above alpha): {expected_resim}")
            
            # Inside beta and below alpha -> recaptured  
            expected_recap = np.sum(inside_beta & below_alpha)
            print(f"  Should be recaptured (inside beta, below alpha): {expected_recap}")
            
            # Outside beta -> escaped
            expected_escape = np.sum(outside_beta)
            print(f"  Should be escaped (outside beta): {expected_escape}")
            print()
            
            # Check why no recapture
            if expected_recap == 0:
                print("❌ NO PARTICLES END BELOW ALPHA!")
                print("This explains why recaptured = 0%")
                print()
                print("Possible reasons:")
                print("1. Initial velocities are too high")
                print("2. Gravity is too weak")
                print("3. Simulation time is too short")
                print("4. Particles are spawned with too much energy")
                print()
                
                # Show radial distance statistics
                print(f"Final radial distances:")
                print(f"  Min: {np.min(final_r):.2e} m")
                print(f"  Max: {np.max(final_r):.2e} m") 
                print(f"  Mean: {np.mean(final_r):.2e} m")
                print(f"  Alpha boundary: {alpha:.2e} m")
                print(f"  Distance above alpha: {np.min(final_r) - alpha:.2e} m")
                
        # Analyze beta crossings if available
        if 'beta_crossings' in df.columns:
            print()
            print("=== BETA CROSSINGS ANALYSIS ===")
            crossings = df['beta_crossings']
            print(f"Beta crossings - Min: {np.min(crossings)}, Max: {np.max(crossings)}, Mean: {np.mean(crossings):.1f}")
            
            # Show distribution
            crossing_counts = crossings.value_counts().sort_index()
            for crossings_val, count in crossing_counts.items():
                print(f"  {crossings_val} crossings: {count} particles")
                
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("No result files found!")

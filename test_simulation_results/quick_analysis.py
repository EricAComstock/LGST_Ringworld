#!/usr/bin/env python3
"""
Quick analysis of the simulation results.
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
        
        # Show first few rows to understand the data structure
        print("First few rows:")
        print(df.head())
        print()
        
        # Check column names more carefully
        print("Column names (exact):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: '{col}'")
        print()
        
        # Try to find result column with different possible names
        result_col = None
        for col in df.columns:
            if 'result' in col.lower() or 'classification' in col.lower():
                result_col = col
                break
        
        if result_col:
            print(f"Found result column: '{result_col}'")
            result_counts = df[result_col].value_counts()
            total = len(df)
            
            print("Classification results:")
            for result, count in result_counts.items():
                percentage = count / total * 100
                print(f"  {result}: {count} ({percentage:.1f}%)")
        else:
            print("No result column found!")
            
        # Check final positions
        final_cols = [col for col in df.columns if 'final' in col.lower()]
        print(f"\nFinal position columns: {final_cols}")
        
        if len(final_cols) >= 3:
            # Try to identify x, y, z columns
            final_x_col = next((col for col in final_cols if 'x' in col.lower()), None)
            final_y_col = next((col for col in final_cols if 'y' in col.lower()), None) 
            final_z_col = next((col for col in final_cols if 'z' in col.lower()), None)
            
            if all([final_x_col, final_y_col, final_z_col]):
                print(f"Using: {final_x_col}, {final_y_col}, {final_z_col}")
                
                final_r = np.sqrt(df[final_x_col]**2 + df[final_y_col]**2)
                final_z_abs = np.abs(df[final_z_col])
                
                # Boundary conditions
                y_floor = 149597870691  # 1 AU
                alpha = y_floor - (218 * 1000)  # Atmosphere boundary
                
                print(f"\nFinal radial distances:")
                print(f"  Min: {np.min(final_r):.2e} m")
                print(f"  Max: {np.max(final_r):.2e} m")
                print(f"  Mean: {np.mean(final_r):.2e} m")
                print(f"  Alpha boundary: {alpha:.2e} m")
                
                above_alpha = np.sum(final_r > alpha)
                below_alpha = np.sum(final_r <= alpha)
                
                print(f"\nParticles above alpha: {above_alpha} ({above_alpha/len(df)*100:.1f}%)")
                print(f"Particles below alpha: {below_alpha} ({below_alpha/len(df)*100:.1f}%)")
                
                if below_alpha == 0:
                    print("\n❌ NO PARTICLES FALL BELOW ATMOSPHERE!")
                    print("This is why recaptured = 0%")
                    print("All particles have too much energy to be recaptured.")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No result files found!")

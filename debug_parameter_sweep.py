#!/usr/bin/env python3
"""
Debug script to test parameter sweep issues.
"""

from StochasticInputRK45Solver import main as run_simulation
import traceback

def test_simulation():
    """Test the simulation with simple parameters."""
    
    print("Testing simulation with minimal parameters...")
    
    # Use very simple parameters first
    sim_params = {
        'num_particles': 5,  # Very small number for quick testing
        'save_results': False,
        'show_plots': False
    }
    
    try:
        print("Calling run_simulation with sim_params:", sim_params)
        results = run_simulation(sim_params=sim_params)
        
        print(f"Results type: {type(results)}")
        if results is not None:
            print(f"Results shape: {results.shape}")
            print(f"Results columns: {list(results.columns)}")
            print("First few rows:")
            print(results.head())
        else:
            print("Results is None!")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

def test_with_parameter_sweep_params():
    """Test with the same parameters that parameter sweep uses."""
    
    print("\n" + "="*50)
    print("Testing with parameter sweep parameters...")
    
    # These are the default parameters from parameter_sweep.py
    sim_params = {
        'radius': 8.19381e+12 * 1000,  # Convert km to meters
        'gravity': 2.743176313,  # m/s²
        't_max': 1e6,
        'dt': 100,
        'is_rotating': True,
        'num_particles': 5,  # Small number for testing
        'save_results': False,
        'show_plots': False,
        'find_leak_rate': True,
        'temperature': 289,  # K
        'y_min': 149597870691 + 218 * 1000,  # Minimum spawn altitude (m)
        'y_max': 149597870691 + 218 * 1000 + 10 * 1000,  # Maximum spawn altitude (m)
        'z_length': 81938128337 * 1000,  # Convert width to meters
        'y_floor': 149597870691,  # Ringworld floor (1 AU)
        'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # diatomic oxygen
    }
    
    try:
        print("Calling run_simulation with full sim_params...")
        results = run_simulation(sim_params=sim_params)
        
        print(f"Results type: {type(results)}")
        if results is not None:
            print(f"Results shape: {results.shape}")
            print(f"Results columns: {list(results.columns)}")
            print("First few rows:")
            print(results.head())
        else:
            print("Results is None!")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_simulation()
    test_with_parameter_sweep_params()

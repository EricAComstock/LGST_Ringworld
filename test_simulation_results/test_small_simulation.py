#!/usr/bin/env python3
"""
Test script to run a small simulation and check results.
"""

from run_simulation import run_simulation_with_params
import sys

# Temporarily modify the run_simulation function to use fewer particles for testing
def test_simulation():
    from StochasticInputRK45Solver_Siyona import main as run_simulation
    
    # Parameters from run_simulation.py but with fewer particles
    ringworld_radius_km = 8.19381e+12  # km
    ringworld_width_km = 81938128337  # km
    gravity = 2.743176313  # m/s²
    
    sim_params = {
        'radius': ringworld_radius_km,
        'gravity': gravity,
        't_max': 1e6,
        'dt': 100,
        'is_rotating': True,
        'num_particles': 50,  # Small test
        'save_results': False,  # Don't save for test
        'show_plots': False,
        'find_leak_rate': True,
        'temperature': 289,
        'y_min': 149597870691 + 218 * 1000,
        'y_max': 149597870691 + 218 * 1000 + 10 * 1000,
        'z_length': ringworld_width_km * 1000,
        'y_floor': 149597870691,
        'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]
    }
    
    print("Running test simulation with 50 particles...")
    print("This will test if the boundary fix resolved the classification issues.")
    print()
    
    try:
        results = run_simulation(sim_params=sim_params)
        
        if results is not None and not results.empty:
            # Count classifications
            escaped = len(results[results['result'] == 'escaped'])
            recaptured = len(results[results['result'] == 'recaptured'])
            resimulate = len(results[results['result'] == 'resimulate'])
            total = len(results)
            
            print("=== TEST SIMULATION RESULTS ===")
            print(f"Total particles: {total}")
            print(f"Escaped: {escaped} ({escaped/total*100:.1f}%)")
            print(f"Recaptured: {recaptured} ({recaptured/total*100:.1f}%)")
            print(f"Need resimulation: {resimulate} ({resimulate/total*100:.1f}%)")
            print()
            
            # Check if results are more reasonable
            if recaptured > 0:
                print("✅ Good! Now seeing recaptured particles (was 0% before)")
            else:
                print("⚠️  Still seeing 0% recaptured particles")
                
            if resimulate < total * 0.5:  # Less than 50%
                print("✅ Good! Resimulation percentage is more reasonable")
            else:
                print("⚠️  Still seeing high resimulation percentage")
                
            # Show some beta crossings statistics
            if 'beta_crossings' in results.columns:
                avg_crossings = results['beta_crossings'].mean()
                max_crossings = results['beta_crossings'].max()
                print(f"Beta crossings - Average: {avg_crossings:.1f}, Max: {max_crossings}")
                
        else:
            print("❌ No results returned from simulation")
            
    except Exception as e:
        print(f"❌ Error during test simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simulation()

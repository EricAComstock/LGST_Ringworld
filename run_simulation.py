# run_simulation.py
from StochasticInputRK45Solver import main as run_simulation
from StochasticInput import SIVarInput
import numpy as np

# Set ringworld parameters
ringworld_radius_km = 8.19381e+12  # km
ringworld_width_km = 81938128337  # km (set your desired width here)
gravity = 9.81  # m/s² (Earth-like gravity for testing)

def run_simulation_with_params():
    # Set simulation parameters
    sim_params = {
        'radius': ringworld_radius_km,
        'gravity': 50.0,  # Very high gravity for testing
        't_max': 1e6,
        'dt': 100,
        'is_rotating': False,  # Test without rotation
        'num_particles': 50,  # Small test to verify fix
        'save_results': True,
        'show_plots': False,
        'find_leak_rate': True,
        'temperature': 100,  # Much colder temperature
        'y_min': 149597870691 + 218 * 1000,  # Minimum spawn altitude (m)
        'y_max': 149597870691 + 218 * 1000 + 10 * 1000,  # Maximum spawn altitude (m)
        'z_length': ringworld_width_km * 1000,  # Convert width to meters
        'y_floor': 149597870691,  # Ringworld floor (1 AU)
        'comp_list': [
            ("O2", 2.6566962e-26 * 2, 0, 100)  # diatomic oxygen
        ]
    }
    
    # Print simulation parameters
    print("Starting particle simulation with parameters:")
    for key, value in sim_params.items():
        if key not in ['y_min', 'y_max', 'y_floor']:  # Skip printing large numbers for better readability
            print(f"  {key}: {value}")
    print(f"  ringworld_width: {ringworld_width_km} km")
    
    # Extract only the parameters that the main function accepts as direct arguments
    direct_params = {}
    main_function_args = ['radius', 'gravity', 't_max', 'dt', 'is_rotating', 
                         'num_particles', 'save_results', 'show_plots', 
                         'find_leak_rate', 'comp_list']
    
    for param in main_function_args:
        if param in sim_params:
            direct_params[param] = sim_params[param]
    
    # Run the simulation with direct parameters and full sim_params
    return run_simulation(**direct_params, sim_params=sim_params)

if __name__ == "__main__":
    try:
        results = run_simulation_with_params()
        print("\nSimulation complete!")
        print(f"Particles recaptured: {results.get('recaptured', 'N/A')}")
        print(f"Particles escaped: {results.get('escaped', 'N/A')}")
        escape_frac = results.get('escape_fraction')
        print(f"Escape fraction: {escape_frac*100:.4f}%" if escape_frac is not None else "Escape fraction: N/A")
    except Exception as e:
        print(f"\nSimulation failed with error: {str(e)}")
        print("Make sure all required modules are imported and parameters are valid.")
        import traceback
        traceback.print_exc()
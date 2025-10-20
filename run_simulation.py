# run_simulation.py
from StochasticInputRK45Solver import main as run_simulation
from StochasticInput import SIVarInput
from TrajectoryClassification import TCVarInput
from SolverSharedCodePlusSolar_Optimized import SSCPSVarInput
from LeakRate import LRVarInput
import numpy as np

# Set ringworld parameters
ringworld_radius_km = 1000.0  # km
ringworld_width_km = 200.0  # km (set your desired width here)
gravity = 9.81  # m/s² (Earth-like gravity for testing)

def run_simulation_with_params():
    # Set simulation parameters
    sim_params = {
        'radius': ringworld_radius_km,
        'gravity': gravity,  # Very high gravity for testing
        't_max': 1e6,
        'dt': 100,
        'is_rotating': False,  # Test without rotation
        'num_particles': 500,  # Small test to verify fix
        'save_results': True,
        'show_plots': False,
        'find_leak_rate': True,
        'temperature': 100,  # Much colder temperature
        'y_min': 149597870691 - 218 * 1000 - 10 * 1000,  # Minimum spawn altitude (m)
        'y_max': 149597870691 - 218 * 1000,  # Maximum spawn altitude (m)
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
    
    # Initialize all modules with the required parameters
    # Physics constants
    G = 6.67430e-11  # Gravitational constant [m³/(kg·s²)]
    
    # Leak rate parameters (using defaults from StochasticInputRK45Solver.py)
    P_0 = 101325  # Pressure at surface [Pa]
    K_b = 1.380649e-23  # Boltzmann constant [J/K]
    m = 2.6566962e-26 * 2  # Molecular mass (O2) [kg]
    n_0 = 100  # Number density at surface [particles/m³]
    d = 3.59e-10  # Molecular diameter [m]
    
    # Calculate trajectory classification parameters
    alpha = sim_params['y_floor'] - (218 * 1000)  # Atmosphere boundary [m]
    beta = sim_params['z_length'] / 2  # Lateral boundary [m]
    
    # Initialize all modules with parameters
    SSCPSVarInput(G)
    SIVarInput(sim_params['temperature'], sim_params['y_min'], sim_params['y_max'], 
               sim_params['z_length'], sim_params['y_floor'])
    TCVarInput(sim_params['z_length'], beta, sim_params['y_floor'], alpha, 
               sim_params['y_min'], sim_params['y_max'])
    LRVarInput(P_0, K_b, sim_params['temperature'], m, sim_params['gravity'], n_0, d)
    
    # Extract only the parameters that the main function accepts as direct arguments
    direct_params = {}
    main_function_args = ['radius', 'gravity', 't_max', 'dt', 'is_rotating', 
                         'num_particles', 'save_results', 'show_plots', 
                         'find_leak_rate', 'comp_list']
    
    for param in main_function_args:
        if param in sim_params:
            direct_params[param] = sim_params[param]
    
    # Run the simulation with direct parameters only
    return run_simulation(**direct_params)

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
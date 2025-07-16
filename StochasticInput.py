"""
StochasticInput.py

Generates randomized initial particle positions and velocities based on a
Maxwell-Boltzmann distribution for thermal simulations.

Version: 1.0
Author: Nick Gaug
Date: April 29, 2025
"""

import numpy as np
from scipy.stats import maxwell
import random

# Constants
T        = 289                       # Temperature (K)
m        = 5.31e-26                  # Mass of oxygen molecule (kg)
y_min    = 149597870691 + 218 * 1000 # Minimum y (meters)
y_max    = 149597870691 + 218 * 1000 + 10 * 1000  # Maximum y (meters)
z_length = 10000 * 1000            # Total z-length (meters)
y_floor  = 149597870691              # Floor value for y (meters)

def SIVarInput(T_i,m_i,y_min_i,y_max_i,z_length_i,y_floor_i):
    global T,m,y_min,y_max,z_length,y_floor
    T = T_i
    m = m_i
    y_min = y_min_i
    y_max = y_max_i
    z_length = z_length_i
    y_floor = y_floor_i

    

def stochastic_initial_conditions(T, y_min, y_max, z_length, comp_list = None):
    """
    Generates stochastic initial conditions for particle position and velocity.

    Parameters:
    T (float): Temperature (K)
    y_min (int): Minimum y-coordinate for particle spawning (m)
    y_max (int): Maximum y-coordinate for particle spawning (m)
    z_length (int): Total z-length of the domain (m)
    comp_list (list of tuples (string, float, float, float)): List of all species at a given altitude. 
        Individual species tuples contain species name, mass, charge, and density (no units, kg, C, particles/m^3)

    Returns:
    list: [x, y, z, vx, vy, vz, m, q] initial position and velocity components, and particle mass and charge
    """
    z_min = -int(z_length / 2)
    z_max = int(z_length / 2)
    z     = random.randint(z_min, z_max)   # Generate random z distance from left wall
    y     = random.randint(y_min, y_max)   # Generate random y altitude
    x     = 0                              # Start at x = 0

    total_density = 0
    for particle in comp_list: total_density += particle[3]
    weights = []
    for particle in comp_list: weights.append(particle[3]/total_density)
    ind = np.random.choice(a=range(len(comp_list)),size=1,p=weights)
    selected_particle = comp_list[ind[0]]
    m = selected_particle[1]
    q = selected_particle[2]

    random_vector = np.random.normal(size=3)                       # Generate random 3D vector
    unit_vector   = random_vector / np.linalg.norm(random_vector)  # Normalize to unit length

    k_B               = 1.380649e-23                      # Boltzmann constant (J/K)
    scale             = np.sqrt(k_B * T / m)              # Maxwell-Boltzmann scale parameter
    velocity_magnitude = maxwell.rvs(scale=scale)         # Random velocity magnitude

    [vx, vy, vz] = [float(v) for v in velocity_magnitude * unit_vector]  # Scale unit vector by magnitude

    return [x, y, z, vx, vy, vz, m, q]



# Testing code - only runs when this file is executed directly
if __name__ == "__main__":
    input_test = stochastic_initial_conditions(T, y_min, y_max, z_length, [("X",1,-1,6), ("Y",2,1,1)])
    print(input_test)
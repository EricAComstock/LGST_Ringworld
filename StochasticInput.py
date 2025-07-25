"""
StochasticInput.py
Generates randomized initial particle positions and velocities based on a
Maxwell-Boltzmann distribution for atmospheric thermal simulations.

V1.0, Nick Gaug, April 29, 2025
"""

import numpy as np
from scipy.stats import maxwell
import random

# Default parameters
T = 289  # Temperature [K]
m = 5.31e-26  # Mass of oxygen molecule [kg]
y_min = 149597870691 + 218 * 1000  # Minimum spawn altitude [m]
y_max = 149597870691 + 218 * 1000 + 10 * 1000  # Maximum spawn altitude [m]
z_length = 10000 * 1000  # Total z-length [m]
y_floor = 149597870691  # Ringworld floor (1 AU) [m]


def SIVarInput(T_i, m_i, y_min_i, y_max_i, z_length_i, y_floor_i):
    """
    Set global parameters for stochastic input generation.
    Called by StochasticInputRK45Solver.py to pass simulation parameters.

    SIVarInput(T_i, m_i, y_min_i, y_max_i, z_length_i, y_floor_i)

    Inputs:
    T_i         Temperature [K]
    m_i         Particle mass [kg]
    y_min_i     Minimum y-coordinate for particle spawning [m]
    y_max_i     Maximum y-coordinate for particle spawning [m]
    z_length_i  Total z-dimension length [m]
    y_floor_i   Floor value for y-coordinate (ringworld surface) [m]

    Outputs:
    None (sets global variables)
    """
    global T, m, y_min, y_max, z_length, y_floor
    T = T_i  # Temperature
    m = m_i  # Particle mass
    y_min = y_min_i  # Minimum spawn altitude
    y_max = y_max_i  # Maximum spawn altitude
    z_length = z_length_i  # Ringworld width
    y_floor = y_floor_i  # Ringworld floor


def stochastic_initial_conditions(m, T, y_min, y_max, z_length):
    """
    Generates stochastic initial conditions for particle position and velocity.
    Velocity sampled from Maxwell-Boltzmann distribution at temperature T.

    [x, y, z, vx, vy, vz] = stochastic_initial_conditions(m, T, y_min, y_max, z_length)

    Inputs:
    m         Particle mass [kg]
    T         Temperature [K]
    y_min     Minimum y-coordinate for particle spawning [m]
    y_max     Maximum y-coordinate for particle spawning [m]
    z_length  Total z-length of the domain [m]

    Outputs:
    [x, y, z, vx, vy, vz]  Initial position and velocity components [m, m/s]
    """
    # Generate random position
    z_min = -int(z_length / 2)  # Left boundary
    z_max = int(z_length / 2)  # Right boundary
    z = random.randint(z_min, z_max)  # Random z position
    y = random.randint(y_min, y_max)  # Random altitude
    x = 0  # Start at x = 0

    # Generate random velocity direction
    random_vector = np.random.normal(size=3)  # Random 3D vector
    unit_vector = random_vector / np.linalg.norm(random_vector)  # Normalize to unit

    # Sample velocity magnitude from Maxwell-Boltzmann distribution
    k_B = 1.380649e-23  # Boltzmann constant [J/K]
    scale = np.sqrt(k_B * T / m)  # MB scale parameter
    velocity_magnitude = maxwell.rvs(scale=scale)  # Random velocity magnitude

    # Scale unit vector by magnitude
    [vx, vy, vz] = [float(v) for v in velocity_magnitude * unit_vector]

    return [x, y, z, vx, vy, vz]


# Testing code - only runs when this file is executed directly
if __name__ == "__main__":
    input_test = stochastic_initial_conditions(m, T, y_min, y_max, z_length)
    print(input_test)
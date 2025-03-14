import numpy as np
from scipy.stats import maxwell
import random

def stochastic_initial_conditions(m,T,y_min,y_max,z_length):
    z_min = - int(z_length/2)
    z_max = int(z_length/2)
    z =  random.randint(z_min, z_max) #generate random distance from left wall
    y = random.randint(y_min, y_max) #generate random altitude
    x = 0

    random_vector = np.random.normal(size=3)  #generate a random vector from a normal distribution
    unit_vector = random_vector / np.linalg.norm(random_vector)  #normalize to make it a unit vector

    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    scale = np.sqrt(k_B * T / m) # Calculate the scale parameter for the Maxwell-Boltzmann distribution
    velocity_magnitude = maxwell.rvs(scale=scale)  # Generate a single random velocity

    [vx, vy, vz] = [float(v) for v in velocity_magnitude * unit_vector] #multiply magnitude by unit vector

    return [x,y,z,vx,vy,vz]

# Constants
T = 289  # Temperature in Kelvin
m = 5.31e-26  # Mass of oxygen molecule (kg)
y_min = 149597870691 + 218*1000 #meters, top of atmosphere based on exponential scale height
y_max = 149597870691 + 218*1000 + 10*1000 #meters, top of atmoshere plus arbitrary 10 km spawning cieling
z_length = 1600000*1000 #meters
y_floor = 149597870691

# Testing code - only runs when this file is executed directly
if __name__ == "__main__":
    input_test = stochastic_initial_conditions(m,T,y_min,y_max,z_length)
    print(input_test)

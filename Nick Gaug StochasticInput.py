import numpy as np
from scipy.stats import maxwell
import random

def stochastic_initial_conditions(m,T,alt_min,alt_max,y_min,y_max):
    y =  random.randint(y_min, y_max) #generate random distance from left wall
    z = random.randint(alt_min, alt_max) #generate random altitude
    x = 0

    random_vector = np.random.normal(size=3)  #generate a random vector from a normal distribution
    unit_vector = random_vector / np.linalg.norm(random_vector)  #normalize to make it a unit vector

    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    scale = np.sqrt(k_B * T / m) # Calculate the scale parameter for the Maxwell-Boltzmann distribution
    velocity_magnitude = maxwell.rvs(scale=scale)  # Generate a single random velocity

    [vx, vy, vz] = [float(v) for v in velocity_magnitude * unit_vector] #multiply magnitude by unit vector

    return [x,y,z,vx,vy,vz]

#Below this is to test functionality

T = 289  # Temperature in Kelvin
m = 5.31e-26  # Mass of oxygem molecule (kg)
alt_min = 149597870691 #meters
alt_max = 149597870691 + 100000 #meters
y_min = 0 #meters
y_max = 1600000*1000 #meters

input_test = stochastic_initial_conditions(m,T,alt_min,alt_max,y_min,y_max)
print(input_test)

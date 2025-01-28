import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Constants to be altered as necessary for different particles and environments 
q = 1.0  # Charge of the particle (Coulombs) (adjust to positive or negative for protons or electrons)
m = 0.01  # Mass of the particle (kg) (adjsut here)
G = 6.67430e-11  # Gravitational constant (m^3 kg^−1 s^−2) (dont change lol)
M = 5.972e24  # Mass of Earth (kg) 
R = 6.371e6  # Radius of Earth (m)
omega = 7.2921159e-5  # Angular velocity of Earth's rotation (rad/s)

# Time parameters
dt = 0.01  # Time step (seconds)
num_steps = 10000  # Number of simulation steps

# Defining the function for time-varying electric field 
def electric_field(t):
    # E-field currently oscillates in the y-direction with a frequency of 0.1 Hz (this can be adjusted to represent different E-Fields)
    return np.array([0.0, np.sin(2 * np.pi * 0.1 * t), 0.0])

# Defining the function for time-varying magnetic field 
def magnetic_field(t):
    # B-field currently oscillates in the z-direction with a frequency of 0.1 Hz (this can be adjusted to represent differnt B-Fields)
    return np.array([0.0, 0.0, np.cos(2 * np.pi * 0.1 * t)])

# Defining the function to compute the Lorentz force (don't adjust this as its just a formula lol)
def lorentz_force(q, E, B, v):
    return q * (E + np.cross(v, B))

# Defining the function to compute the gravitational force (dont adjust this formula too lol)
def gravitational_force(M, m, r):
    distance = np.linalg.norm(r) #calculates distance from particle as and stores as value r
    return -G * M * m * r / (distance**3)

# Defining the function to compute the centrifugal force (also dont adjust here)
def centrifugal_force(m, omega, r):
    return m * np.cross(omega, np.cross(omega, r))

# Function to compute the derivatives which will be used later in the Runge-Kutta operation (dont adjust any of this as its a set function)
def derivatives(t, y, E, B, M, m, omega):
    position = y[:3]
    velocity = y[3:]
    
    # Computes the forces based on predefined functions
    F_lorentz = lorentz_force(q, E, B, velocity)
    F_gravity = gravitational_force(M, m, position)
    F_centrifugal = centrifugal_force(m, np.array([0, 0, omega]), position)
    
    # Calculates the derivatives 
    dydt = np.zeros_like(y)
    dydt[:3] = velocity  
    dydt[3:] = (F_lorentz + F_gravity + F_centrifugal) / m  
    
    return dydt

# The first array sets the initial position for 5 3D particles as a random multiplier between 0 and 1.5 of Earths radius, can be adjusted as needed
# The second array sets the initial velocities for the 5 3D particles, can be adjusted to more accurately reflect the typical particle velocity in the upper atmosphere
initial_conditions = [
    [np.array([np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R]), np.array([0.0, 1.0, 0.0])],  # Particle 1
    [np.array([np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R]), np.array([0.0, 1.5, 0.0])],  # Particle 2
    [np.array([np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R]), np.array([0.0, 1.2, 0.0])],  # etc
    [np.array([np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R]), np.array([0.0, 1.0, 0.0])],  
    [np.array([np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R, np.random.uniform(0,1.5) * R]), np.array([0.0, 1.0, 0.0])]  
]
num_particles = len(initial_conditions)
start_conditions = initial_conditions[:] #holds initial values to be added to spreadsheet


# Arrays that store position and velocity data for plotting
all_positions = [[] for i in range(num_particles)]

# This is a time integration loop based on the Runge-Kutta 4th order method to update the positions and velocities of the particles (dont adjust this as it is a set operation) 
for step in range(num_steps):
    t = step * dt
    
    # Stores the current positions for all particles
    for i in range(num_particles):
        position, velocity = initial_conditions[i]
        all_positions[i].append(position.copy())

    # Runge-Kutta method applied for each particle
    for i in range(num_particles):
        position, velocity = initial_conditions[i]
        y = np.concatenate((position, velocity))  # Concatenate position and velocity into a single array

        # Creates fields that vary based on time
        E = electric_field(t)
        B = magnetic_field(t)

        # Runge-Kutta method 4 intermediate slopes used to accurately update the state of the particle
        k1 = derivatives(t, y, E, B, M, m, omega)
        k2 = derivatives(t, y + 0.5 * dt * k1, E, B, M, m, omega)
        k3 = derivatives(t, y + 0.5 * dt * k2, E, B, M, m, omega)
        k4 = derivatives(t, y + dt * k3, E, B, M, m, omega)

        # Update the state of particle using the 4 k steps found
        y_next = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Extract new position and velocity
        new_position = y_next[:3]
        new_velocity = y_next[3:]
        
        # Update the particle's position and velocity
        initial_conditions[i] = [new_position, new_velocity]

# will create a spreadsheet that has the particles initial and final position and velocity
def get_data(particle_num):
    return [start_conditions[i][0], initial_conditions[i][0], start_conditions[i][1], initial_conditions[i][1]]

particle_data = [["Inital Position (m)", "Final Position (m)", "Initial Velocity (m/s)", "Final Velocity (m/s)"]] # Initial Header
for i in range(num_particles):
    temp = get_data(i)
    particle_data.append(temp)

# Write to CSV file
with open("C:/Users/Matthew/Desktop/4699 Code/output2.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(particle_data)

print("done")
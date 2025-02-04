import numpy as np
from scipy.stats import maxwell
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from StochasticInput import stochastic_initial_conditions

# Constants
T = 289  # Temperature in Kelvin
m = 5.31e-26  # Mass of oxygem molecule (kg)
y_min = 149597870691 #meters
y_max = 149597870691 + 100000 #meters
z_length = 1600000*1000 #meters

runs = 10000
results = np.empty((1,6))
for i in range(runs):
    result = stochastic_initial_conditions(m,T,y_min,y_max,z_length)
    results = np.vstack((results,result))

def check_distro(results):

    x = results[1:, 0]
    y = results[1:, 1]
    z = results[1:, 2]
    vx = results[1:, 3]
    vy = results[1:, 4]
    vz = results[1:, 5]

    # Create a histogram of the velocities

    plt.figure(figsize=(8, 6))
    plt.hist(vx, bins=60, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("x velocity distribution", fontsize=14)
    plt.xlabel("Velocity (m/s)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(vy, bins=60, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("y velocity distribution", fontsize=14)
    plt.xlabel("Velocity (m/s)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(vz, bins=60, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("z velocity distribution", fontsize=14)
    plt.xlabel("Velocity (m/s)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    mag = np.sqrt(vx**2 + vy**2 + vz**2)

    plt.figure(figsize=(8, 6))
    plt.hist(mag, bins=60, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("velocity magnitude distribution", fontsize=14)
    plt.xlabel("Velocity (m/s)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(x, bins=55, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("x position distribution", fontsize=14)
    plt.xlabel("Position (m)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=60, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("y position distribution", fontsize=14)
    plt.xlabel("Position (m)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(z, bins=60, density=True, alpha=0.7, color='blue', label="Velocity Distribution")
    plt.title("z position distribution", fontsize=14)
    plt.xlabel("Position (m)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

check_distro(results)
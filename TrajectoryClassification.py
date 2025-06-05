"""
TrajectoryClassification.py

Defines boundary conditions and logic to classify particle trajectories as
escaped, recaptured, or needing resimulation based on their path.

Version: 1.0
Author: Nick Gaug
Date: April 29, 2025
"""


import numpy as np
import pandas as pd

# Boundary conditions - change these values to modify simulation boundaries
z_length = 10000 * 1000  # meters
beta     = z_length / 2
y_floor  = 149597870691
alpha    = y_floor - (218 * 1000)
y_min    = alpha - 10000
y_max    = alpha  # meters

# A particle escapes if it travels below alpha while outside of beta or ends the simulation outside of beta
# A particle is recaptured if it ends the simulation below alpha and within beta or if it collides with beta below alpha
# A particle is resimulated if it ends the simulation above alpha but within beta (it may leave beta and then reenter)
def TCVarInput(z_length_i,beta_i,y_floor_i,alpha_i,y_min_i,y_max_i):
    global z_length,beta,y_floor,alpha,y_min,y_max
    z_length = z_length_i
    beta = beta_i
    y_floor = y_floor_i
    alpha = alpha_i
    y_min = y_min_i
    y_max = y_max_i


def classify_trajectory(alpha, beta, y_floor, trajectories):
    """
    Classifies a particle trajectory based on its path and final position.

    Parameters:
    alpha (float): Threshold radius value (typically y_min + atmosphere height)
    beta (float): Threshold z value (typically half of z_length)
    y_floor (float): Minimum y value (floor of the simulation area)
    trajectories (pd.DataFrame): DataFrame containing trajectory coordinates [x, y, z, ...] in columns 0, 1, 2

    Returns:
    tuple: (beta_crossings, result) where:
        - beta_crossings (int): Count of how many times the particle crossed the beta boundary
        - result (str): Classification as 'recaptured', 'escaped', or 'resimulate'
    """
    recaptured     = False
    escaped        = False
    resimulate     = False
    beta_crossings = 0

    for i in range(1, len(trajectories)):
        # Current position
        x  = trajectories.iloc[i, 0]
        y  = trajectories.iloc[i, 1]
        z  = trajectories.iloc[i, 2]
        r  = np.sqrt(x ** 2 + y ** 2)

        # Previous position
        x_prev = trajectories.iloc[i - 1, 0]
        y_prev = trajectories.iloc[i - 1, 1]
        z_prev = trajectories.iloc[i - 1, 2]
        r_prev = np.sqrt(x_prev ** 2 + y_prev ** 2)

        # Check if particle has entered the side of ringworld
        if abs(z) >= beta and r > alpha and not recaptured and not escaped:
            if abs(z_prev) < beta:  # If this is true, it hit the side
                recaptured = True
            else:  # Otherwise, it came from above
                escaped = True

        # Check if we hit the bottom of the ringworld
        if r < y_floor and r_prev > y_floor and abs(z) <= beta:
            escaped = True

        # Log beta crossings by seeing if the particle crossed beta between two timesteps
        if abs(z) >= beta and abs(z_prev) < beta:
            beta_crossings += 1
        if abs(z) <= beta and abs(z_prev) > beta:
            beta_crossings += 1

    # Account for particles that did not hit one of the initial ending conditions
    if not recaptured and not escaped:
        if abs(z) <= beta and r < alpha:
            resimulate = True
        elif abs(z) <= beta and r > alpha:
            recaptured = True
        else:
            escaped = True

    # Determine final result
    if recaptured:
        result = 'recaptured'
    elif escaped:
        result = 'escaped'
    else:
        result = 'resimulate'

    return beta_crossings, result

# Testing code - only runs when this file is executed directly
if __name__ == "__main__":
    try:
        # Try to load test data
        trajectories = pd.read_excel('test_trajectory.xlsx')
        beta_crossings, result = classify_trajectory(alpha, beta, y_floor, trajectories)
        print(f"Test trajectory classification: {result} with {beta_crossings} beta crossings")
    except Exception as e:
        print(f"Error running test: {e}")
        print("This is normal if test_trajectory.xlsx doesn't exist")

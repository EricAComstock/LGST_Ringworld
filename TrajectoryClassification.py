"""
TrajectoryClassification.py
Defines boundary conditions and classifies particle trajectories as
escaped, recaptured, or needing resimulation based on their path.

V1.0, Nick Gaug, April 29, 2025
"""

import numpy as np
import pandas as pd

# Default boundary conditions
z_length = 10000 * 1000  # Total z-length [m]
beta = z_length / 2  # Lateral boundary [m]
y_floor = 149597870691  # Ringworld floor [m]
alpha = y_floor - (218 * 1000)  # Atmosphere boundary [m]
y_min = alpha - 10000  # Minimum spawn height [m]
y_max = alpha  # Maximum spawn height [m]


def TCVarInput(z_length_i, beta_i, y_floor_i, alpha_i, y_min_i, y_max_i):
    """
    Set global parameters for trajectory classification.
    Called by StochasticInputRK45Solver.py to pass boundary parameters.

    TCVarInput(z_length_i, beta_i, y_floor_i, alpha_i, y_min_i, y_max_i)

    Inputs:
    z_length_i  Total z-dimension length (ringworld width) [m]
    beta_i      Lateral boundary threshold (half of z_length) [m]
    y_floor_i   Minimum y value (ringworld floor) [m]
    alpha_i     Atmosphere boundary threshold [m]
    y_min_i     Minimum particle spawn height [m]
    y_max_i     Maximum particle spawn height [m]

    Outputs:
    None (sets global variables)
    """
    global z_length, beta, y_floor, alpha, y_min, y_max
    z_length = z_length_i  # Ringworld width
    beta = beta_i  # Lateral boundary
    y_floor = y_floor_i  # Floor boundary
    alpha = alpha_i  # Atmosphere boundary
    y_min = y_min_i  # Min spawn height
    y_max = y_max_i  # Max spawn height


def classify_trajectory(alpha, beta, y_floor, trajectories):
    """
    Classifies particle trajectory based on path and final position.
    Determines if particle escaped, was recaptured, or needs resimulation.

    beta_crossings, result = classify_trajectory(alpha, beta, y_floor, trajectories)

    Inputs:
    alpha        Atmosphere boundary radius [m]
    beta         Lateral boundary threshold [m]
    y_floor      Minimum y value (ringworld floor) [m]
    trajectories DataFrame with columns [x, y, z] coordinates

    Outputs:
    beta_crossings  Number of beta boundary crossings [int]
    result          Classification: 'escaped', 'recaptured', or 'resimulate' [string]

    Classification rules:
    - Escaped: Goes below alpha while outside beta, or ends outside beta
    - Recaptured: Ends below alpha within beta, or hits beta while below alpha
    - Resimulate: Ends above alpha within beta (undetermined fate)
    """
    # Initialize classification flags
    recaptured = False
    escaped = False
    resimulate = False
    beta_crossings = 0

    # Analyze trajectory timestep by timestep
    for i in range(1, len(trajectories)):
        # Current position
        x = trajectories.iloc[i, 0]  # x-coordinate
        y = trajectories.iloc[i, 1]  # y-coordinate
        z = trajectories.iloc[i, 2]  # z-coordinate
        r = np.sqrt(x ** 2 + y ** 2)  # Radial distance

        # Previous position
        x_prev = trajectories.iloc[i - 1, 0]  # Previous x
        y_prev = trajectories.iloc[i - 1, 1]  # Previous y
        z_prev = trajectories.iloc[i - 1, 2]  # Previous z
        r_prev = np.sqrt(x_prev ** 2 + y_prev ** 2)  # Previous radius

        # Check if particle entered side of ringworld
        if abs(z) >= beta and r > alpha and not recaptured and not escaped:
            if abs(z_prev) < beta:  # Hit the side boundary
                recaptured = True
            else:  # Came from above atmosphere
                escaped = True

        # Check if hit bottom of ringworld
        if r < y_floor and r_prev > y_floor and abs(z) <= beta:
            escaped = True

        # Count beta boundary crossings
        if abs(z) >= beta and abs(z_prev) < beta:
            beta_crossings += 1
        if abs(z) <= beta and abs(z_prev) > beta:
            beta_crossings += 1

    # Classify particles that didn't hit ending conditions during trajectory
    if not recaptured and not escaped:
        if abs(z) <= beta and r < alpha:  # Inside bounds, below atmosphere
            resimulate = True
        elif abs(z) <= beta and r > alpha:  # Inside bounds, above atmosphere
            recaptured = True
        else:  # Outside lateral bounds
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
"""
LeakRate.py
Calculates the molecular leak rate and corresponding lifetime of an atmosphere
based on particle simulation results from ringworld simulations.

V1.1, Edwin Ontiveros, June 28, 2025
V1.0, Nick Gaug, April 29, 2025
"""

import math
import pandas as pd


def LRVarInput(P_0_i, K_b_i, T_i, m_i, g_i, n_0_i, d_i):
    """
    Set parameters from the main solver for leak rate calculations.
    This function is called by StochasticInputRK45Solver.py before find_lifetime.

    LRVarInput(P_0_i, K_b_i, T_i, m_i, g_i, n_0_i, d_i)

    Inputs:
    P_0_i  Atmospheric pressure at sea level [Pa]
    K_b_i  Boltzmann constant [J/K]
    T_i    Temperature [K]
    m_i    Molecular mass [kg]
    g_i    Gravitational acceleration [m/s²]
    n_0_i  Molecular density [1/m³]
    d_i    Molecular diameter [m]

    Outputs:
    None (sets global variables)
    """
    global P_0, K_b, T, m, g, n_0, d
    P_0 = P_0_i  # Atmospheric pressure
    K_b = K_b_i  # Boltzmann constant
    T = T_i  # Temperature
    m = m_i  # Molecular mass
    g = g_i  # Gravitational acceleration
    n_0 = n_0_i  # Molecular density
    d = d_i  # Molecular diameter


def find_lifetime(file):
    """
    Calculate atmospheric lifetime based on particle escape rates.
    Uses kinetic theory to estimate mass flux from escape fraction.

    result = find_lifetime(file)

    Inputs:
    file  Path to Excel file containing particle simulation results [string]

    Outputs:
    result  Lifetime result message [string]
    """
    # Load particle data from Excel file
    result = ""
    df = pd.read_excel(file, sheet_name='Particles')
    target_words = ["resimulate", "recaptured", "escaped"]
    counts = [int((df['Result'].astype(str) == word).sum()) for word in target_words]

    # Check for division by zero
    if counts[1] == 0:
        return 'No particles recaptured; cannot calculate escape fraction'

    # Calculate escape fraction
    f_escape = counts[2] / counts[1]

    # Calculate molecular parameters
    sigma = 0.25 * math.pi * d ** 2  # Collision cross section
    lam = (sigma * n_0) ** -1  # Mean free path
    h_s = K_b * T / (m * g)  # Scale height
    alt = h_s * math.log(h_s / lam)  # Exobase altitude
    n = 429.7 / 1e12 / m  # Molecular density at exobase
    phi_m = n * m * 340 * f_escape  # Mass flux

    # Calculate lifetime
    if phi_m == 0:
        result = 'No particles escaped; lifetime indefinite'
    else:
        lifetime_years = (P_0 / g) / phi_m / 86400 / 365.2425
        result = 'Lifetime: ' + str(lifetime_years) + ' yrs'

    return result
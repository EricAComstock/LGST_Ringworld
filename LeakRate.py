"""
LeakRate.py
Calculates the molecular leak rate and corresponding lifetime of an atmosphere
based on particle simulation results from ringworld simulations.

V1.1, Edwin Ontiveros, June 28, 2025
V1.0, Nick Gaug, April 29, 2025
"""

import numpy as np
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


def lifetime_years(escaped, recaptured):
    """Calculate lifetime for counts or arrays, using NaN for undefined estimates."""
    escaped, recaptured = np.broadcast_arrays(escaped, recaptured)
    valid = (escaped > 0) & (recaptured > 0)
    years = np.full(escaped.shape, np.nan)
    if valid.any():
        f_escape = escaped[valid] / recaptured[valid]
        n = 429.7 / 1e12 / m  # Molecular density at exobase
        phi_m = n * m * 340 * f_escape  # Mass flux
        years[valid] = (P_0 / g) / phi_m / 86400 / 365.2425
    return years.item() if years.ndim == 0 else years


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
    df = pd.read_excel(file, sheet_name='Particles')
    recaptured = int((df['Result'] == 'recaptured').sum())
    escaped = int((df['Result'] == 'escaped').sum())
    if recaptured == 0:
        return 'No particles recaptured; cannot calculate escape fraction'
    if escaped == 0:
        return 'No particles escaped; lifetime indefinite'
    return f'Lifetime: {lifetime_years(escaped, recaptured)} yrs'

"""
LeakRate.py

Calculates the molecular leak rate and corresponding lifetime of an atmosphere
based on particle simulation results.

Version: 1.0
Author: Nick Gaug
Date: April 29, 2025
"""

import math
import pandas as pd


def LRVarInput(P_0_i, K_b_i, T_i, m_i, g_i, n_0_i, d_i):
    """
    Set parameters from the main solver.
    This is called by StochasticInputRK45Solver.py before find_lifetime.
    """
    global P_0, K_b, T, m, g, n_0, d
    P_0 = P_0_i
    K_b = K_b_i
    T = T_i
    m = m_i
    g = g_i
    n_0 = n_0_i
    d = d_i


def find_lifetime(file):
    """
    Calculate atmospheric lifetime based on particle escape rates.

    Parameters:
    file: Path to Excel file containing particle simulation results

    Returns:
    str: Lifetime result message
    """
    result = ""
    df = pd.read_excel(file, sheet_name='Particles')
    target_words = ["resimulate", "recaptured", "escaped"]
    counts = [int((df['Result'].astype(str) == word).sum()) for word in target_words]

    if counts[1] == 0:
        return 'No particles recaptured; cannot calculate escape fraction'

    f_escape = counts[2] / counts[1]

    sigma = 0.25 * math.pi * d ** 2
    lam = (sigma * n_0) ** -1
    h_s = K_b * T / (m * g)
    alt = h_s * math.log(h_s / lam)
    n = 429.7 / 1e12 / m
    phi_m = n * m * 340 * f_escape

    if phi_m == 0:
        result = 'No particles escaped; lifetime indefinite'
    else:
        result = 'Lifetime: ' + str((P_0 / g) / phi_m / 86400 / 365.2425) + ' yrs'

    return result
"""
LeakRate.py

Calculates the molecular leak rate and corresponding lifetime of an atmosphere
based on particle simulation results.

Version: 1.0
Author: Nick Gaug
Date: April 29, 2025 .
"""

import math
import pandas as pd


### CHANGE FILE ###

file = 'particle_data_20250523_181833.xlsx'

### CONSTANTS ###

P_0  = 101325            # Atmospheric pressure at sea level (Pa)
K_b  = 1.380649e-23      # Boltzmann constant (J/K)
T  = 289               # Standard temperature (K)
m    = 2.6566962e-26 * 2 # Mass of diatomic molecule (kg)
g    = 9.81              # Gravitational acceleration (m/s^2)
n_0  = 2.687e25          # Standard atmospheric molecular density (1/m^3)
d    = 3.59e-10          # Molecular diameter (m)

### CALCULATE F_ESCAPE ###



### CALCULATE LEAK RATE ###
def LRVarInput(P_0_i,K_b_i,T_i,m_i,g_i,n_0_i,d_i):
    global P_0,K_b,T,m,g,n_0,d
    P_0 = P_0_i
    K_b = K_b_i
    T = T_i
    m = m_i
    g = g_i
    n_0 = n_0_i
    d = d_i

def find_lifetime (file):
    result = ""
    df = pd.read_excel(file, sheet_name='Particles')
    target_words = ["resimulate", "recaptured", "escaped"]
    counts = [int((df['Result'].astype(str) == word).sum()) for word in target_words]

    f_escape = counts[2] / counts[1]

    sigma = 0.25 * math.pi * d ** 2

    lam = (sigma * n_0) ** -1

    h_s = K_b * T / (m * g)

    alt = h_s * math.log(h_s / lam)

    # n = (P_0 / (K_b * T_0)) * math.e ** (-alt / h_s)
    n = 429.7 / 1e12 / m

    phi_m = n * m * 340 * f_escape


    if(phi_m == 0):
        result = 'No particles escaped; lifetime indefinite'
    else:
        result = ('Lifetime: '+ str((P_0 / g) / phi_m / 86400 / 365.2425)+ ' yrs')


    return result


    
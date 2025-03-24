# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:01:17 2025

@author: NickGaug
"""

import math as math
import pandas as pd

###CONSTANTS####

P_0 = 101325 #Pa
K_b = 1.380649e-23 #J/K
T_0 = 300 #K
m = 2.6566962e-26 #kg
g = 9.81 #m/s^2
n_0 = 2.687e25 #1/m^3
d = 0.000359 #m  7.3e-11 #m

###CALCULATE F_ESCAPE###

df = pd.read_excel('particle_data_20250304_123751.xlsx', sheet_name='Particles')
target_words = ["resimulate", "recaptured", "escaped"]
counts = [int((df['Result'].astype(str) == word).sum()) for word in target_words]

###CALCULATE LEAK RATE###

f_escape = counts[2] / counts[1]

sigma = 0.25*math.pi*d**2

lam = (sigma*n_0)**-1

h_s = K_b*T_0/(m*g)

alt = h_s*math.log(h_s/lam)

n = (P_0/(K_b*T_0))*math.e**(-alt/h_s)

phi_m = n*m

print(phi_m)


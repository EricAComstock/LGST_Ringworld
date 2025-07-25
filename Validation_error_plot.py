"""
SolverValidationPlot.py

Creates solver validation error plot showing RK45 numerical accuracy over time.

1.0 , Edwin Ontiveros, 27/05/2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib fonts
plt.rcdefaults()
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
mpl.rcParams['font.size'] = 10

# Create figure
fig = plt.figure(figsize=(3.5, 3))
ax = fig.add_subplot(111)

# Validation data from solver tests
time_points = [10, 100, 1000, 10000, 100000]  # Integration times (s)
errors = [1e-7, 0.000031, 0.000001, 0.000214, 0.016699]  # Position errors (m)

# Plot error vs time
ax.loglog(time_points, errors, 'bo-', linewidth=1.5, markersize=6)
ax.set_xlabel('Integration time (s)')
ax.set_ylabel('Position error (m)')
ax.set_xlim(10, 1e5)
ax.set_ylim(1e-7, 1e-1)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add 1 micron reference line
ax.axhline(y=1e-6, color='#808080', linestyle='--', alpha=0.5, linewidth=0.8)

# Add tolerance label
ax.text(0.05, 0.05, 'RK45, tol = 10$^{-12}$',
        transform=ax.transAxes, fontsize=9)

# Subtle grid
ax.grid(True, alpha=0.1, which='major', linestyle='-', linewidth=0.5)

# Clean tick marks
ax.tick_params(direction='out', top=False, right=False)

# Save and display
plt.tight_layout()
plt.savefig('validation_error.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
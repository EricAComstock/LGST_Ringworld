"""
Ringworld gravity.py

Generates plots of the gravity field induced by a ringworld and its magnitude and direction

% V1.2, Edwin Ontiveros, 27/05/2025
% V1.1, Eric Comstock, 15/05/2025
% V1.0, Eric Comstock, 25/02/2025
"""

import numpy as np
import scipy.integrate as ingz
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set matplotlib to use Arial font and appropriate sizes
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'


def grav_vecs(V, x, y, z):
    # This function calculates the forces induced to a particle from the gravity of the ringworld
    #
    # Inputs:
    #   V   is the function for the ringworld gravitational potential calculated from position inputs
    #   x   is the position along the first axis (in the rotation plane of the ringworld) [km]
    #   y   is the position along the second axis (in the rotation plane of the ringworld) [km]
    #   z   is the position along the third axis (on the axis of the ringworld) [km]
    #
    # Outputs:
    #   u   is the acceleration along the first axis (in the rotation plane of the ringworld) [km / s^2]
    #   v   is the acceleration along the second axis (in the rotation plane of the ringworld) [km / s^2]
    #   w   is the acceleration along the third axis (on the axis of the ringworld) [km / s^2]

    delta = 1e-2  # Finite difference constant [km]
    pot_base = V(x, y, z)  # Base potential [km^2 / s^2]

    # Calculate accelerations with FDM [km / s^2]
    u = -(pot_base - V(x - delta, y, z)) / delta
    v = -(pot_base - V(x, y - delta, z)) / delta
    w = -(pot_base - V(x, y, z - delta)) / delta
    return u, v, w


def V_Ring(x, y, z, R=149597870.7, a=1.609344 * 1e6):
    # This function calculates the ringworld gravitational potential calculated from position inputs
    #
    # Inputs:
    #   x   is the position along the first axis (in the rotation plane of the ringworld) [km]
    #   y   is the position along the second axis (in the rotation plane of the ringworld) [km]
    #   z   is the position along the third axis (on the axis of the ringworld) [km]
    #   R   is the ringworld radius [km]
    #   a   is the ringworld width [km]
    #
    # Outputs:
    #   The gravitational position of the ringworld [km / s^2]

    G = 6.67430e-11 / 1e9  # Gravitational constant in units of km-s-kg
    rho_A = 1e6 * 70000  # density per km^2
    mu_A = G * rho_A  # Gravitational parameter per unit area [km / s^2]

    def V_Ring_point(s, h):
        # This function calculates the gravitational potential of a single ringworld point
        #
        # Inputs:
        #   s   is the distance around the ringworld of it
        #   h   is the distance of the point from the ringworld's rotation plane
        #
        # Outputs:
        #   The gravitational position of the ringworld point [km^3 / s^2]

        th = s / R  # Finding angle of point
        r = np.sqrt((x - R * np.cos(th)) ** 2 + (y - R * np.sin(th)) ** 2 + (
                    z - h) ** 2)  # Compute radius between point of ringworld and point of measurement
        V_r = -mu_A / r  # Find gravitational potential
        return V_r

    # Calculate potentials from near ringworld sections and far ringworld sections
    result1 = ingz.dblquad(V_Ring_point, -a / 2, a / 2, lambda x: -0.02 * R, lambda x: 0.02 * R)
    result2 = ingz.dblquad(V_Ring_point, -a / 2, a / 2, lambda x: 0.02 * R, lambda x: (2 * np.pi - 0.02) * R)

    # Adds near and far potentials together
    return result1[0] + result2[0]


# Generating plotting field
N = 38

x = np.linspace(149597870.7 - 1000000, 149597870.7 + 1000000, N)
y = 0
z = np.linspace(-1000000, 1000000, N)
X, Z = np.meshgrid(x, z)

# Set this to true to plot only the potential, false to plot everything
potential = False

# Finding
u = X - X
v = X - X
w = X - X
V = X - X

for i in range(len(x)):
    for j in range(len(z)):
        print((len(z) * i + j) / len(x) / len(z) * 100, '%')
        if potential:
            V[i, j] = V_Ring(X[i, j], y, Z[i, j])
        else:
            ui, vi, wi = grav_vecs(V_Ring, X[i, j], y, Z[i, j])
            u[i, j] = ui
            v[i, j] = vi
            w[i, j] = wi

plt.close('all')

# Define colorblind-safe colormap
colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
cmap = LinearSegmentedColormap.from_list('blues', colors)

# Smaller figure size for publication
plt.figure(figsize=(3.5, 3))
if potential:
    plt.contourf(X, Z, V)
    plt.colorbar()
else:
    A = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    # Scale up by 1e5 for display
    con = plt.contourf(X, Z, A * 1000 * 1e5, 15, cmap=cmap)
    plt.quiver(X[::3, ::3], Z[::3, ::3], u[::3, ::3] / A[::3, ::3], w[::3, ::3] / A[::3, ::3], scale=30)
    plt.xlabel('Radial distance (km)')
    plt.ylabel('Vertical distance (km)')
    # Remove title - captions describe figures
    cbar = plt.colorbar(con)
    cbar.set_label('Acceleration (10$^{-5}$ m s$^{-2}$)', rotation=270, labelpad=15)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save as PDF
plt.tight_layout()
plt.savefig('ringworld_gravity_field.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
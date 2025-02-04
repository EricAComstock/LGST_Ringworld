# Import libraries
import numpy as np
from scipy.integrate import solve_ivp
from Nick_Gaug_StochasticInput import *

def equations_of_motion(t, state, omega):
    r = np.array(state[:3])
    v = np.array(state[3:])
    coriolis_force = -2 * np.cross(omega, v)
    centrifugal_force = -np.cross(omega, np.cross(omega, r))
    return [*v, *(coriolis_force + centrifugal_force)]

# Use Nick's stored test values
initial_conditions = input_test

# Get simulation parameters
t_final = float(input("Enter simulation time in seconds (default 10.0): ") or 10.0)
dt = float(input("Enter time step in seconds (default 0.1): ") or 0.1)
is_rotating = input("Is the frame rotating? (yes/no, default yes): ").strip().lower() or "yes"

t_span = (0, t_final)
t_eval = np.arange(0, t_final, dt)
omega = np.array([0, 0, np.sqrt(9.81/alt_min) if is_rotating == "yes" else 0])
print(f"\nFrame is {'rotating' if is_rotating == 'yes' else 'not rotating'}")

solution = solve_ivp(
    equations_of_motion,
    t_span,
    initial_conditions,
    t_eval=t_eval,
    args=(omega,),
    method='RK45',
    rtol=1e-8,
    atol=1e-8
)

final_position = np.array([solution.y[0, -1], solution.y[1, -1], solution.y[2, -1]])
final_velocity = np.array([solution.y[3, -1], solution.y[4, -1], solution.y[5, -1]])

print("\nFinal State:")
print(f"Position: {final_position}")
print(f"Velocity: {final_velocity}")

'''
# Optional: Print all integration steps
for i in range(len(solution.t)):
    print(f"\nTime: {solution.t[i]:.2f}s")
    print(f"Position: {solution.y[:3, i]}")
    print(f"Velocity: {solution.y[3:, i]}")
'''
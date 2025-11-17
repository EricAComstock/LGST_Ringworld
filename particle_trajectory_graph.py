import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def plot_trajectories_from_pkl(pkl_path):
    # --- Load the pickle file ---
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    particles = data["particles"]

    # Assume only 1 particle for now (can easily support many)
    p = particles[0]
    traj = p["trajectory"]      # shape (T, 3)
    time = p["time_steps"]

    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    # -----------------------
    # Plot XY trajectory
    # -----------------------
    plt.figure(figsize=(7, 6))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY Trajectory")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # -----------------------
    # Plot ZY trajectory
    # -----------------------
    plt.figure(figsize=(7, 6))
    plt.plot(z, y, linewidth=2)
    plt.xlabel("Z")
    plt.ylabel("Y")
    plt.title("ZY Trajectory")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

plot_trajectories_from_pkl("Bishop_Ring_1particles_20251116_184037_trajectories.pkl")
"""
def plot_trajectories_from_tensor(positions, title="Particle trajectories"):
    positions: np.ndarray of shape (T, N, 3)
               positions[t, i] = [x, y, z] of particle i at time index t
    positions = np.asarray(positions)
    T, N, D = positions.shape
    assert D == 3, "Last dimension of positions must be 3 (x, y, z)."

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N):
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        z = positions[:, i, 2]
        ax.plot(x, y, z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


# Example usage:
T = 500
N = 5
t = np.linspace(0, 10, T)
positions = np.zeros((T, N, 3))

for i in range(N):
    positions[:, i, 0] = np.cos(t + 0.5 * i)
    positions[:, i, 1] = np.sin(t + 0.5 * i)
    positions[:, i, 2] = 0.1 * t

plot_trajectories_from_tensor(positions)
"""
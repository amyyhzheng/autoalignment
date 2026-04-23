from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

Coord = Tuple[float, float, float]

def get_xyzs(coords: List[Coord]):
    return [p[0] for p in coords], [p[1] for p in coords], [p[2] for p in coords]

def euc_xy(a: Coord, b: Coord) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])

def distance_along_branch(branch: List[Coord], start_idx: int, end_idx: int) -> float:
    d = 0.0
    for i in range(start_idx, end_idx):
        d += euc_xy(branch[i], branch[i + 1])
    return d

# def fit_branch_spline(x: np.ndarray, y: np.ndarray, z: np.ndarray, n_points: int = 100):
#     tck, u = splprep([x, y, z], s=0)
#     u_new = np.linspace(0, 1, n_points)
#     x_new, y_new, z_new = splev(u_new, tck)
#     return list(zip(x_new, y_new, z_new))

def fit_branch_spline(x: np.ndarray, y: np.ndarray, z: np.ndarray, n_points: int = 100):
    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("len(x):", len(x), "len(y):", len(y), "len(z):", len(z))
    tck, u = splprep([x, y, z], s=0)
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new, z_new = splev(u_new, tck)
    return list(zip(x_new, y_new, z_new))

def plot_branch_and_synapses(branch_points, synapse_points):
    # branch_points: list of (x, y, z) tuples
    # synapse_points: list of (x, y, z) tuples

    bx, by, bz = zip(*branch_points)
    sx, sy, sz = zip(*synapse_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(bx, by, bz, '-k', label='Branch')
    ax.scatter(sx, sy, sz, c='r', label='Synapses')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
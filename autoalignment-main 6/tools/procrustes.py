import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

# -------------------------------------------------------------------------
# Load whole-cell coordinates (MAP) and the analyzed segment coordinates (MAP)
# -------------------------------------------------------------------------

filename = 'MAP_1387_2024-09-08_1387_10x_Overview_xyzCoordinates.csv'
data = pd.read_csv(filename)

x_full = data['x']
y_full = data['y']
z_full = data['z']

# Load analyzed segment coordinates
analyzed_data = pd.read_csv('MAP_1387_analayezed_seg.csv')
x_a = analyzed_data['x']
y_a = analyzed_data['y']
z_a = analyzed_data['z']

# -------------------------------------------------------------------------
# Plot the analyzed segments on top of the full neuron MAP
# -------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_full, y_full, z_full, '.', color='b', markersize=2)
ax.scatter(x_a, y_a, z_a, color='r', s=2, label='analyzed')

ax.set_xlabel('X (μm)')
ax.set_ylabel('Y (μm)')
ax.set_zlabel('Z (μm)')
ax.set_title('3D Trace with Synapses')
ax.legend(loc='best')
ax.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Load additional data files
# -------------------------------------------------------------------------
exp_data = pd.read_csv('exp_branches_cell1387.csv')
syn_data = pd.read_csv('synapses_remapped.csv')

# Normalize analyzed_data.path column to match exp_data and syn_data
analyzed_data['path'] = analyzed_data['path'].str.split('_on').str[0]

# -------------------------------------------------------------------------
# Compute translation offsets (dx, dy, dz) for each analyzed segment
# -------------------------------------------------------------------------
unique_paths = exp_data['path'].unique()

diff_table = pd.DataFrame(columns=['path', 'dx', 'dy', 'dz'])

for current_path in unique_paths:
    exp_rows = exp_data[exp_data['path'] == current_path]
    if exp_rows.empty:
        continue

    exp_first = exp_rows.iloc[0]
    analyzed_rows = analyzed_data[analyzed_data['path'] == current_path]

    if not analyzed_rows.empty:
        analyzed_first = analyzed_rows.iloc[0]
        dx = exp_first['x'] - analyzed_first['x']
        dy = exp_first['y'] - analyzed_first['y']
        dz = exp_first['z'] - analyzed_first['z']

        diff_table.loc[len(diff_table)] = [current_path, dx, dy, dz]
    else:
        print(f'Path {current_path} not found in analyzed data.')

# -------------------------------------------------------------------------
# Apply translation offsets (shift) to align 60x data with MAP
# -------------------------------------------------------------------------
exp_data_corrected = exp_data.copy()
syn_data_corrected = syn_data.copy()

for _, row in diff_table.iterrows():
    path = row['path']
    dx, dy, dz = row['dx'], row['dy'], row['dz']

    mask_exp = exp_data_corrected['path'] == path
    mask_syn = syn_data_corrected['path'] == path

    exp_data_corrected.loc[mask_exp, 'x'] -= dx
    exp_data_corrected.loc[mask_exp, 'y'] -= dy
    exp_data_corrected.loc[mask_exp, 'z'] -= dz

    syn_data_corrected.loc[mask_syn, 'X'] -= dx
    syn_data_corrected.loc[mask_syn, 'Y'] -= dy
    syn_data_corrected.loc[mask_syn, 'Z'] -= dz

# -------------------------------------------------------------------------
# Rotation: Align 60x data to MAP
# -------------------------------------------------------------------------
exp_data_rotated = exp_data_corrected.copy()
syn_data_rotated = syn_data_corrected.copy()

for current_path in unique_paths:
    mask_exp = exp_data_corrected['path'] == current_path
    mask_anl = analyzed_data['path'] == current_path
    mask_syn = syn_data_corrected['path'] == current_path

    exp_points = exp_data_corrected.loc[mask_exp, ['x', 'y', 'z']].to_numpy()
    analyzed_points = analyzed_data.loc[mask_anl, ['x', 'y', 'z']].to_numpy()
    syn_points = syn_data_corrected.loc[mask_syn, ['X', 'Y', 'Z']].to_numpy()

    if len(exp_points) < 3 or len(analyzed_points) < 3:
        continue  # not enough points for alignment

    num_points_to_match = 12
    idxs_exp = np.round(np.linspace(0, len(exp_points)-1, num_points_to_match)).astype(int)
    idxs_anl = np.round(np.linspace(0, len(analyzed_points)-1, num_points_to_match)).astype(int)

    P = exp_points[idxs_exp, :]
    Q = analyzed_points[idxs_anl, :]

    # SciPy’s procrustes standardizes data; to get transform params, use custom calc
    # We'll use a helper to compute rotation/scale/translation manually
    def compute_procrustes(Q, P):
        # Subtract centroids
        mu_P = P.mean(axis=0)
        mu_Q = Q.mean(axis=0)
        P_centered = P - mu_P
        Q_centered = Q - mu_Q

        # Compute rotation with SVD
        U, _, Vt = np.linalg.svd(P_centered.T @ Q_centered)
        R = Vt.T @ U.T

        # Correct for reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute scaling
        scale = np.trace((Q_centered.T @ P_centered @ R)) / np.trace(P_centered.T @ P_centered)
        t = mu_Q - scale * mu_P @ R
        return scale, R, t

    scale, R, t = compute_procrustes(Q, P)

    rotated = scale * exp_points @ R + t
    rotated_syn = scale * syn_points @ R + t

    exp_data_rotated.loc[mask_exp, ['x', 'y', 'z']] = rotated
    syn_data_rotated.loc[mask_syn, ['X', 'Y', 'Z']] = rotated_syn

# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------
exp_data_rotated.to_csv('exp_cell1387_data_matched_rotated.csv', index=False)
syn_data_rotated.to_csv('synapses_cell1387_data_matched_rotated.csv', index=False)

print("Alignment complete. Files saved.")

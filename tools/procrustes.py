import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

# -------------------------------------------------------------------------
# Load whole-cell coordinates (MAP) and the analyzed segment coordinates (MAP)
# -------------------------------------------------------------------------

filename = r"Z:\Amy\Imaging\3562analysis\3562_10xOverview_A01_G003_0001_2_xyzCoordinates.csv"
data = pd.read_csv(filename)

x_full = data['x']
y_full = data['y']
z_full = data['z']

# Load analyzed segment coordinates
analyzed_data = pd.read_csv(r"Z:\Amy\Imaging\3562analysis\3562_10xOverview_A01_G003_0001_2_xyzCoordinates.csv")
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

exp_data = pd.read_csv(r"Z:\Amy\Imaging\3562analysis\3562_60x_Cell1_A01_G009_0001_apical4_xyzCoordinates.csv")
syn_data = pd.read_csv(r"Z:\Bettina\_Imaging_Analysis\SNT_Traces\Blinded_Analysis_Sept2025\unblind by groups_processedForSending\Amy\apical4.csv")
print(f"syn_data{syn_data}")


# Normalize analyzed_data.path column to match exp_data and syn_data
analyzed_data['path'] = analyzed_data['path'].str.split('_on').str[0]
print(analyzed_data['path'])

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

    num_points_to_match = 2
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

exp_data_rotated.to_csv('exp_cell1387_data_matched_rotated.csv', index=False)
syn_data_rotated.to_csv('synapses_cell1387_data_matched_rotated.csv', index=False)

print("Alignment complete. Files saved.")

# -------------------------------------------------------------------------
# Optional visualization after alignment
# -------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot MAP neuron (background)
ax.plot(x_full, y_full, z_full, '.', color='gray', markersize=1, alpha=0.3, label='MAP (10x)')

# Plot aligned 60x branches
ax.scatter(
    exp_data_rotated['x'],
    exp_data_rotated['y'],
    exp_data_rotated['z'],
    s=2, color='blue', label='Aligned 60x branches'
)

# Plot aligned synapses
ax.scatter(
    syn_data_rotated['X'],
    syn_data_rotated['Y'],
    syn_data_rotated['Z'],
    s=5, color='red', label='Synapses'
)

ax.set_xlabel('X (μm)')
ax.set_ylabel('Y (μm)')
ax.set_zlabel('Z (μm)')
ax.set_title('Aligned 60x Data in MAP Coordinate Frame')
ax.legend(loc='best')
ax.grid(True)
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

# # -----------------------------------------------------------
# # Load data
# # -----------------------------------------------------------

# # 10× MAP full neuron (for background plotting)
# map_file = r"Z:\Amy\Imaging\3562analysis\3562_10xOverview_A01_G003_0001_2_xyzCoordinates.csv"
# map_data = pd.read_csv(map_file)

# # 10× analyzed subset (used as reference for alignment)
# analyzed_file = r"Z:\Amy\Imaging\3562analysis\3562_10xOverview_A01_G003_0001_2_xyzCoordinates.csv"
# analyzed_data = pd.read_csv(analyzed_file)

# # 60× experimental branches
# exp_file = r"Z:\Amy\Imaging\3562analysis\3562_60x_Cell1_A01_G009_0001_apical4_xyzCoordinates.csv"
# exp_data = pd.read_csv(exp_file)

# # 60× synapses (unprocessed independent coordinates)
# syn_file = r"Z:\Bettina\_Imaging_Analysis\SNT_Traces\Blinded_Analysis_Sept2025\unblind by groups_processedForSending\Amy\apical4.csv"
# syn_data = pd.read_csv(syn_file)

# print("Loaded:")
# print(f"MAP points: {len(map_data)}")
# print(f"Analyzed 10× segments: {len(analyzed_data)}")
# print(f"60× branch points: {len(exp_data)}")
# print(f"Synapses: {len(syn_data)}")

# # -----------------------------------------------------------
# # Normalize the path names so they match between datasets
# # -----------------------------------------------------------

# if "path" in analyzed_data.columns:
#     analyzed_data["path"] = analyzed_data["path"].astype(str).str.split("_on").str[0]

# if "path" in exp_data.columns:
#     exp_data["path"] = exp_data["path"].astype(str)

# if "path" in syn_data.columns:
#     syn_data["path"] = syn_data["path"].astype(str)

# # -----------------------------------------------------------
# # Resampling helper
# # -----------------------------------------------------------

# def resample_branch(points, n):
#     """Resample a polyline to n equally spaced points along its arc length."""
#     if len(points) < 2:
#         return None
#     diffs = np.diff(points, axis=0)
#     dist = np.insert(np.cumsum(np.linalg.norm(diffs, axis=1)), 0, 0)
#     if dist[-1] == 0:
#         return None
#     interp = interp1d(dist, points, axis=0)
#     new_d = np.linspace(0, dist[-1], n)
#     return interp(new_d)

# # -----------------------------------------------------------
# # Collect corresponding pairs of points for GLOBAL alignment
# # -----------------------------------------------------------

# corr_exp = []   # 60×
# corr_map = []   # 10×

# for path in exp_data["path"].unique():
#     exp_pts = exp_data.loc[exp_data["path"] == path, ["x", "y", "z"]].to_numpy()
#     map_pts = analyzed_data.loc[analyzed_data["path"] == path, ["x", "y", "z"]].to_numpy()

#     if len(exp_pts) < 3 or len(map_pts) < 3:
#         print(f"Skipping path {path}: too few points for alignment")
#         continue

#     # use up to 50 points per branch to get good global stability
#     n = 5
#     exp_rs = resample_branch(exp_pts, n)
#     map_rs = resample_branch(map_pts, n)

#     if exp_rs is None or map_rs is None:
#         continue

#     corr_exp.append(exp_rs)
#     corr_map.append(map_rs)

# # Stack all correspondence pairs
# corr_exp = np.vstack(corr_exp)
# corr_map = np.vstack(corr_map)

# print(f"Total matched points for global alignment: {corr_exp.shape[0]}")

# # -----------------------------------------------------------
# # Compute global rigid transform (rotation + translation)
# # -----------------------------------------------------------

# def rigid_fit(A, B):
#     """Compute rotation + translation that maps A → B."""
#     muA = A.mean(axis=0)
#     muB = B.mean(axis=0)
#     A0 = A - muA
#     B0 = B - muB

#     U, _, Vt = np.linalg.svd(A0.T @ B0)
#     R = Vt.T @ U.T

#     # prevent reflection
#     if np.linalg.det(R) < 0:
#         Vt[-1] *= -1
#         R = Vt.T @ U.T

#     t = muB - muA @ R
#     return R, t

# R, t = rigid_fit(corr_exp, corr_map)

# print("Global rigid transform computed.")
# print("Rotation matrix:\n", R)
# print("Translation vector:", t)

# # -----------------------------------------------------------
# # Apply transform to entire 60× dataset
# # -----------------------------------------------------------

# exp_xyz = exp_data[["x", "y", "z"]].to_numpy()
# syn_xyz = syn_data[["X", "Y", "Z"]].to_numpy()

# exp_aligned = exp_xyz @ R + t
# syn_aligned = syn_xyz @ R + t

# exp_data_aligned = exp_data.copy()
# syn_data_aligned = syn_data.copy()

# exp_data_aligned[["x", "y", "z"]] = exp_aligned
# syn_data_aligned[["X", "Y", "Z"]] = syn_aligned

# # -----------------------------------------------------------
# # Save results
# # -----------------------------------------------------------

# exp_out = "exp_data_global_aligned.csv"
# syn_out = "syn_data_global_aligned.csv"

# exp_data_aligned.to_csv(exp_out, index=False)
# syn_data_aligned.to_csv(syn_out, index=False)

# print(f"Saved global aligned 60× data to: {exp_out}")
# print(f"Saved global aligned synapses to: {syn_out}")

# # -----------------------------------------------------------
# # Visualization
# # -----------------------------------------------------------

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # MAP background
# ax.scatter(map_data["x"], map_data["y"], map_data["z"],
#            s=1, color="gray", alpha=0.3, label="MAP (10×)")

# # aligned branches
# ax.scatter(exp_data_aligned["x"], exp_data_aligned["y"], exp_data_aligned["z"],
#            s=4, color="blue", label="Aligned 60× branches")

# # aligned synapses
# ax.scatter(syn_data_aligned["X"], syn_data_aligned["Y"], syn_data_aligned["Z"],
#            s=8, color="red", label="Aligned synapses")

# ax.set_xlabel("X (µm)")
# ax.set_ylabel("Y (µm)")
# ax.set_zlabel("Z (µm)")
# ax.set_title("Global Alignment of 60× Data to 10× MAP")
# ax.legend(loc="best")
# ax.grid(True)
# ax.view_init(elev=20, azim=45)

# plt.tight_layout()
# plt.show()

# print("Visualization complete.")


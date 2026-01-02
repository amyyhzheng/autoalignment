import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import napari
import networkx as nx
from scipy.spatial import cKDTree


def load_napari_points_as_xyz(csv_path: str) -> np.ndarray:
    """
    Load napari Points-layer CSV and return landmarks as (N,3) in XYZ order.

    Handles common napari exports:
      - columns include z,y,x (case-insensitive)
      - columns include axis-0, axis-1, axis-2 (napari axis order)
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "z" in cols_lower and "y" in cols_lower and "x" in cols_lower:
        zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
        ZYX = df[[zc, yc, xc]].to_numpy(dtype=float)
        XYZ = ZYX[:, [2, 1, 0]]
        return XYZ

    axis_cols = [k for k in cols_lower.keys() if k.startswith("axis-")]
    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))
        A012 = df[[c0, c1, c2]].to_numpy(dtype=float)  # napari order (usually z,y,x)
        XYZ = A012[:, [2, 1, 0]]  # -> x,y,z
        return XYZ

    raise ValueError(f"Unrecognized napari CSV columns: {list(df.columns)}")


def rigid_transform_3d(A, B):
    """
    Fit rigid transform mapping B -> A.
    Returns R, t such that B_aligned = (B @ R.T) + t
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    U, S, Vt = np.linalg.svd(BB.T @ AA)  # B->A
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_A - centroid_B @ R.T
    return R, t

def apply_rigid(P_xyz, R, t):
    P = np.asarray(P_xyz, float)
    return (P @ R.T) + t


def plot_branches_and_landmarks_3d(df_ref, df_mov, A_xyz, B_xyz, title):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(df_ref["x"], df_ref["y"], df_ref["z"], s=4, label="ref branch")
    ax.scatter(df_mov["x"], df_mov["y"], df_mov["z"], s=4, label="moving branch")

    ax.scatter(A_xyz[:,0], A_xyz[:,1], A_xyz[:,2], s=60, marker="o", label="ref landmarks")
    ax.scatter(B_xyz[:,0], B_xyz[:,1], B_xyz[:,2], s=60, marker="^", label="mov landmarks")

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    plt.show()

# LOAD TRACES
df_ref = pd.read_csv('/Users/amyzheng/Desktop/coronalsectionplot/3562Trace/3562_10xOverview_A01_G003_0001_2_xyzCoordinates.csv')
df_mov = pd.read_csv('/Users/amyzheng/Desktop/coronalsectionplot/3562Trace/3562_60x_Cell1_A01_G009_0001_apical4_xyzCoordinates - Copy (3).csv')

# LANDMARK CSVs (napari format)
A = load_napari_points_as_xyz('/Users/amyzheng/Desktop/coronalsectionplot/apical4refnew4.csv')     # reference
B = load_napari_points_as_xyz('/Users/amyzheng/Desktop/coronalsectionplot/apical4mapnew3.csv')  # moving

# FIT TRANSFORM
R, t = rigid_transform_3d(A, B)

# APPLY TO MOVING TRACE
df_mov_aligned = df_mov.copy()
df_mov_aligned[["x","y","z"]] = apply_rigid(df_mov[["x","y","z"]].to_numpy(), R, t)

extra_points_csv ='/Users/amyzheng/Desktop/coronalsectionplot/Amy/3562_60x_Cell1_A01_G009_0001_apical4.csv'
extra_xyz = load_napari_points_as_xyz(extra_points_csv)
extra_xyz_aligned = apply_rigid(extra_xyz, R, t)

# (Optional) show before/after in matplotlib
# plot_branches_and_landmarks_3d(df_ref, df_mov, A, B, "Before alignment")
# plot_branches_and_landmarks_3d(df_ref, df_mov_aligned, A, apply_rigid(B, R, t), "After rigid alignment")

# -----------------------------
# NAPARI VISUALIZATION
# -----------------------------
viewer = napari.Viewer(ndisplay=3)

def rand_color(name: str):
    seed = abs(hash(name)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(3)

# # 10x ref: if it has path, color per path; else single layer
if "path" in df_ref.columns:
    for path_name, g in df_ref.groupby("path"):
        pts_zyx = g[["z", "y", "x"]].to_numpy(dtype=float)
        viewer.add_points(
            pts_zyx,
            name=f"10x::{path_name}",
            size=2,
            face_color=rand_color(str(path_name)),
            opacity=0.6,
        )
else:
    pts_zyx = df_ref[["z", "y", "x"]].to_numpy(dtype=float)
    viewer.add_points(
        pts_zyx,
        name="10x::all",
        size=2,
        face_color=[0.7, 0.7, 0.7],
        opacity=0.5,
    )

# aligned moving trace: single layer (or per-path if you prefer)
if "path" in df_mov_aligned.columns:
    # per-path layers
    for path_name, g in df_mov_aligned.groupby("path"):
        pts_zyx = g[["z", "y", "x"]].to_numpy(dtype=float)
        viewer.add_points(
            pts_zyx,
            name=f"60x_aligned::{path_name}",
            size=3,
            face_color=rand_color("mov_" + str(path_name)),
            opacity=1.0,
        )
else:
    pts_zyx = df_mov_aligned[["z", "y", "x"]].to_numpy(dtype=float)
    viewer.add_points(
        pts_zyx,
        name="60x_aligned::all",
        size=3,
        face_color=[0.2, 0.6, 1.0],
        opacity=1.0,
    )

# extra napari points after transform
viewer.add_points(
    extra_xyz_aligned[:, [2,1,0]],
    name="extra_points_aligned",
    size=8,
    face_color=[1.0, 0.2, 0.2],
    opacity=1
)

napari.run()

def build_dendrite_tree_from_xyz_df(full_trace_df: pd.DataFrame,
                                   branch_col: str = "path",
                                   coord_cols=("x","y","z"),
                                   max_edge_length=5.0,
                                   reference_point=None):
    """
    Build a dendrite graph from a trace dataframe.
    Nodes are XYZ tuples. Edges connect consecutive points within each branch/path.
    Optionally add a 'center' node by taking the centroid of the closest terminal nodes
    (one per connected component) to a provided reference_point.
    """
    G = nx.Graph()

    if branch_col in full_trace_df.columns:
        groups = full_trace_df.groupby(branch_col)
    else:
        # treat all rows as one branch if no branch column
        groups = [(0, full_trace_df)]

    for branch_id, group in groups:
        coords = group.loc[:, coord_cols].dropna().to_numpy(dtype=float)
        if len(coords) < 2:
            continue

        for i in range(len(coords) - 1):
            p1 = tuple(coords[i])
            p2 = tuple(coords[i + 1])
            if p1 == p2:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[i + 1]))
            if dist < max_edge_length:
                G.add_edge(p1, p2, weight=dist)

    center = None
    if reference_point is not None and len(G) > 0:
        reference_point = np.asarray(reference_point, dtype=float)
        components = list(nx.connected_components(G))
        closest_terminal_nodes = []

        for comp in components:
            terminal_nodes = [node for node in comp if G.degree[node] == 1]
            if not terminal_nodes:
                continue
            terminal_coords = np.array(terminal_nodes, dtype=float)
            dists = np.linalg.norm(terminal_coords - reference_point[None, :], axis=1)
            closest_terminal_nodes.append(terminal_nodes[int(np.argmin(dists))])

        if closest_terminal_nodes:
            closest_coords = np.array(closest_terminal_nodes, dtype=float)
            center = tuple(np.mean(closest_coords, axis=0))
            G.add_node(center)
            for node in closest_terminal_nodes:
                dist = float(np.linalg.norm(np.asarray(node) - np.asarray(center)))
                G.add_edge(node, center, weight=dist)

    return G, center


def find_nearest_node(coord_xyz, kdtree, node_coords):
    dist, idx = kdtree.query(np.asarray(coord_xyz, dtype=float))
    return tuple(node_coords[idx]), float(dist)

# df_ref must have columns x,y,z (and optionally path)
G10x, center10x = build_dendrite_tree_from_xyz_df(
    df_ref,
    branch_col="path" if "path" in df_ref.columns else "__no_path__",
    coord_cols=("x","y","z"),
    max_edge_length=5.0,
    # pick a reference point only if you want a center node; otherwise set None
    reference_point=None
)

node_coords = np.array(list(G10x.nodes), dtype=float)
kdtree = cKDTree(node_coords)

extra_xyz = np.asarray(extra_xyz_aligned, dtype=float)  # (N,3) XYZ
nearest_nodes = []
nearest_dists = []

for p in extra_xyz:
    nn, d = find_nearest_node(p, kdtree, node_coords)
    nearest_nodes.append(nn)
    nearest_dists.append(d)

nearest_nodes = np.array(nearest_nodes, dtype=float)  # (N,3)
nearest_dists = np.array(nearest_dists, dtype=float)

# Save results 
matches = pd.DataFrame({
    "extra_i": np.arange(len(extra_xyz)),
    "extra_x": extra_xyz[:,0],
    "extra_y": extra_xyz[:,1],
    "extra_z": extra_xyz[:,2],
    "nearest10x_x": nearest_nodes[:,0],
    "nearest10x_y": nearest_nodes[:,1],
    "nearest10x_z": nearest_nodes[:,2],
    "euclid_dist_to_graph_node": nearest_dists
})
print(matches.sort_values("euclid_dist_to_graph_node").head(10))

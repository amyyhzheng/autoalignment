import numpy as np
import pandas as pd
import napari
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter


# -----------------------------
# IO HELPERS
# -----------------------------
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


def load_napari_points_df(csv_path: str) -> pd.DataFrame:
    """
    Load napari Points-layer CSV as DataFrame and add standardized coordinate columns:
      _x, _y, _z  (in XYZ order)

    Preserves any other columns (e.g., 'label').
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    if {"z", "y", "x"} <= set(cols_lower):
        zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
        df["_x"] = df[xc].astype(float)
        df["_y"] = df[yc].astype(float)
        df["_z"] = df[zc].astype(float)
        return df

    axis_cols = [k for k in cols_lower if k.startswith("axis-")]
    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))
        # napari axis order is usually z,y,x
        df["_x"] = df[c2].astype(float)
        df["_y"] = df[c1].astype(float)
        df["_z"] = df[c0].astype(float)
        return df

    raise ValueError(f"Unrecognized napari CSV columns: {list(df.columns)}")


# -----------------------------
# RIGID ALIGNMENT
# -----------------------------
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


# -----------------------------
# DENDRITE GRAPH + DISTANCES
# -----------------------------
def build_dendrite_tree_from_xyz_df(
    full_trace_df: pd.DataFrame,
    branch_col: str = "path",
    coord_cols=("x", "y", "z"),
    max_edge_length=5.0,
    reference_point=None,
):
    """
    Build a dendrite graph from a trace dataframe.
    Nodes are XYZ tuples. Edges connect consecutive points within each branch/path.

    If reference_point is provided, also add a 'center' node by taking the centroid
    of the closest terminal nodes (one per connected component) to reference_point.
    """
    G = nx.Graph()

    if branch_col in full_trace_df.columns:
        groups = full_trace_df.groupby(branch_col)
    else:
        groups = [(0, full_trace_df)]

    for _, group in groups:
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

    center = (771.7282062274531, 777.3587841588766, 172.87154911171154)
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

            G.add_node(center)
            for node in closest_terminal_nodes:
                dist = float(np.linalg.norm(np.asarray(node) - np.asarray(center)))
                G.add_edge(node, center, weight=dist)

    return G, center


def find_nearest_node(coord_xyz, kdtree, node_coords):
    dist, idx = kdtree.query(np.asarray(coord_xyz, dtype=float))
    return tuple(node_coords[idx]), float(dist)


def graph_distance_to_center_from_xyz(coord_xyz, center_xyz, G, kdtree, node_coords):
    nearest_node, _ = find_nearest_node(coord_xyz, kdtree, node_coords)
    try:
        dist = nx.shortest_path_length(G, source=nearest_node, target=tuple(center_xyz), weight="weight")
    except nx.NetworkXNoPath:
        dist = np.inf
    return float(dist)


# -----------------------------
# MAIN
# -----------------------------
# LOAD TRACES
df_ref = pd.read_csv(
    "/Users/amyzheng/Desktop/coronalsectionplot/3562Trace/3562_10xOverview_A01_G003_0001_2_xyzCoordinates.csv"
)
df_mov = pd.read_csv(
    '/Users/amyzheng/Desktop/coronalsectionplot/3562Trace/3562_60x_Cell1_A01_G003_0001_somaapical1_xyzCoordinates.csv'
)

# LANDMARK CSVs (napari format)
A = load_napari_points_as_xyz('/Users/amyzheng/Desktop/coronalsectionplot/apical5ref.csv')  # reference
B = load_napari_points_as_xyz('/Users/amyzheng/Desktop/coronalsectionplot/apical5map.csv')  # moving

# FIT TRANSFORM
R, t = rigid_transform_3d(A, B)

# APPLY TO MOVING TRACE
df_mov_aligned = df_mov.copy()
df_mov_aligned[["x", "y", "z"]] = apply_rigid(df_mov[["x", "y", "z"]].to_numpy(), R, t)

# EXTRA NAPARI POINTS AFTER TRANSFORM (KEEP LABELS)
extra_points_csv = '/Users/amyzheng/Desktop/coronalsectionplot/apical5_inh_nooverlap.csv'
extra_df = load_napari_points_df(extra_points_csv)
extra_xyz = extra_df[["_x", "_y", "_z"]].to_numpy(dtype=float)
# extra_xyz_aligned = apply_rigid(extra_xyz, R, t)
extra_xyz_aligned = extra_xyz
# -----------------------------
# BUILD 10x DENDRITE GRAPH + NEAREST ASSIGNMENTS + DISTANCE TO CENTER
# -----------------------------
REFERENCE_POINT_FOR_CENTER = (771.7282062274531, 777.3587841588766, 172.87154911171154) 
G10x, center10x = build_dendrite_tree_from_xyz_df(
    df_ref,
    branch_col="path" if "path" in df_ref.columns else "__no_path__",
    coord_cols=("x", "y", "z"),
    max_edge_length=5.0,
    reference_point=REFERENCE_POINT_FOR_CENTER,
)

if center10x is None:
    raise RuntimeError(
        "center10x is None. Graph may be empty or no terminal nodes were found. "
        "Try increasing max_edge_length or check df_ref units/columns."
    )

node_coords = np.array(list(G10x.nodes), dtype=float)
kdtree = cKDTree(node_coords)

extra_xyz_arr = np.asarray(extra_xyz_aligned, dtype=float)  # (N,3) aligned XYZ

nearest_nodes = []
nearest_dists = []
dist_to_center = []

for p in extra_xyz_arr:
    nn, d = find_nearest_node(p, kdtree, node_coords)
    nearest_nodes.append(nn)
    nearest_dists.append(d)
    dist_to_center.append(graph_distance_to_center_from_xyz(p, center10x, G10x, kdtree, node_coords))

nearest_nodes = np.array(nearest_nodes, dtype=float)
nearest_dists = np.array(nearest_dists, dtype=float)
dist_to_center = np.array(dist_to_center, dtype=float)

# Build matches table (INCLUDES ORIGINAL NAPARI LABEL)
matches = pd.DataFrame(
    {
        "extra_i": np.arange(len(extra_xyz_arr)),
        "extra_type": extra_df["type"].values if "type" in extra_df.columns else np.array([None] * len(extra_xyz_arr)),
        "extra_x": extra_xyz_arr[:, 0],
        "extra_y": extra_xyz_arr[:, 1],
        "extra_z": extra_xyz_arr[:, 2],
        "nearest10x_x": nearest_nodes[:, 0],
        "nearest10x_y": nearest_nodes[:, 1],
        "nearest10x_z": nearest_nodes[:, 2],
        "euclid_dist_to_graph_node": nearest_dists,
        "dendritic_dist_to_center": dist_to_center,
    }
)
print(matches.head(10))

# Export matches CSV
out_csv = "/Users/amyzheng/Desktop/coronalsectionplot/metrics/apical5inhmetrics.csv"
matches.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# Unique assigned nodes on the 10x dendrite
assigned_nodes_xyz = np.unique(nearest_nodes, axis=0)
print(f"Assigned 10x nodes: {len(assigned_nodes_xyz)} / {len(node_coords)} total")

# Unassigned nodes (for visual contrast)
assigned_set = set(map(tuple, assigned_nodes_xyz))
all_nodes_set = set(map(tuple, node_coords))
unassigned_nodes_xyz = np.array([n for n in all_nodes_set if n not in assigned_set], dtype=float)

# Optional: count how many extra points map to each assigned node
counts = Counter(map(tuple, nearest_nodes))
assigned_sizes = np.array([counts[tuple(n)] for n in assigned_nodes_xyz], dtype=float)

# -----------------------------
# NAPARI VISUALIZATION
# -----------------------------
viewer = napari.Viewer(ndisplay=3)

def rand_color(name: str):
    seed = abs(hash(name)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(3)

# Show 10x trace
if "path" in df_ref.columns:
    for path_name, g in df_ref.groupby("path"):
        pts_zyx = g[["z", "y", "x"]].to_numpy(dtype=float)
        viewer.add_points(
            pts_zyx,
            name=f"10x::{path_name}",
            size=2,
            face_color=rand_color(str(path_name)),
            opacity=0.35,
        )
else:
    pts_zyx = df_ref[["z", "y", "x"]].to_numpy(dtype=float)
    viewer.add_points(
        pts_zyx,
        name="10x::all",
        size=2,
        face_color=[0.7, 0.7, 0.7],
        opacity=0.35,
    )

# Show aligned moving trace
if "path" in df_mov_aligned.columns:
    for path_name, g in df_mov_aligned.groupby("path"):
        pts_zyx = g[["z", "y", "x"]].to_numpy(dtype=float)
        viewer.add_points(
            pts_zyx,
            name=f"60x_aligned::{path_name}",
            size=3,
            face_color=rand_color("mov_" + str(path_name)),
            opacity=0.9,
        )
else:
    pts_zyx = df_mov_aligned[["z", "y", "x"]].to_numpy(dtype=float)
    viewer.add_points(
        pts_zyx,
        name="60x_aligned::all",
        size=3,
        face_color=[0.2, 0.6, 1.0],
        opacity=0.9,
    )

# Extra points (aligned)
viewer.add_points(
    extra_xyz_arr[:, [2, 1, 0]],  # XYZ -> ZYX
    name="extra_points_aligned",
    size=8,
    face_color=[1.0, 0.2, 0.2],
    opacity=1.0,
)

# Center node
viewer.add_points(
    np.array(center10x, dtype=float)[None, [2, 1, 0]],
    name="10x_center_node",
    size=12,
    face_color=[0.0, 1.0, 1.0],
    opacity=1.0,
)

# Graph nodes colored by assignment status
assigned_nodes_zyx = assigned_nodes_xyz[:, [2, 1, 0]]
unassigned_nodes_zyx = unassigned_nodes_xyz[:, [2, 1, 0]]

if len(unassigned_nodes_zyx) > 0:
    viewer.add_points(
        unassigned_nodes_zyx,
        name="10x_graph_nodes_unassigned",
        size=1.5,
        face_color=[0.6, 0.6, 0.6],
        opacity=0.15,
    )

viewer.add_points(
    assigned_nodes_zyx,
    name="10x_graph_nodes_assigned",
    size=4,
    face_color=[1.0, 0.85, 0.1],
    opacity=1.0,
)

viewer.add_points(
    assigned_nodes_zyx,
    name="10x_graph_nodes_assigned_weighted",
    size=2 + 1.5 * assigned_sizes,
    face_color=[1.0, 0.4, 0.0],
    opacity=0.9,
)

# Vectors from each extra point to its nearest 10x graph node
extra_zyx = extra_xyz_arr[:, [2, 1, 0]]
nearest_zyx = nearest_nodes[:, [2, 1, 0]]

vecs = np.zeros((len(extra_zyx), 2, 3), dtype=float)
vecs[:, 0, :] = extra_zyx
vecs[:, 1, :] = nearest_zyx - extra_zyx

viewer.add_vectors(
    vecs,
    name="extra_to_nearest10x_vectors",
    edge_width=2,
    opacity=0.7,
)

napari.run()

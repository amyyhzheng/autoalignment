"""
Manual branch-order segments by INDEXING into branchpoints.csv

You will define order segments explicitly as pairs of branchpoint indices:
    order 0: bp[i0] -> bp[i1]
    order 1: bp[i1] -> bp[i2]
    ...

Then:
- build dendrite graph from TRACE_CSV
- build "analyzed_edges" from analyzed.csv (coverage denominator)
- for each order segment, compute shortest-path edges between endpoints
- intersect with analyzed_edges to get analyzed length per order
- assign synapses to nearest order-segment path (by distance to path nodes)
- compute density + plots
- optional napari visualization (one layer per order)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from collections import defaultdict
from matplotlib import colormaps
import matplotlib.colors as mcolors

# -----------------------------
# USER SETTINGS
# -----------------------------
FOLDER = '/Users/amyzheng/Desktop/coronalsectionplot/1317_analysis_Jan2026/metrics'
ANALYZED_BASENAME = "analyzed.csv"

DIST_COL = "dendritic_dist_to_center"
TYPE_COL = "extra_type"
TYPES_TO_INCLUDE = [
    "Shaft_Geph+Bsn_+SynTd",
    "Spine_Geph+Bsn_+SynTd",
    # "Shaft_Geph+Bsn_NoSynTd",
    # "Spine_Geph+Bsn_NoSynTd",    
]
DIST_SCALE = 0.657

TRACE_CSV = '/Users/amyzheng/Desktop/coronalsectionplot/1317_analysis_Jan2026/1317Analysis/2024-8-30_session2_xyzCoordinates.csv'
BRANCHPOINTS_CSV = '/Users/amyzheng/Desktop/coronalsectionplot/1317_analysis_Jan2026/branchpointsnew3.csv'

XCOL, YCOL, ZCOL = "extra_x", "extra_y", "extra_z"
MAX_EDGE_LENGTH = 5.0

# If you want order_0 to start at soma, include soma as an extra "branchpoint" yourself:
# e.g. define SOMA_XYZ and use it as the start of order_0 (see ORDER_SEGMENTS section).
SOMA_XYZ = (771.7282062274531, 777.3587841588766, 172.87154911171154)

# -----------------------------
# MANUAL ORDER SEGMENTS (EDIT THIS)
# -----------------------------
# Option A (recommended): use branchpoint indices directly.
# Example (you will edit indices):
#   order 0: bp[0] -> bp[3]
#   order 1: bp[3] -> bp[7]
#   order 2: bp[7] -> bp[10]
ORDER_SEGMENTS = [
    # (order, start_bp_index, end_bp_index)
    # (0, 4, 0),
    (1, 0, 0),
    (2, 0, 1),
    (3, 1, 2),
    (4, 2, 3),

]

# Option B: use soma as start for order 0:
# ORDER_SEGMENTS = [
#     (0, "SOMA", 0),   # SOMA_XYZ -> bp[0]
#     (1, 0, 1),
#     (2, 1, 2),
# ]

# -----------------------------
# VIS STYLE
# -----------------------------
CYAN = "#00FFFF"
CYAN_SOFT = "#66FFFF"

def black_bg(ax):
    ax.set_facecolor("black")
    ax.figure.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color="white", alpha=0.15, linewidth=0.6)

# -----------------------------
# HELPERS: LOAD NAPARI POINTS
# -----------------------------
def load_napari_points_as_xyz(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    if "z" in cols_lower and "y" in cols_lower and "x" in cols_lower:
        zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
        ZYX = df[[zc, yc, xc]].to_numpy(dtype=float)
        return ZYX[:, [2, 1, 0]]  # -> XYZ

    axis_cols = [k for k in cols_lower.keys() if k.startswith("axis-")]
    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))
        A012 = df[[c0, c1, c2]].to_numpy(dtype=float)  # napari order (z,y,x)
        return A012[:, [2, 1, 0]]  # -> XYZ

    raise ValueError(f"Unrecognized napari CSV columns: {list(df.columns)}")

def print_branchpoints_with_indices(branch_xyz: np.ndarray, n=50):
    n = min(n, len(branch_xyz))
    rows = []
    for i in range(n):
        x, y, z = branch_xyz[i]
        rows.append({"idx": i, "x": x, "y": y, "z": z})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

# -----------------------------
# YOUR EXISTING COVERAGE LOGIC
# -----------------------------
def load_points_csv(path, dist_col=DIST_COL, dist_scale=DIST_SCALE):
    df = pd.read_csv(path)
    if dist_col not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing required column '{dist_col}'")
    df = df.copy()
    df[dist_col] = pd.to_numeric(df[dist_col], errors="coerce")
    df[dist_col] = df[dist_col] * dist_scale
    df = df[np.isfinite(df[dist_col]) & (df[dist_col] >= 0)].copy()
    return df

def intervals_from_analyzed_df(analyzed_df, dist_col=DIST_COL):
    s = analyzed_df[dist_col].to_numpy(dtype=float)
    if len(s) < 2:
        raise ValueError("analyzed.csv must have at least 2 points (one segment).")
    if len(s) % 2 != 0:
        raise ValueError("analyzed.csv must have an even number of rows (pairs of endpoints).")

    intervals = []
    for i in range(0, len(s), 2):
        a, b = float(s[i]), float(s[i + 1])
        lo, hi = (a, b) if a <= b else (b, a)
        if hi > lo:
            intervals.append((lo, hi))

    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)

    return [(float(a), float(b)) for a, b in merged]

def filter_points_in_intervals(s_vals, intervals):
    s = np.asarray(s_vals, float)
    mask = np.zeros(len(s), dtype=bool)
    for a, b in intervals:
        mask |= (s >= a) & (s <= b)
    return mask

# -----------------------------
# BUILD GRAPH FROM TRACE
# -----------------------------
def build_dendrite_graph(trace_df, branch_col="path", coord_cols=("x","y","z"), max_edge_length=5.0):
    G = nx.Graph()
    if branch_col in trace_df.columns:
        groups = trace_df.groupby(branch_col)
    else:
        groups = [(0, trace_df)]

    for _, g in groups:
        coords = g.loc[:, coord_cols].dropna().to_numpy(dtype=float)
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            p1 = tuple(coords[i])
            p2 = tuple(coords[i + 1])
            if p1 == p2:
                continue
            w = float(np.linalg.norm(coords[i + 1] - coords[i]))
            if w <= max_edge_length:
                G.add_edge(p1, p2, weight=w)
    return G

def snap_points_to_nodes(points_xyz: np.ndarray, kdtree: cKDTree, node_coords: np.ndarray):
    dists, idx = kdtree.query(points_xyz.astype(float))
    snapped = [tuple(node_coords[i]) for i in idx]
    return snapped, dists

# -----------------------------
# ANALYZED EDGES (coverage)
# -----------------------------
def analyzed_endpoint_pairs_from_xyz(analyzed_df, x="extra_x", y="extra_y", z="extra_z"):
    P = analyzed_df[[x, y, z]].to_numpy(dtype=float)
    if len(P) < 2 or len(P) % 2 != 0:
        raise ValueError("analyzed.csv XYZ must have even number of rows >=2.")
    return [(P[i], P[i + 1]) for i in range(0, len(P), 2)]

def mark_analyzed_edges_from_pairs(G, endpoint_pairs_xyz, kdtree, node_coords):
    analyzed_edges = set()
    for a_xyz, b_xyz in endpoint_pairs_xyz:
        (a_node,), _ = snap_points_to_nodes(np.array([a_xyz]), kdtree, node_coords)
        (b_node,), _ = snap_points_to_nodes(np.array([b_xyz]), kdtree, node_coords)
        try:
            path = nx.shortest_path(G, a_node, b_node, weight="weight")
        except nx.NetworkXNoPath:
            continue
        for u, v in zip(path[:-1], path[1:]):
            analyzed_edges.add(frozenset((u, v)))
    return analyzed_edges

def edges_from_path(path_nodes):
    return [frozenset((u, v)) for u, v in zip(path_nodes[:-1], path_nodes[1:])]

def length_of_edge_set(G, edge_set):
    L = 0.0
    for e in edge_set:
        u, v = tuple(e)
        L += float(G[u][v].get("weight", 0.0))
    return float(L)

# -----------------------------
# MANUAL ORDER SEGMENTS -> PATHS/EDGES
# -----------------------------
def get_segment_endpoints_xyz(branch_xyz, seg, soma_xyz=None):
    """
    seg is (order, start_idx_or_SOMA, end_idx)
    """
    order, a, b = seg
    if isinstance(a, str) and a.upper() == "SOMA":
        if soma_xyz is None:
            raise ValueError("Segment uses 'SOMA' but soma_xyz is None.")
        a_xyz = np.asarray(soma_xyz, float)
    else:
        a_xyz = np.asarray(branch_xyz[int(a)], float)
    b_xyz = np.asarray(branch_xyz[int(b)], float)
    return order, a_xyz, b_xyz

def compute_order_paths(G, branch_xyz, order_segments, kdtree, node_coords, soma_xyz=None):
    """
    Returns:
      order_to_path_nodes: order -> list of graph nodes along shortest path
      order_to_edges: order -> set of frozenset edges along path
    """
    order_to_path_nodes = {}
    order_to_edges = {}

    for seg in order_segments:
        order, a_xyz, b_xyz = get_segment_endpoints_xyz(branch_xyz, seg, soma_xyz=soma_xyz)
        (a_node,), _ = snap_points_to_nodes(np.array([a_xyz]), kdtree, node_coords)
        (b_node,), _ = snap_points_to_nodes(np.array([b_xyz]), kdtree, node_coords)

        try:
            path_nodes = nx.shortest_path(G, a_node, b_node, weight="weight")
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between endpoints for order={order}. Check your indices / graph connectivity.")

        order_to_path_nodes[int(order)] = path_nodes
        order_to_edges[int(order)] = set(edges_from_path(path_nodes))

    return order_to_path_nodes, order_to_edges

# -----------------------------
# SYNAPSE ASSIGNMENT: nearest order-path nodes
# -----------------------------
def assign_synapses_to_orders_by_nearest_path(syn_xyz, order_to_path_nodes):
    """
    Assign each synapse to the order whose path nodes are closest (Euclidean).
    Returns:
      syn_order: (N,) int
      syn_dist:  (N,) float distance to chosen path
    """
    orders = sorted(order_to_path_nodes.keys())
    if not orders:
        raise ValueError("No orders defined.")

    # build KDTree per order over its path nodes
    kdtrees = {}
    for o in orders:
        P = np.array(order_to_path_nodes[o], dtype=float)  # XYZ nodes
        kdtrees[o] = cKDTree(P)

    syn_order = np.full((syn_xyz.shape[0],), -1, dtype=int)
    syn_dist = np.full((syn_xyz.shape[0],), np.inf, dtype=float)

    for o in orders:
        d, _ = kdtrees[o].query(syn_xyz.astype(float))
        better = d < syn_dist
        syn_dist[better] = d[better]
        syn_order[better] = o

    return syn_order, syn_dist

# ============================================================
# A) LOAD ANALYZED INTERVALS (distance-domain filter)
# ============================================================
analyzed_path = os.path.join(FOLDER, ANALYZED_BASENAME)
analyzed_df = load_points_csv(analyzed_path, dist_col=DIST_COL)
analyzed_intervals = intervals_from_analyzed_df(analyzed_df, dist_col=DIST_COL)

# ============================================================
# B) READ + FILTER SYNAPSES (your existing strategy)
# ============================================================
paths = sorted(glob.glob(os.path.join(FOLDER, "*.csv")))
paths = [p for p in paths if os.path.basename(p) != ANALYZED_BASENAME]

dfs = []
for p in paths:
    try:
        df = load_points_csv(p, dist_col=DIST_COL)
    except Exception as e:
        print(f"Skipping {os.path.basename(p)}: {e}")
        continue
    df["source_file"] = os.path.basename(p)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

if TYPES_TO_INCLUDE is not None:
    all_df = all_df[all_df[TYPE_COL].isin(TYPES_TO_INCLUDE)].copy()

mask_cov = filter_points_in_intervals(all_df[DIST_COL].to_numpy(float), analyzed_intervals)
all_df = all_df.loc[mask_cov].copy()

# ============================================================
# C) BUILD GRAPH + KDTree over graph nodes
# ============================================================
trace_df = pd.read_csv(TRACE_CSV)
G = build_dendrite_graph(trace_df, branch_col="path", coord_cols=("x", "y", "z"), max_edge_length=MAX_EDGE_LENGTH)

node_coords = np.array(list(G.nodes), dtype=float)
kdtree_nodes = cKDTree(node_coords)

# ============================================================
# D) LOAD BRANCHPOINTS (and print indices so you can choose)
# ============================================================
branch_xyz = load_napari_points_as_xyz(BRANCHPOINTS_CSV)
print(f"Loaded {len(branch_xyz)} branchpoints from {BRANCHPOINTS_CSV}")
_ = print_branchpoints_with_indices(branch_xyz, n=min(60, len(branch_xyz)))

# ============================================================
# E) BUILD ANALYZED EDGES (coverage in XYZ space)
# ============================================================
if not all(c in analyzed_df.columns for c in ["extra_x", "extra_y", "extra_z"]):
    raise ValueError("analyzed.csv missing extra_x/extra_y/extra_z (needed for analyzed-edge mapping).")

endpoint_pairs = analyzed_endpoint_pairs_from_xyz(analyzed_df, x="extra_x", y="extra_y", z="extra_z")
analyzed_edges = mark_analyzed_edges_from_pairs(G, endpoint_pairs, kdtree_nodes, node_coords)

print(f"Analyzed edges (coverage) count: {len(analyzed_edges)}")

# ============================================================
# F) MANUAL ORDER SEGMENTS -> ORDER PATHS
# ============================================================
order_to_path_nodes, order_to_edges = compute_order_paths(
    G=G,
    branch_xyz=branch_xyz,
    order_segments=ORDER_SEGMENTS,
    kdtree=kdtree_nodes,
    node_coords=node_coords,
    soma_xyz=SOMA_XYZ,
)

print("Orders defined:", sorted(order_to_edges.keys()))
for o in sorted(order_to_edges.keys()):
    print(f"order {o}: path nodes={len(order_to_path_nodes[o])}, path edges={len(order_to_edges[o])}")

# ============================================================
# G) DENOMINATOR: analyzed length per order (intersection with analyzed_edges)
# ============================================================
L = {}
for o, path_edges in order_to_edges.items():
    covered_edges = path_edges.intersection(analyzed_edges)
    L[o] = length_of_edge_set(G, covered_edges)

print("Analyzed length per order:")
for o in sorted(L.keys()):
    print(f"  order {o}: {L[o]:.3f}")

# ============================================================
# H) NUMERATOR: synapses per order (nearest path assignment)
# ============================================================
syn_xyz = all_df[[XCOL, YCOL, ZCOL]].to_numpy(dtype=float)
syn_order, syn_dist = assign_synapses_to_orders_by_nearest_path(syn_xyz, order_to_path_nodes)

# optional: warn if snapping looks bad
print("Synapse->path distance (min/median/max):",
      float(np.min(syn_dist)), float(np.median(syn_dist)), float(np.max(syn_dist)))

C = pd.Series(syn_order).value_counts().to_dict()

orders = sorted(set(L.keys()) | set(C.keys()))
out = pd.DataFrame({
    "branch_order": orders,
    "analyzed_length": [L.get(o, 0.0) for o in orders],
    "synapse_count": [int(C.get(o, 0)) for o in orders],
})
out["density"] = out["synapse_count"] / out["analyzed_length"].replace(0, np.nan)

out_csv = os.path.join(FOLDER, "density_by_branch_order_manual_segments.csv")
out.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")
print(out)

# ============================================================
# I) PLOTS (all)
# ============================================================
out = out.sort_values("branch_order").reset_index(drop=True)
x = out["branch_order"].to_numpy(int)
dens = out["density"].to_numpy(float)
counts = out["synapse_count"].to_numpy(float)
lengths = out["analyzed_length"].to_numpy(float)

# 1) Density
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, dens, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order (manual segments)")
ax.set_ylabel("Density (synapses / analyzed length)")
ax.set_title("Synapse density by branch order")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# 2) Counts
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, counts, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order (manual segments)")
ax.set_ylabel("Synapse count")
ax.set_title("Synapse count by branch order")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# 3) Analyzed lengths
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, lengths, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order (manual segments)")
ax.set_ylabel("Analyzed length (XYZ units)")
ax.set_title("Analyzed length by branch order (coverage intersected)")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# 4) Density annotated with n and L
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, dens, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order (manual segments)")
ax.set_ylabel("Density (synapses / analyzed length)")
ax.set_title("Density by branch order (labels show n and L)")
ax.set_xticks(x)

for b, n, Lval in zip(bars, counts.astype(int), lengths):
    h = b.get_height()
    if np.isfinite(h) and h > 0:
        ax.text(
            b.get_x() + b.get_width()/2,
            h,
            f"n={n}\nL={Lval:.1f}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
        )

black_bg(ax)
plt.tight_layout()
plt.show()

# ============================================================
# J) OPTIONAL: NAPARI visualization (one layer per order)
# ============================================================
DO_NAPARI = True
if DO_NAPARI:
    import napari

    v = napari.Viewer(ndisplay=3)

    # synapses (ZYX)
    syn_zyx = syn_xyz[:, [2, 1, 0]]
    v.add_points(syn_zyx, name="synapses_filtered", size=4, face_color="red", opacity=0.8)

    # branchpoints (ZYX) + show indices via properties
    bp_zyx = branch_xyz[:, [2, 1, 0]]
    v.add_points(bp_zyx, name="branchpoints", size=6, face_color="yellow",
                 properties={"idx": np.arange(len(branch_xyz), dtype=int)})
    v.layers["branchpoints"].show_table = True

    # order paths: one Shapes line layer per order
    cmap = colormaps.get_cmap("viridis")
    o_list = sorted(order_to_path_nodes.keys())
    vmin = min(o_list)
    vmax = max(o_list) if max(o_list) > min(o_list) else min(o_list) + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for o in o_list:
        path_nodes = np.array(order_to_path_nodes[o], dtype=float)  # XYZ nodes
        # convert to a list of 2-pt line segments in ZYX for napari Shapes
        segs = []
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            a_zyx = a[[2, 1, 0]]
            b_zyx = b[[2, 1, 0]]
            segs.append(np.stack([a_zyx, b_zyx], axis=0))

        color_hex = mcolors.to_hex(cmap(norm(o)), keep_alpha=True)
        v.add_shapes(
            segs,
            shape_type="line",
            name=f"order_{o}",
            edge_color=color_hex,
            edge_width=1.0,
            opacity=0.9,
        )

    napari.run()


# import os, glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# from scipy.spatial import cKDTree
# from collections import defaultdict, deque

# # -----------------------------
# # USER SETTINGS
# # -----------------------------
# FOLDER = '/Users/amyzheng/Desktop/coronalsectionplot/1317_analysis_Jan2026/metrics'
# ANALYZED_BASENAME = "analyzed.csv"

# DIST_COL = "dendritic_dist_to_center"
# TYPE_COL = "extra_type"
# TYPES_TO_INCLUDE = [
#     # "Shaft_Geph+Bsn_NoSynTd",
#     "Shaft_Geph+Bsn_+SynTd",
#     # "Spine_Geph+Bsn_NoSynTd",
#     "Spine_Geph+Bsn_+SynTd",
# ]
# DIST_SCALE = 0.657

# # dendrite + branchpoints inputs
# TRACE_CSV = '/Users/amyzheng/Desktop/coronalsectionplot/1317_analysis_Jan2026/1317Analysis/2024-8-30_session2_xyzCoordinates.csv' # MUST have x,y,z and ideally path
# BRANCHPOINTS_CSV = '/Users/amyzheng/Desktop/coronalsectionplot/1317_analysis_Jan2026/branchpoints.csv'      # napari export

# # synapse XYZ columns in your metrics CSVs
# XCOL, YCOL, ZCOL = "extra_x", "extra_y", "extra_z"

# # graph construction
# MAX_EDGE_LENGTH = 5.0  # same idea as you used
# ROOT_REFERENCE_POINT = (771.7282062274531, 777.3587841588766, 172.87154911171154)  # set to soma xyz if you want; else we will pick a reasonable root

# # -----------------------------
# # VIS STYLE
# # -----------------------------
# CYAN ="#00FFFF"
# CYAN_SOFT ="#66FFFF"

# def black_bg(ax):
#     ax.set_facecolor("black")
#     ax.figure.set_facecolor("black")
#     for spine in ax.spines.values():
#         spine.set_color("white")
#     ax.tick_params(colors="white", which="both")
#     ax.xaxis.label.set_color("white")
#     ax.yaxis.label.set_color("white")
#     ax.title.set_color("white")
#     ax.grid(True, color="white", alpha=0.15, linewidth=0.6)

# # -----------------------------
# # HELPERS: LOAD NAPARI POINTS
# # -----------------------------
# def load_napari_points_as_xyz(csv_path: str) -> np.ndarray:
#     df = pd.read_csv(csv_path)
#     cols_lower = {c.lower(): c for c in df.columns}

#     if "z" in cols_lower and "y" in cols_lower and "x" in cols_lower:
#         zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
#         ZYX = df[[zc, yc, xc]].to_numpy(dtype=float)
#         return ZYX[:, [2, 1, 0]]  # -> XYZ

#     axis_cols = [k for k in cols_lower.keys() if k.startswith("axis-")]
#     if len(axis_cols) >= 3:
#         axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
#         c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))
#         A012 = df[[c0, c1, c2]].to_numpy(dtype=float)  # napari order (z,y,x)
#         return A012[:, [2, 1, 0]]  # -> XYZ

#     raise ValueError(f"Unrecognized napari CSV columns: {list(df.columns)}")

# # -----------------------------
# # YOUR EXISTING COVERAGE LOGIC
# # -----------------------------
# def load_points_csv(path, dist_col=DIST_COL, dist_scale=DIST_SCALE):
#     df = pd.read_csv(path)
#     if dist_col not in df.columns:
#         raise ValueError(f"{os.path.basename(path)} missing required column '{dist_col}'")
#     df = df.copy()
#     df[dist_col] = pd.to_numeric(df[dist_col], errors="coerce")
#     df[dist_col] = df[dist_col] * dist_scale
#     df = df[np.isfinite(df[dist_col]) & (df[dist_col] >= 0)].copy()
#     return df

# def intervals_from_analyzed_df(analyzed_df, dist_col=DIST_COL):
#     s = analyzed_df[dist_col].to_numpy(dtype=float)
#     if len(s) < 2:
#         raise ValueError("analyzed.csv must have at least 2 points (one segment).")
#     if len(s) % 2 != 0:
#         raise ValueError("analyzed.csv must have an even number of rows (pairs of endpoints).")

#     intervals = []
#     for i in range(0, len(s), 2):
#         a, b = float(s[i]), float(s[i + 1])
#         lo, hi = (a, b) if a <= b else (b, a)
#         if hi > lo:
#             intervals.append((lo, hi))

#     intervals.sort()
#     merged = []
#     for lo, hi in intervals:
#         if not merged or lo > merged[-1][1]:
#             merged.append([lo, hi])
#         else:
#             merged[-1][1] = max(merged[-1][1], hi)

#     return [(float(a), float(b)) for a, b in merged]

# def filter_points_in_intervals(s_vals, intervals):
#     s = np.asarray(s_vals, float)
#     mask = np.zeros(len(s), dtype=bool)
#     for a, b in intervals:
#         mask |= (s >= a) & (s <= b)
#     return mask

# # -----------------------------
# # NEW: BUILD GRAPH FROM TRACE
# # -----------------------------
# def build_dendrite_graph(trace_df, branch_col="path", coord_cols=("x","y","z"), max_edge_length=5.0):
#     G = nx.Graph()
#     if branch_col in trace_df.columns:
#         groups = trace_df.groupby(branch_col)
#     else:
#         groups = [(0, trace_df)]

#     for _, g in groups:
#         coords = g.loc[:, coord_cols].dropna().to_numpy(dtype=float)
#         if len(coords) < 2:
#             continue
#         for i in range(len(coords)-1):
#             p1 = tuple(coords[i])
#             p2 = tuple(coords[i+1])
#             if p1 == p2:
#                 continue
#             w = float(np.linalg.norm(coords[i+1] - coords[i]))
#             if w <= max_edge_length:
#                 G.add_edge(p1, p2, weight=w)
#     return G

# def pick_root_node(G, reference_point=None):
#     if reference_point is None:
#         # heuristic: pick a high-degree node if exists, else arbitrary
#         deg = dict(G.degree)
#         return max(deg, key=deg.get)
#     # snap reference to nearest node by brute (graph is usually manageable)
#     ref = np.asarray(reference_point, float)
#     nodes = np.array(list(G.nodes), float)
#     idx = np.argmin(np.linalg.norm(nodes - ref[None,:], axis=1))
#     return tuple(nodes[idx])

# # -----------------------------
# # NEW: SNAP + ROOT TREE + ORDERS
# # -----------------------------
# def snap_points_to_nodes(points_xyz: np.ndarray, kdtree: cKDTree, node_coords: np.ndarray):
#     dists, idx = kdtree.query(points_xyz.astype(float))
#     snapped = [tuple(node_coords[i]) for i in idx]
#     return snapped, dists

# def root_graph_bfs_tree(G: nx.Graph, root):
#     root = tuple(root)
#     parent = {root: None}
#     children = defaultdict(list)
#     q = deque([root])
#     while q:
#         u = q.popleft()
#         for v in G.neighbors(u):
#             if v in parent:
#                 continue
#             parent[v] = u
#             children[u].append(v)
#             q.append(v)
#     return parent, children

# def compute_centrifugal_order(G, root, branch_nodes):
#     parent, children = root_graph_bfs_tree(G, root)
#     node_order = {root: 0}
#     edge_order = {}
#     q = deque([root])
#     while q:
#         u = q.popleft()
#         for v in children[u]:
#             incr = 1 if (u in branch_nodes) else 0
#             node_order[v] = node_order[u] + incr
#             edge_order[frozenset((u,v))] = node_order[v]
#             q.append(v)
#     return node_order, edge_order
# import numpy as np
# import networkx as nx
# from collections import defaultdict, deque

# def root_graph_dijkstra_tree(G: nx.Graph, root):
#     """
#     Parent pointers from Dijkstra shortest paths (weighted).
#     Produces a rooted tree (actually a shortest-path arborescence).
#     """
#     lengths, paths = nx.single_source_dijkstra(G, root, weight="weight")
#     parent = {root: None}
#     children = defaultdict(list)

#     for node, path in paths.items():
#         if node == root:
#             continue
#         p = path[-2]
#         parent[node] = p
#         children[p].append(node)

#     return parent, children

# def compute_centrifugal_order_dijkstra(G, root, branch_nodes_set):
#     parent, children = root_graph_dijkstra_tree(G, root)

#     node_order = {root: 0}
#     edge_order = {}

#     q = deque([root])
#     while q:
#         u = q.popleft()
#         for v in children[u]:
#             # increment when crossing a branchpoint node
#             incr = 1 if (u in branch_nodes_set) else 0
#             node_order[v] = node_order[u] + incr
#             edge_order[frozenset((u, v))] = node_order[v]
#             q.append(v)

#     return node_order, edge_order, parent, children

# # -----------------------------
# # NEW: ANALYZED LENGTH PER ORDER
# # Requires analyzed.csv has XYZ endpoints (recommended).
# # -----------------------------
# def analyzed_endpoint_pairs_from_xyz(analyzed_df, x="x", y="y", z="z"):
#     P = analyzed_df[[x,y,z]].to_numpy(dtype=float)
#     if len(P) < 2 or len(P) % 2 != 0:
#         raise ValueError("analyzed.csv XYZ must have even number of rows >=2.")
#     return [(P[i], P[i+1]) for i in range(0, len(P), 2)]

# def mark_analyzed_edges_from_pairs(G, endpoint_pairs_xyz, kdtree, node_coords):
#     analyzed_edges = set()
#     for a_xyz, b_xyz in endpoint_pairs_xyz:
#         (a_node,), _ = snap_points_to_nodes(np.array([a_xyz]), kdtree, node_coords)
#         (b_node,), _ = snap_points_to_nodes(np.array([b_xyz]), kdtree, node_coords)
#         try:
#             path = nx.shortest_path(G, a_node, b_node, weight="weight")
#         except nx.NetworkXNoPath:
#             continue
#         for u,v in zip(path[:-1], path[1:]):
#             analyzed_edges.add(frozenset((u,v)))
#     return analyzed_edges

# def analyzed_length_per_order(G, analyzed_edges, edge_order):
#     L = defaultdict(float)
#     for e in analyzed_edges:
#         u,v = tuple(e)
#         w = float(G[u][v].get("weight", 0.0))
#         o = edge_order.get(e, None)
#         if o is not None:
#             L[o] += w
#     return dict(L)

# # -----------------------------
# # NEW: SYNAPSE COUNT PER ORDER (after your filtering)
# # -----------------------------
# def synapse_count_per_order(syn_xyz, kdtree, node_coords, node_order):
#     snapped_nodes, _ = snap_points_to_nodes(syn_xyz, kdtree, node_coords)
#     C = defaultdict(int)
#     for n in snapped_nodes:
#         o = node_order.get(n, None)
#         if o is None or o == 0:
#             continue
#         C[o] += 1
#     return dict(C)

# # ============================================================
# # STEP A: LOAD ANALYZED INTERVALS (for synapse filtering)
# # ============================================================
# analyzed_path = os.path.join(FOLDER, ANALYZED_BASENAME)
# analyzed_df = load_points_csv(analyzed_path, dist_col=DIST_COL)
# analyzed_intervals = intervals_from_analyzed_df(analyzed_df, dist_col=DIST_COL)

# # ============================================================
# # STEP B: READ ALL SYNAPSES (your strategy, unchanged)
# # ============================================================
# paths = sorted(glob.glob(os.path.join(FOLDER, "*.csv")))
# paths = [p for p in paths if os.path.basename(p) != ANALYZED_BASENAME]

# dfs = []
# for p in paths:
#     try:
#         df = load_points_csv(p, dist_col=DIST_COL)
#     except Exception as e:
#         print(f"Skipping {os.path.basename(p)}: {e}")
#         continue
#     df["source_file"] = os.path.basename(p)
#     dfs.append(df)

# all_df = pd.concat(dfs, ignore_index=True)

# # type filter
# if TYPES_TO_INCLUDE is not None:
#     all_df = all_df[all_df[TYPE_COL].isin(TYPES_TO_INCLUDE)].copy()

# # coverage filter by distance intervals (same as now)
# mask_cov = filter_points_in_intervals(all_df[DIST_COL].to_numpy(float), analyzed_intervals)
# all_df = all_df.loc[mask_cov].copy()

# # ============================================================
# # STEP C: BUILD GRAPH + ORDERS
# # ============================================================
# trace_df = pd.read_csv(TRACE_CSV)
# G = build_dendrite_graph(trace_df, branch_col="path", coord_cols=("x","y","z"), max_edge_length=MAX_EDGE_LENGTH)

# root = pick_root_node(G, reference_point=ROOT_REFERENCE_POINT)

# node_coords = np.array(list(G.nodes), dtype=float)
# kdtree = cKDTree(node_coords)

# # branchpoints snapped to nodes
# branch_xyz = load_napari_points_as_xyz(BRANCHPOINTS_CSV)
# branch_nodes, _ = snap_points_to_nodes(branch_xyz, kdtree, node_coords)
# branch_nodes = set(branch_nodes)

# # also include all degree>=3 as branch nodes (optional but usually helpful)
# for n in G.nodes:
#     if G.degree[n] >= 3:
#         branch_nodes.add(n)
# node_order, edge_order, parent, children = compute_centrifugal_order_dijkstra(G, root, branch_nodes)

# # ============================================================
# # STEP D: ANALYZED LENGTH PER ORDER (denominator)
# # ============================================================
# # This requires analyzed.csv has x,y,z columns in the SAME space as TRACE_CSV.
# # If your analyzed.csv does not have x/y/z, add them when you export analyzed segments.
# if all(c in analyzed_df.columns for c in ["extra_x","extra_y","extra_z"]):
#     endpoint_pairs = analyzed_endpoint_pairs_from_xyz(analyzed_df, x="extra_x", y="extra_y", z="extra_z")
#     analyzed_edges = mark_analyzed_edges_from_pairs(G, endpoint_pairs, kdtree, node_coords)
#     L = analyzed_length_per_order(G, analyzed_edges, edge_order)
# else:
#     raise ValueError("analyzed.csv is missing x,y,z. Needed to compute analyzed length per branch order.")

# # ============================================================
# # STEP E: SYNAPSE COUNTS PER ORDER (numerator)
# # ============================================================
# syn_xyz = all_df[[XCOL, YCOL, ZCOL]].to_numpy(dtype=float)
# C = synapse_count_per_order(syn_xyz, kdtree, node_coords, node_order)

# orders = sorted(set(L.keys()) | set(C.keys()))
# out = pd.DataFrame({
#     "branch_order": orders,
#     "analyzed_length": [L.get(o, 0.0) for o in orders],
#     "synapse_count": [C.get(o, 0) for o in orders],
# })
# out["density"] = out["synapse_count"] / out["analyzed_length"].replace(0, np.nan)

# out_csv = os.path.join(FOLDER, "density_by_branch_order.csv")
# out.to_csv(out_csv, index=False)
# print(f"Saved {out_csv}")
# print(out)

# # ============================================================
# # STEP F: PLOT
# # ============================================================
# fig, ax = plt.subplots(figsize=(7,4))
# ax.bar(out["branch_order"], out["density"], color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
# ax.set_xlabel("Branch order (centrifugal)")
# ax.set_ylabel("Density (synapses / analyzed length)")
# ax.set_title("Branch-order density (coverage + type filtered)")
# black_bg(ax)
# plt.tight_layout()
# plt.show()

# import numpy as np
# import napari

# import numpy as np
# import napari
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors

# def visualize_branch_orders_in_napari(
#     viewer,
#     G,
#     node_order: dict,
#     edge_order: dict,
#     branch_nodes: set,
#     root,
#     show_edges: bool = True,
#     node_size: float = 2.0,
#     edge_width: float = 0.6,
#     edge_opacity: float = 0.7,
#     cmap_name: str = "viridis",
# ):
#     # -------- nodes --------
#     nodes = list(G.nodes)
#     node_xyz = np.array(nodes, dtype=float)          # (N,3) XYZ
#     node_zyx = node_xyz[:, [2, 1, 0]]                # napari wants ZYX

#     orders = np.array([node_order.get(n, -1) for n in nodes], dtype=int)
#     degrees = np.array([G.degree[n] for n in nodes], dtype=int)
#     is_bp = np.array([n in branch_nodes for n in nodes], dtype=bool)
#     is_root = np.array([tuple(n) == tuple(root) for n in nodes], dtype=bool)

#     pts = viewer.add_points(
#         node_zyx,
#         name="dendrite_nodes",
#         size=node_size,
#         properties={
#             "branch_order": orders,
#             "degree": degrees,
#             "is_branchpoint": is_bp,
#             "is_root": is_root,
#         },
#     )
#     pts.face_color = "branch_order"
#     pts.face_colormap = cmap_name
#     pts.face_contrast_limits = (int(orders.min()), int(orders.max()))
#     pts.show_table = True

#     # -------- edges (colored) --------
#     if show_edges:
#         # colormap for edge orders
#         eo_vals = [edge_order.get(frozenset((u, v)), -1) for (u, v) in G.edges]
#         eo_vals = np.array(eo_vals, dtype=int)
#         eo_min, eo_max = int(eo_vals.min()), int(eo_vals.max())

#         norm = mcolors.Normalize(vmin=eo_min, vmax=eo_max if eo_max > eo_min else eo_min + 1)
#         cmap = cm.get_cmap(cmap_name)

#         edge_colors = [cmap(norm(edge_order.get(frozenset((u, v)), eo_min))) for (u, v) in G.edges]
#         # convert to RGBA float tuples napari accepts
#         edge_colors = [tuple(map(float, c)) for c in edge_colors]

#         node_to_i = {n: i for i, n in enumerate(nodes)}

#         # each edge is a 2-point polyline in ZYX
#         edge_lines = []
#         for (u, v) in G.edges:
#             i, j = node_to_i[u], node_to_i[v]
#             edge_lines.append(np.stack([node_zyx[i], node_zyx[j]], axis=0))

#         viewer.add_shapes(
#             edge_lines,
#             shape_type="line",
#             name="graph_edges_colored",
#             edge_width=edge_width,
#             edge_color=edge_colors,   # <-- per-edge colors
#             opacity=edge_opacity,
#         )

#     return pts

# import numpy as np
# import napari
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# from collections import defaultdict
# import numpy as np
# from collections import defaultdict
# import matplotlib.colors as mcolors
# from matplotlib import colormaps  # modern API

# def add_edge_layers_per_order(
#     viewer,
#     G,
#     edge_order: dict,
#     cmap_name="viridis",
#     edge_width=0.8,
#     opacity=0.9,
# ):
#     by_order = defaultdict(list)
#     for (u, v) in G.edges:
#         o = edge_order.get(frozenset((u, v)), None)
#         if o is None:
#             continue
#         by_order[int(o)].append((u, v))

#     if not by_order:
#         raise ValueError("No edges had an order in edge_order.")

#     orders = sorted(by_order.keys())
#     vmin = int(min(orders))
#     vmax = int(max(orders)) if max(orders) > min(orders) else int(min(orders)) + 1

#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     cmap = colormaps.get_cmap(cmap_name)

#     created_layers = {}
#     for o in orders:
#         lines = []
#         for (u, v) in by_order[o]:
#             u_xyz = np.asarray(u, dtype=float)
#             v_xyz = np.asarray(v, dtype=float)
#             u_zyx = u_xyz[[2, 1, 0]]
#             v_zyx = v_xyz[[2, 1, 0]]
#             lines.append(np.stack([u_zyx, v_zyx], axis=0))

#         hexc = mcolors.to_hex(cmap(norm(o)), keep_alpha=True)

#         layer = viewer.add_shapes(
#             lines,
#             shape_type="line",
#             name=f"order_{o}",
#             edge_color=hexc,          # <-- use hex
#             edge_width=edge_width,
#             opacity=opacity,
#         )
#         created_layers[o] = layer

#     return created_layers

# v = napari.Viewer(ndisplay=3)

# # nodes (optional, still useful)
# # visualize_branch_orders_in_napari(...)  # if you want nodes + table

# # edges: one layer per order
# edge_layers = add_edge_layers_per_order(
#     viewer=v,
#     G=G,
#     edge_order=edge_order,
#     cmap_name="viridis",
#     edge_width=0.8,
#     opacity=0.9,
# )

# # synapses
# syn_xyz = all_df[["extra_x","extra_y","extra_z"]].to_numpy(float)
# syn_zyx = syn_xyz[:, [2,1,0]]
# v.add_points(syn_zyx, name="synapses_filtered", size=4, face_color="red", opacity=0.8)

# napari.run()

# syn_xyz = all_df[["extra_x","extra_y","extra_z"]].to_numpy(float)
# syn_zyx = syn_xyz[:, [2,1,0]]

# v = napari.Viewer(ndisplay=3)

# visualize_branch_orders_in_napari(
#     viewer=v,
#     G=G,
#     node_order=node_order,
#     edge_order=edge_order,
#     branch_nodes=branch_nodes,
#     root=root,
#     show_edges=True,
#     node_size=2.0,
# )

# v.add_points(syn_zyx, name="synapses_filtered", size=4, face_color="red", opacity=0.8)

# import numpy as np
# import pandas as pd
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors

# def print_order_color_codes(orders, cmap_name="viridis"):
#     orders = np.asarray(list(orders), dtype=int)
#     orders = orders[np.isfinite(orders)]
#     orders = np.unique(orders)

#     vmin = int(orders.min())
#     vmax = int(orders.max()) if int(orders.max()) > int(orders.min()) else int(orders.min()) + 1

#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     cmap = cm.get_cmap(cmap_name)

#     rows = []
#     for o in orders:
#         rgba = tuple(map(float, cmap(norm(o))))  # (r,g,b,a) floats
#         hexc = mcolors.to_hex(rgba, keep_alpha=True)
#         rows.append({"order": int(o), "hex": hexc, "rgba": rgba})

#     df = pd.DataFrame(rows).sort_values("order")
#     print(df.to_string(index=False))

#     return df

# # branch_order colors (nodes)
# orders_for_nodes = set(node_order.values())
# node_color_df = print_order_color_codes(orders_for_nodes, cmap_name="viridis")

# napari.run()

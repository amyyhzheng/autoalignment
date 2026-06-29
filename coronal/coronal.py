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
FOLDER ='/Users/amyzheng/Desktop/sstnew/OutputManualAlignment'
ANALYZED_BASENAME = '3562analyzed.csv'

DIST_COL = "dendritic_dist_to_center"
TYPE_COL = "extra_type"
TYPES_TO_INCLUDE = [
    "Shaft_Geph+Bsn_+SynTd",
    "Spine_Geph+Bsn_+SynTd",
    "Shaft_Geph+Bsn_NoSynTd",
    "Spine_Geph+Bsn_NoSynTd",
]
DIST_SCALE = 1

TRACE_CSV = '/Users/amyzheng/Desktop/sstnew/TestingAlignment/3652_0224_Slice17_20xoverview_allcells_A01_G002_0001_final_xyzCoordinates.csv'
BRANCHPOINTS_CSV = '/Users/amyzheng/Desktop/sstnew/TestingAlignment/branchpoints10x.csv'

XCOL, YCOL, ZCOL = "extra_x", "extra_y", "extra_z"
MAX_EDGE_LENGTH = np.inf

SOMA_XYZ = (373.5216369, 486.2967972, 58.7670531)
#373.5216369, 486.2967972, 58.7670531)
# -----------------------------
# MANUAL ORDER SEGMENTS
# -----------------------------
# Format:
#   (branch_order, start_bp_index, end_bp_index)
#
# Multiple segments with the same branch_order are now MERGED correctly.
ORDER_SEGMENTS = [
    (1, 0, 1), 
    (2, 1, 2), 
    (2, 1, 3), 
    (3, 3, 4), 
    (3, 3, 5), 
    (4, 4, 6), 
    (4, 4, 7), 
    (5, 7, 8), 
    (5, 7, 11), 
    (6, 8, 10), 
    (6, 8, 9), 
    (6, 11, 12), 
    (6, 11, 13)
    #these points are for the sst cell
]

# Example if using soma:
# ORDER_SEGMENTS = [
#     (0, "SOMA", 0),
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
        return ZYX[:, [2, 1, 0]]  # ZYX -> XYZ

    axis_cols = [k for k in cols_lower.keys() if k.startswith("axis-")]
    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))
        A012 = df[[c0, c1, c2]].to_numpy(dtype=float)  # napari order ZYX
        return A012[:, [2, 1, 0]]  # ZYX -> XYZ

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
# COVERAGE LOGIC
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
        raise ValueError("analyzed.csv must have at least 2 points.")
    if len(s) % 2 != 0:
        raise ValueError("analyzed.csv must have an even number of rows, as endpoint pairs.")

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
def build_dendrite_graph(trace_df, branch_col="path", coord_cols=("x", "y", "z"), max_edge_length=5.0):
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
# ANALYZED EDGES
# -----------------------------
def analyzed_endpoint_pairs_from_xyz(analyzed_df, x="extra_x", y="extra_y", z="extra_z"):
    P = analyzed_df[[x, y, z]].to_numpy(dtype=float)

    if len(P) < 2 or len(P) % 2 != 0:
        raise ValueError("analyzed.csv XYZ must have an even number of rows >= 2.")

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

def add_virtual_bridge(G, kdtree_nodes, node_coords, a_xyz, b_xyz):
    """
    Adds a single virtual edge between the nearest graph nodes to a_xyz and b_xyz.
    """
    (a_node,), (da,) = snap_points_to_nodes(np.array([a_xyz], float), kdtree_nodes, node_coords)
    (b_node,), (db,) = snap_points_to_nodes(np.array([b_xyz], float), kdtree_nodes, node_coords)

    w = float(np.linalg.norm(np.asarray(a_node) - np.asarray(b_node)))
    G.add_edge(a_node, b_node, weight=w, is_virtual=True)

    virtual_edges = {frozenset((a_node, b_node))}

    return virtual_edges, a_node, b_node, float(da), float(db)

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
    seg is:
      (order, start_idx_or_SOMA, end_idx)
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
      order_to_path_nodes: order -> list of graph nodes from ALL segments of that order
      order_to_edges: order -> set of edges from ALL segments of that order
      order_to_segments: order -> details for each individual segment

    FIXED:
      Multiple ORDER_SEGMENTS with the same order are MERGED, not overwritten.
    """
    order_to_path_nodes = defaultdict(list)
    order_to_edges = defaultdict(set)
    order_to_segments = defaultdict(list)

    for seg in order_segments:
        order, a_xyz, b_xyz = get_segment_endpoints_xyz(
            branch_xyz,
            seg,
            soma_xyz=soma_xyz
        )
        order = int(order)

        (a_node,), _ = snap_points_to_nodes(np.array([a_xyz]), kdtree, node_coords)
        (b_node,), _ = snap_points_to_nodes(np.array([b_xyz]), kdtree, node_coords)

        try:
            path_nodes = nx.shortest_path(G, a_node, b_node, weight="weight")
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path between endpoints for order={order}, segment={seg}. "
                "Check branchpoint indices / graph connectivity."
            )

        path_edges = edges_from_path(path_nodes)

        # IMPORTANT FIX:
        # accumulate segments with the same branch order instead of overwriting
        order_to_path_nodes[order].extend(path_nodes)
        order_to_edges[order].update(path_edges)

        order_to_segments[order].append({
            "segment": seg,
            "start_node": a_node,
            "end_node": b_node,
            "n_path_nodes": len(path_nodes),
            "n_path_edges": len(path_edges),
        })

    return dict(order_to_path_nodes), dict(order_to_edges), dict(order_to_segments)

# -----------------------------
# SYNAPSE ASSIGNMENT
# -----------------------------
def assign_synapses_to_orders_by_nearest_path(syn_xyz, order_to_path_nodes):
    """
    Assign each synapse to the order whose combined path nodes are closest.

    If multiple branches have the same order, their path nodes are combined
    before building the KDTree.
    """
    orders = sorted(order_to_path_nodes.keys())

    if not orders:
        raise ValueError("No orders defined.")

    kdtrees = {}

    for o in orders:
        P = np.array(order_to_path_nodes[o], dtype=float)

        # Remove duplicate nodes within this order
        P = np.unique(P, axis=0)

        if len(P) == 0:
            continue

        kdtrees[o] = cKDTree(P)

    syn_order = np.full((syn_xyz.shape[0],), -1, dtype=int)
    syn_dist = np.full((syn_xyz.shape[0],), np.inf, dtype=float)

    for o, tree in kdtrees.items():
        d, _ = tree.query(syn_xyz.astype(float))
        better = d < syn_dist
        syn_dist[better] = d[better]
        syn_order[better] = o

    return syn_order, syn_dist

# ============================================================
# A) LOAD ANALYZED INTERVALS
# ============================================================
analyzed_path = os.path.join(FOLDER, ANALYZED_BASENAME)
analyzed_df = load_points_csv(analyzed_path, dist_col=DIST_COL)
analyzed_intervals = intervals_from_analyzed_df(analyzed_df, dist_col=DIST_COL)

# ============================================================
# B) READ + FILTER SYNAPSES
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

if len(dfs) == 0:
    raise ValueError("No valid synapse CSV files found.")

all_df = pd.concat(dfs, ignore_index=True)

if TYPES_TO_INCLUDE is not None:
    all_df = all_df[all_df[TYPE_COL].isin(TYPES_TO_INCLUDE)].copy()

mask_cov = filter_points_in_intervals(all_df[DIST_COL].to_numpy(float), analyzed_intervals)
all_df = all_df.loc[mask_cov].copy()

print(f"Synapses after type + analyzed-interval filtering: {len(all_df)}")

# ============================================================
# C) BUILD GRAPH + KDTree over graph nodes
# ============================================================
trace_df = pd.read_csv(TRACE_CSV)
G = build_dendrite_graph(
    trace_df,
    branch_col="path",
    coord_cols=("x", "y", "z"),
    max_edge_length=MAX_EDGE_LENGTH
)

if len(G.nodes) == 0:
    raise ValueError("Graph has zero nodes. Check TRACE_CSV and coordinate columns.")

node_coords = np.array(list(G.nodes), dtype=float)
kdtree_nodes = cKDTree(node_coords)

# Optional virtual bridge
VIRTUAL_EDGES, a_node, b_node, da, db = add_virtual_bridge(
    G,
    kdtree_nodes,
    node_coords,
    (766.318395770469, 723.7611452, 188.26122184763236),
    (762.5074913980607, 689.3125372443525, 187.501791969937)
)

print("Added virtual bridge:")
print(f"  node A = {a_node}, snap distance = {da:.3f}")
print(f"  node B = {b_node}, snap distance = {db:.3f}")

# Rebuild KDTree after bridge
node_coords = np.array(list(G.nodes), dtype=float)
kdtree_nodes = cKDTree(node_coords)

# ============================================================
# D) LOAD BRANCHPOINTS
# ============================================================
branch_xyz = load_napari_points_as_xyz(BRANCHPOINTS_CSV)

print(f"Loaded {len(branch_xyz)} branchpoints from {BRANCHPOINTS_CSV}")
_ = print_branchpoints_with_indices(branch_xyz, n=min(60, len(branch_xyz)))

# ============================================================
# E) BUILD ANALYZED EDGES
# ============================================================
if not all(c in analyzed_df.columns for c in ["extra_x", "extra_y", "extra_z"]):
    raise ValueError("analyzed.csv missing extra_x/extra_y/extra_z.")

endpoint_pairs = analyzed_endpoint_pairs_from_xyz(
    analyzed_df,
    x="extra_x",
    y="extra_y",
    z="extra_z"
)

analyzed_edges = mark_analyzed_edges_from_pairs(
    G,
    endpoint_pairs,
    kdtree_nodes,
    node_coords
)

print(f"Analyzed edges count: {len(analyzed_edges)}")

# ============================================================
# F) MANUAL ORDER SEGMENTS -> ORDER PATHS
# ============================================================
order_to_path_nodes, order_to_edges, order_to_segments = compute_order_paths(
    G=G,
    branch_xyz=branch_xyz,
    order_segments=ORDER_SEGMENTS,
    kdtree=kdtree_nodes,
    node_coords=node_coords,
    soma_xyz=SOMA_XYZ,
)

print("Orders defined:", sorted(order_to_edges.keys()))

for o in sorted(order_to_edges.keys()):
    unique_nodes = np.unique(np.array(order_to_path_nodes[o], dtype=float), axis=0)

    print(
        f"order {o}: "
        f"segments={len(order_to_segments[o])}, "
        f"unique path nodes={len(unique_nodes)}, "
        f"path edges={len(order_to_edges[o])}"
    )

    for info in order_to_segments[o]:
        print(
            f"    segment={info['segment']}, "
            f"path nodes={info['n_path_nodes']}, "
            f"path edges={info['n_path_edges']}"
        )

# ============================================================
# G) DENOMINATOR: ANALYZED LENGTH PER ORDER
# ============================================================
L = {}

for o, path_edges in order_to_edges.items():
    covered_edges = path_edges.intersection(analyzed_edges)
    L[o] = length_of_edge_set(G, covered_edges)

print("Analyzed length per order:")

for o in sorted(L.keys()):
    print(f"  order {o}: {L[o]:.3f}")

# ============================================================
# H) NUMERATOR: SYNAPSES PER ORDER
# ============================================================
syn_xyz = all_df[[XCOL, YCOL, ZCOL]].to_numpy(dtype=float)

if len(syn_xyz) == 0:
    print("Warning: no synapses remain after filtering.")
    syn_order = np.array([], dtype=int)
    syn_dist = np.array([], dtype=float)
else:
    syn_order, syn_dist = assign_synapses_to_orders_by_nearest_path(
        syn_xyz,
        order_to_path_nodes
    )

    print(
        "Synapse->path distance min/median/max:",
        float(np.min(syn_dist)),
        float(np.median(syn_dist)),
        float(np.max(syn_dist))
    )

C = pd.Series(syn_order).value_counts().to_dict() if len(syn_order) > 0 else {}

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
# I) PLOTS
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
ax.set_title("Analyzed length by branch order")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# 4) Density annotated with n and L
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, dens, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order (manual segments)")
ax.set_ylabel("Density (synapses / analyzed length)")
ax.set_title("Density by branch order")
ax.set_xticks(x)

for b, n, Lval in zip(bars, counts.astype(int), lengths):
    h = b.get_height()

    if np.isfinite(h) and h > 0:
        ax.text(
            b.get_x() + b.get_width() / 2,
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
# J) OPTIONAL: NAPARI VISUALIZATION
# ============================================================
DO_NAPARI = True

if DO_NAPARI:
    import napari

    v = napari.Viewer(ndisplay=3)

    # Synapses in ZYX
    if len(syn_xyz) > 0:
        syn_zyx = syn_xyz[:, [2, 1, 0]]
        v.add_points(
            syn_zyx,
            name="synapses_filtered",
            size=4,
            face_color="red",
            opacity=0.8
        )

    # Branchpoints in ZYX
    bp_zyx = branch_xyz[:, [2, 1, 0]]

    v.add_points(
        bp_zyx,
        name="branchpoints",
        size=6,
        face_color="yellow",
        properties={"idx": np.arange(len(branch_xyz), dtype=int)},
        text={
            "string": "{idx}",
            "size": 10,
            "color": "white",
            "anchor": "upper_left",
            "translation": np.array([0, 3, 3]),
        },
    )

    v.layers["branchpoints"].show_table = True

    # Order paths: one layer per order
    cmap = colormaps.get_cmap("viridis")
    o_list = sorted(order_to_path_nodes.keys())

    vmin = min(o_list)
    vmax = max(o_list) if max(o_list) > min(o_list) else min(o_list) + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for o in o_list:
        # Use the combined unique path nodes for this order
        path_nodes = np.array(order_to_path_nodes[o], dtype=float)
        path_nodes = np.unique(path_nodes, axis=0)

        # For napari lines, better to draw actual edges rather than sorted unique nodes
        segs = []

        for e in order_to_edges[o]:
            u, vtx = tuple(e)

            u_xyz = np.asarray(u, dtype=float)
            v_xyz = np.asarray(vtx, dtype=float)

            u_zyx = u_xyz[[2, 1, 0]]
            v_zyx = v_xyz[[2, 1, 0]]

            segs.append(np.stack([u_zyx, v_zyx], axis=0))

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

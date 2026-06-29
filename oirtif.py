import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from collections import defaultdict
from matplotlib import colormaps
import matplotlib.colors as mcolors

# ============================================================
# USER SETTINGS
# ============================================================
FOLDER = '/Users/amyzheng/Desktop/sstnew/OutputManualAlignment'

# This analyzed CSV should be napari-exported points:
# axis-0, axis-1, axis-2
# where axis-0 = z, axis-1 = y, axis-2 = x
# Every 2 rows are one analyzed segment endpoint pair.
ANALYZED_BASENAME = 'analyzed.csv'

TYPE_COL = "extra_type"
TYPES_TO_INCLUDE = [
    # "Ignore"
    "Shaft_Geph+Bsn_+SynTd",
    "Spine_Geph+Bsn_+SynTd",
    # "Shaft_Geph+Bsn_NoSynTd",
    # "Spine_Geph+Bsn_NoSynTd",
]

TRACE_CSV = '/Users/amyzheng/Desktop/sstnew/TestingAlignment/3652_0224_Slice17_20xoverview_allcells_A01_G002_0001_final_xyzCoordinates.csv'
BRANCHPOINTS_CSV = '/Users/amyzheng/Desktop/sstnew/TestingAlignment/branchpoints10x.csv'

XCOL, YCOL, ZCOL = "extra_x", "extra_y", "extra_z"

# Safer to use a real value if possible, e.g. 5 or 10.
# np.inf allows any consecutive trace points to connect.
MAX_EDGE_LENGTH = np.inf

SOMA_XYZ = (373.5216369, 486.2967972, 58.7670531)

# Optional: only use this if your trace has a real break you need to bridge.
USE_VIRTUAL_BRIDGE = False
VIRTUAL_BRIDGE_A_XYZ = (766.318395770469, 723.7611452, 188.26122184763236)
VIRTUAL_BRIDGE_B_XYZ = (762.5074913980607, 689.3125372443525, 187.501791969937)

# Optional cutoff after assignment.
# If None, every synapse is assigned to the nearest covered branch-order tree.
# If a number, synapses farther than this from covered tree are excluded.
MAX_SYN_TO_COVERED_TREE_DIST = None
# Example:
# MAX_SYN_TO_COVERED_TREE_DIST = 5.0

# ============================================================
# MANUAL ORDER SEGMENTS
# ============================================================
# Format:
#   (branch_order, start_bp_index, end_bp_index)
#
# Multiple segments with the same branch_order are merged.
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
    (6, 11, 13),
]

# Example if using soma:
# ORDER_SEGMENTS = [
#     (0, "SOMA", 0),
#     (1, 0, 1),
#     (2, 1, 2),
# ]

# ============================================================
# PLOT STYLE
# ============================================================
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

# ============================================================
# COORDINATE HELPERS
# ============================================================
def load_napari_points_as_xyz(csv_path: str) -> np.ndarray:
    """
    Load napari points CSV and return XYZ coordinates.

    Accepts either:
      axis-0, axis-1, axis-2  where axis-0=z, axis-1=y, axis-2=x
    or:
      z, y, x

    Returns:
      array with columns x, y, z
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    # Case 1: columns named z, y, x
    if "z" in cols_lower and "y" in cols_lower and "x" in cols_lower:
        zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
        zyx = df[[zc, yc, xc]].to_numpy(dtype=float)
        return zyx[:, [2, 1, 0]]  # ZYX -> XYZ

    # Case 2: napari axis-0, axis-1, axis-2
    axis_cols = [k for k in cols_lower.keys() if k.startswith("axis-")]
    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))
        zyx = df[[c0, c1, c2]].to_numpy(dtype=float)
        return zyx[:, [2, 1, 0]]  # ZYX -> XYZ

    raise ValueError(
        f"Unrecognized napari CSV columns in {csv_path}: {list(df.columns)}"
    )

def print_points_with_indices(points_xyz: np.ndarray, name="points", n=100):
    n = min(n, len(points_xyz))
    rows = []
    for i in range(n):
        x, y, z = points_xyz[i]
        rows.append({"idx": i, "x": x, "y": y, "z": z})
    df = pd.DataFrame(rows)
    print(f"\n{name}:")
    print(df.to_string(index=False))
    return df

# ============================================================
# GRAPH HELPERS
# ============================================================
def build_dendrite_graph(
    trace_df,
    branch_col="path",
    coord_cols=("x", "y", "z"),
    max_edge_length=5.0,
):
    """
    Build graph from trace CSV.
    Nodes are XYZ tuples.
    Edges connect consecutive points within each path.
    """
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
    """
    Snap arbitrary XYZ points to nearest graph nodes.
    """
    dists, idx = kdtree.query(points_xyz.astype(float))
    snapped = [tuple(node_coords[i]) for i in idx]
    return snapped, dists

def edges_from_path(path_nodes):
    """
    Convert node path into edge set.
    """
    return [frozenset((u, v)) for u, v in zip(path_nodes[:-1], path_nodes[1:])]

def length_of_edge_set(G, edge_set):
    """
    Sum graph edge lengths.
    """
    L = 0.0

    for e in edge_set:
        u, v = tuple(e)
        L += float(G[u][v].get("weight", 0.0))

    return float(L)

def edge_set_to_nodes(edge_set):
    """
    Convert an edge set into unique XYZ node array.
    """
    nodes = []

    for e in edge_set:
        u, v = tuple(e)
        nodes.append(u)
        nodes.append(v)

    if len(nodes) == 0:
        return np.empty((0, 3), dtype=float)

    return np.unique(np.array(nodes, dtype=float), axis=0)

def add_virtual_bridge(G, kdtree_nodes, node_coords, a_xyz, b_xyz):
    """
    Adds a single virtual edge between nearest graph nodes to a_xyz and b_xyz.
    Only use if you know the trace is broken at that location.
    """
    (a_node,), (da,) = snap_points_to_nodes(
        np.array([a_xyz], float),
        kdtree_nodes,
        node_coords,
    )
    (b_node,), (db,) = snap_points_to_nodes(
        np.array([b_xyz], float),
        kdtree_nodes,
        node_coords,
    )

    w = float(np.linalg.norm(np.asarray(a_node) - np.asarray(b_node)))
    G.add_edge(a_node, b_node, weight=w, is_virtual=True)

    virtual_edges = {frozenset((a_node, b_node))}

    return virtual_edges, a_node, b_node, float(da), float(db)

# ============================================================
# ANALYZED TREE FROM NAPARI POINTS
# ============================================================
def analyzed_endpoint_pairs_from_napari_csv(analyzed_csv):
    """
    Read analyzed.csv as napari points.

    The CSV should have an even number of rows.
    Every 2 rows define one analyzed tree path:
      row 0 -> row 1
      row 2 -> row 3
      etc.

    Napari coordinates are converted from ZYX to XYZ.
    """
    analyzed_xyz = load_napari_points_as_xyz(analyzed_csv)

    if len(analyzed_xyz) < 2:
        raise ValueError("analyzed.csv must have at least 2 points.")

    if len(analyzed_xyz) % 2 != 0:
        raise ValueError(
            "analyzed.csv must have an even number of rows. "
            "Every 2 rows should be one endpoint pair."
        )

    endpoint_pairs = []
    for i in range(0, len(analyzed_xyz), 2):
        endpoint_pairs.append((analyzed_xyz[i], analyzed_xyz[i + 1]))

    return endpoint_pairs, analyzed_xyz

def mark_edges_from_endpoint_pairs(G, endpoint_pairs_xyz, kdtree, node_coords, label="analyzed"):
    """
    Convert endpoint pairs into graph edges.

    For each pair:
      1. snap start point to nearest trace node
      2. snap end point to nearest trace node
      3. find shortest path on trace graph
      4. add all path edges to edge set
    """
    marked_edges = set()
    segment_info = []

    for i, (a_xyz, b_xyz) in enumerate(endpoint_pairs_xyz):
        (a_node,), (da,) = snap_points_to_nodes(
            np.array([a_xyz]),
            kdtree,
            node_coords,
        )
        (b_node,), (db,) = snap_points_to_nodes(
            np.array([b_xyz]),
            kdtree,
            node_coords,
        )

        try:
            path = nx.shortest_path(G, a_node, b_node, weight="weight")
        except nx.NetworkXNoPath:
            print(f"WARNING: no graph path for {label} segment {i}")
            continue

        path_edges = edges_from_path(path)
        marked_edges.update(path_edges)

        segment_info.append({
            "segment_index": i,
            "start_node": a_node,
            "end_node": b_node,
            "start_snap_dist": float(da),
            "end_snap_dist": float(db),
            "n_path_nodes": len(path),
            "n_path_edges": len(path_edges),
            "path_length": length_of_edge_set(G, set(path_edges)),
        })

    return marked_edges, segment_info

# ============================================================
# BRANCH ORDER TREE FROM BRANCHPOINTS
# ============================================================
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
    Convert ORDER_SEGMENTS into branch-order graph edges.

    Multiple segments with the same order are merged.
    """
    order_to_path_nodes = defaultdict(list)
    order_to_edges = defaultdict(set)
    order_to_segments = defaultdict(list)

    for seg in order_segments:
        order, a_xyz, b_xyz = get_segment_endpoints_xyz(
            branch_xyz,
            seg,
            soma_xyz=soma_xyz,
        )

        order = int(order)

        (a_node,), (da,) = snap_points_to_nodes(
            np.array([a_xyz]),
            kdtree,
            node_coords,
        )
        (b_node,), (db,) = snap_points_to_nodes(
            np.array([b_xyz]),
            kdtree,
            node_coords,
        )

        try:
            path_nodes = nx.shortest_path(G, a_node, b_node, weight="weight")
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path between endpoints for order={order}, segment={seg}. "
                "Check branchpoint indices / graph connectivity."
            )

        path_edges = edges_from_path(path_nodes)

        order_to_path_nodes[order].extend(path_nodes)
        order_to_edges[order].update(path_edges)

        order_to_segments[order].append({
            "segment": seg,
            "start_node": a_node,
            "end_node": b_node,
            "start_snap_dist": float(da),
            "end_snap_dist": float(db),
            "n_path_nodes": len(path_nodes),
            "n_path_edges": len(path_edges),
            "path_length": length_of_edge_set(G, set(path_edges)),
        })

    return dict(order_to_path_nodes), dict(order_to_edges), dict(order_to_segments)

# ============================================================
# SYNAPSE ASSIGNMENT TO OVERLAP TREE
# ============================================================
def assign_synapses_to_orders_by_covered_edges(syn_xyz, order_to_covered_edges):
    """
    Assign synapses to the nearest analyzed-overlap tree for each branch order.

    order_to_covered_edges[order] should already be:
      analyzed_edges ∩ order_to_edges[order]
    """
    kdtrees = {}

    for o, edges in order_to_covered_edges.items():
        P = edge_set_to_nodes(edges)

        if len(P) == 0:
            continue

        kdtrees[o] = cKDTree(P)

    if len(kdtrees) == 0:
        raise ValueError(
            "No covered/analyzed branch-order edges found. "
            "Check analyzed.csv endpoint pairs and ORDER_SEGMENTS."
        )

    syn_order = np.full((syn_xyz.shape[0],), -1, dtype=int)
    syn_dist = np.full((syn_xyz.shape[0],), np.inf, dtype=float)

    for o, tree in kdtrees.items():
        d, _ = tree.query(syn_xyz.astype(float))
        better = d < syn_dist
        syn_dist[better] = d[better]
        syn_order[better] = o

    return syn_order, syn_dist

# ============================================================
# A) READ SYNAPSE CSV FILES
# ============================================================
analyzed_path = os.path.join(FOLDER, ANALYZED_BASENAME)

paths = sorted(glob.glob(os.path.join(FOLDER, "*.csv")))

# Exclude analyzed.csv and obvious branchpoint files from synapse inputs
paths = [
    p for p in paths
    if os.path.basename(p) != ANALYZED_BASENAME
    and "branchpoint" not in os.path.basename(p).lower()
]

dfs = []

for p in paths:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"Skipping {os.path.basename(p)}: could not read CSV: {e}")
        continue

    required = {XCOL, YCOL, ZCOL, TYPE_COL}
    missing = required - set(df.columns)

    if missing:
        print(f"Skipping {os.path.basename(p)}: missing columns {sorted(missing)}")
        continue

    df = df.copy()
    df[XCOL] = pd.to_numeric(df[XCOL], errors="coerce")
    df[YCOL] = pd.to_numeric(df[YCOL], errors="coerce")
    df[ZCOL] = pd.to_numeric(df[ZCOL], errors="coerce")

    df = df[np.isfinite(df[XCOL]) & np.isfinite(df[YCOL]) & np.isfinite(df[ZCOL])].copy()
    df["source_file"] = os.path.basename(p)

    dfs.append(df)

if len(dfs) == 0:
    raise ValueError("No valid synapse CSV files found.")

all_df = pd.concat(dfs, ignore_index=True)

if TYPES_TO_INCLUDE is not None:
    all_df = all_df[all_df[TYPE_COL].isin(TYPES_TO_INCLUDE)].copy()

print(f"Synapses after type filtering: {len(all_df)}")

# ============================================================
# B) BUILD DENDRITE GRAPH
# ============================================================
trace_df = pd.read_csv(TRACE_CSV)

G = build_dendrite_graph(
    trace_df,
    branch_col="path",
    coord_cols=("x", "y", "z"),
    max_edge_length=MAX_EDGE_LENGTH,
)

if len(G.nodes) == 0:
    raise ValueError("Graph has zero nodes. Check TRACE_CSV and coordinate columns.")

node_coords = np.array(list(G.nodes), dtype=float)
kdtree_nodes = cKDTree(node_coords)

if USE_VIRTUAL_BRIDGE:
    VIRTUAL_EDGES, a_node, b_node, da, db = add_virtual_bridge(
        G,
        kdtree_nodes,
        node_coords,
        VIRTUAL_BRIDGE_A_XYZ,
        VIRTUAL_BRIDGE_B_XYZ,
    )

    print("Added virtual bridge:")
    print(f"  node A = {a_node}, snap distance = {da:.3f}")
    print(f"  node B = {b_node}, snap distance = {db:.3f}")

    # Rebuild KDTree after bridge
    node_coords = np.array(list(G.nodes), dtype=float)
    kdtree_nodes = cKDTree(node_coords)

print(f"Graph nodes: {len(G.nodes)}")
print(f"Graph edges: {len(G.edges)}")

# ============================================================
# C) LOAD BRANCHPOINTS AND ANALYZED POINTS
# ============================================================
branch_xyz = load_napari_points_as_xyz(BRANCHPOINTS_CSV)
print_points_with_indices(branch_xyz, name="Branchpoints", n=min(100, len(branch_xyz)))

analyzed_endpoint_pairs, analyzed_xyz = analyzed_endpoint_pairs_from_napari_csv(analyzed_path)
print_points_with_indices(analyzed_xyz, name="Analyzed endpoint points", n=min(100, len(analyzed_xyz)))

print(f"Analyzed endpoint pairs: {len(analyzed_endpoint_pairs)}")

# Coordinate range sanity check
syn_xyz_all = all_df[[XCOL, YCOL, ZCOL]].to_numpy(dtype=float)

print("\nSynapse XYZ range:")
print(pd.DataFrame(syn_xyz_all, columns=["x", "y", "z"]).agg(["min", "max"]))

print("\nTrace node XYZ range:")
print(pd.DataFrame(node_coords, columns=["x", "y", "z"]).agg(["min", "max"]))

print("\nBranchpoint XYZ range:")
print(pd.DataFrame(branch_xyz, columns=["x", "y", "z"]).agg(["min", "max"]))

print("\nAnalyzed endpoint XYZ range:")
print(pd.DataFrame(analyzed_xyz, columns=["x", "y", "z"]).agg(["min", "max"]))

# ============================================================
# D) BUILD ANALYZED TREE EDGES
# ============================================================
analyzed_edges, analyzed_segment_info = mark_edges_from_endpoint_pairs(
    G,
    analyzed_endpoint_pairs,
    kdtree_nodes,
    node_coords,
    label="analyzed",
)

print(f"\nAnalyzed edges count: {len(analyzed_edges)}")
print(f"Analyzed total tree length: {length_of_edge_set(G, analyzed_edges):.3f}")

print("\nAnalyzed segment details:")
for info in analyzed_segment_info:
    print(
        f"  segment {info['segment_index']}: "
        f"edges={info['n_path_edges']}, "
        f"length={info['path_length']:.3f}, "
        f"snap=({info['start_snap_dist']:.3f}, {info['end_snap_dist']:.3f})"
    )

# ============================================================
# E) BUILD BRANCH-ORDER TREE EDGES
# ============================================================
order_to_path_nodes, order_to_edges, order_to_segments = compute_order_paths(
    G=G,
    branch_xyz=branch_xyz,
    order_segments=ORDER_SEGMENTS,
    kdtree=kdtree_nodes,
    node_coords=node_coords,
    soma_xyz=SOMA_XYZ,
)

print("\nOrders defined:", sorted(order_to_edges.keys()))

for o in sorted(order_to_edges.keys()):
    unique_nodes = np.unique(np.array(order_to_path_nodes[o], dtype=float), axis=0)
    total_order_length = length_of_edge_set(G, order_to_edges[o])

    print(
        f"order {o}: "
        f"segments={len(order_to_segments[o])}, "
        f"unique path nodes={len(unique_nodes)}, "
        f"path edges={len(order_to_edges[o])}, "
        f"total length={total_order_length:.3f}"
    )

    for info in order_to_segments[o]:
        print(
            f"    segment={info['segment']}, "
            f"path edges={info['n_path_edges']}, "
            f"length={info['path_length']:.3f}, "
            f"snap=({info['start_snap_dist']:.3f}, {info['end_snap_dist']:.3f})"
        )

# ============================================================
# F) OVERLAP TREE: analyzed_edges ∩ branch-order edges
# ============================================================
order_to_covered_edges = {}
L = {}

for o, path_edges in order_to_edges.items():
    covered_edges = path_edges.intersection(analyzed_edges)
    order_to_covered_edges[o] = covered_edges
    L[o] = length_of_edge_set(G, covered_edges)

print("\nAnalyzed-overlap length per order:")
for o in sorted(L.keys()):
    print(
        f"  order {o}: "
        f"L={L[o]:.3f}, "
        f"covered edges={len(order_to_covered_edges[o])}"
    )

# ============================================================
# G) ASSIGN SYNAPSES TO ANALYZED-OVERLAP TREE
# ============================================================
syn_xyz = all_df[[XCOL, YCOL, ZCOL]].to_numpy(dtype=float)

if len(syn_xyz) == 0:
    print("Warning: no synapses remain after type filtering.")
    syn_order = np.array([], dtype=int)
    syn_dist = np.array([], dtype=float)

else:
    syn_order, syn_dist = assign_synapses_to_orders_by_covered_edges(
        syn_xyz,
        order_to_covered_edges,
    )

    if MAX_SYN_TO_COVERED_TREE_DIST is not None:
        keep = syn_dist <= MAX_SYN_TO_COVERED_TREE_DIST

        all_df = all_df.loc[keep].copy()
        syn_xyz = syn_xyz[keep]
        syn_order = syn_order[keep]
        syn_dist = syn_dist[keep]

        print(
            f"Synapses after distance cutoff "
            f"{MAX_SYN_TO_COVERED_TREE_DIST}: {len(all_df)}"
        )

    print("\nAssigned order counts:")
    print(pd.Series(syn_order).value_counts().sort_index())

    print("\nSynapse distance to nearest analyzed-overlap tree:")
    print(pd.Series(syn_dist).describe())

# Add assignment columns to output debug table
assigned_df = all_df.copy()
assigned_df["assigned_branch_order"] = syn_order
assigned_df["distance_to_analyzed_order_tree"] = syn_dist

debug_csv = os.path.join(FOLDER, "debug_synapse_assignment_to_analyzed_order_tree.csv")
assigned_df.to_csv(debug_csv, index=False)
print(f"\nSaved debug assignment CSV: {debug_csv}")

# ============================================================
# H) OUTPUT DENSITY TABLE
# ============================================================
C = pd.Series(syn_order).value_counts().to_dict() if len(syn_order) > 0 else {}

orders = sorted(set(L.keys()) | set(C.keys()))

out = pd.DataFrame({
    "branch_order": orders,
    "analyzed_overlap_length": [L.get(o, 0.0) for o in orders],
    "synapse_count": [int(C.get(o, 0)) for o in orders],
})

out["density"] = out["synapse_count"] / out["analyzed_overlap_length"].replace(0, np.nan)

out_csv = os.path.join(FOLDER, "density_by_branch_order_analyzed_overlap.csv")
out.to_csv(out_csv, index=False)

print(f"\nSaved density CSV: {out_csv}")
print(out)

# ============================================================
# I) PLOTS
# ============================================================
out = out.sort_values("branch_order").reset_index(drop=True)

x = out["branch_order"].to_numpy(int)
dens = out["density"].to_numpy(float)
counts = out["synapse_count"].to_numpy(float)
lengths = out["analyzed_overlap_length"].to_numpy(float)

# Density
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, dens, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order")
ax.set_ylabel("Density (synapses / analyzed-overlap length)")
ax.set_title("Synapse density by branch order")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# Counts
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, counts, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order")
ax.set_ylabel("Synapse count")
ax.set_title("Synapse count by branch order")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# Analyzed-overlap lengths
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, lengths, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order")
ax.set_ylabel("Analyzed-overlap length")
ax.set_title("Analyzed tree length by branch order")
ax.set_xticks(x)
black_bg(ax)
plt.tight_layout()
plt.show()

# Density annotated with n and L
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, dens, color=CYAN_SOFT, edgecolor=CYAN, linewidth=0.9)
ax.set_xlabel("Branch order")
ax.set_ylabel("Density (synapses / analyzed-overlap length)")
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
# J) NAPARI VISUALIZATION
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
            name="synapses_assigned_to_analyzed_order_tree",
            size=4,
            face_color="red",
            opacity=0.8,
            properties={
                "assigned_order": syn_order,
                "dist_to_tree": syn_dist,
            },
        )

    # Branchpoints with labels
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

    # Analyzed endpoint points with labels
    analyzed_zyx = analyzed_xyz[:, [2, 1, 0]]

    v.add_points(
        analyzed_zyx,
        name="analyzed_endpoint_points",
        size=5,
        face_color="magenta",
        properties={"idx": np.arange(len(analyzed_xyz), dtype=int)},
        text={
            "string": "{idx}",
            "size": 8,
            "color": "white",
            "anchor": "upper_left",
            "translation": np.array([0, 3, 3]),
        },
    )

    # Full analyzed tree edges
    analyzed_segs = []

    for e in analyzed_edges:
        u, vtx = tuple(e)
        u_xyz = np.asarray(u, dtype=float)
        v_xyz = np.asarray(vtx, dtype=float)
        analyzed_segs.append(
            np.stack([u_xyz[[2, 1, 0]], v_xyz[[2, 1, 0]]], axis=0)
        )

    if len(analyzed_segs) > 0:
        v.add_shapes(
            analyzed_segs,
            shape_type="line",
            name="analyzed_tree_edges",
            edge_color="magenta",
            edge_width=1.0,
            opacity=0.5,
        )

    # Branch-order overlap paths: one layer per order
    cmap = colormaps.get_cmap("viridis")
    o_list = sorted(order_to_covered_edges.keys())

    vmin = min(o_list)
    vmax = max(o_list) if max(o_list) > min(o_list) else min(o_list) + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for o in o_list:
        segs = []

        for e in order_to_covered_edges[o]:
            u, vtx = tuple(e)

            u_xyz = np.asarray(u, dtype=float)
            v_xyz = np.asarray(vtx, dtype=float)

            u_zyx = u_xyz[[2, 1, 0]]
            v_zyx = v_xyz[[2, 1, 0]]

            segs.append(np.stack([u_zyx, v_zyx], axis=0))

        if len(segs) == 0:
            continue

        color_hex = mcolors.to_hex(cmap(norm(o)), keep_alpha=True)

        v.add_shapes(
            segs,
            shape_type="line",
            name=f"analyzed_overlap_order_{o}",
            edge_color=color_hex,
            edge_width=1.4,
            opacity=0.95,
        )

    napari.run()


import numpy as np
import pandas as pd
import napari
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter


# -----------------------------
# USER SETTINGS
# -----------------------------
TRACE_CSV = '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5/SNTTrace/Full_Cell_SNT/SOM055_Image2_FullTrace_xyzCoordinates.csv'
EXTRA_POINTS_CSV ='/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5/Alignment_and_checking/AfterManualEdits/checked_Branch2/Image2_branch2_with_empty_markers.csv'

OUT_CSV ="/Users/amyzheng/Desktop/testingdistance/test1.csv"

#includes scaling
REFERENCE_POINT_FOR_CENTER = (
    382/4,
    324/4,
    27

)

MAX_EDGE_LENGTH = 5.0



def load_napari_points_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    if {"z", "y", "x"} <= set(cols_lower):
        zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
        #because Joe originally had scale factors 1, 1, 4
        df["_x"] = df[xc].astype(float)*0.25
        df["_y"] = df[yc].astype(float)*0.25
        df["_z"] = df[zc].astype(float)* 0.25
        return df

    axis_cols = [k for k in cols_lower if k.startswith("axis-")]
    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))

        # napari axis order is usually z, y, x
        df["_x"] = df[c2].astype(float)*0.25
        df["_y"] = df[c1].astype(float)*0.25
        df["_z"] = df[c0].astype(float) * 0.25
        return df

    raise ValueError(f"Unrecognized napari CSV columns: {list(df.columns)}")


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
    G = nx.Graph()

    if branch_col in full_trace_df.columns:
        groups = full_trace_df.groupby(branch_col)
    else:
        groups = [(0, full_trace_df)]

    for _, group in groups:
        coords = group.loc[:, coord_cols].dropna().to_numpy(dtype=float)

        #because the snt coordinates were xyz 111 and actual is 0.25, 0.25, 1
        # scale x, y, z
        coords[:, 0] *= 0.25  # x
        coords[:, 1] *= 0.25  # y

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

    center = tuple(reference_point) if reference_point is not None else None

    if center is not None and len(G) > 0:
        reference_point = np.asarray(reference_point, dtype=float)
        components = list(nx.connected_components(G))
        closest_terminal_nodes = []

        for comp in components:
            terminal_nodes = [node for node in comp if G.degree[node] == 1]

            if not terminal_nodes:
                continue

            terminal_coords = np.array(terminal_nodes, dtype=float)
            dists = np.linalg.norm(
                terminal_coords - reference_point[None, :],
                axis=1,
            )

            closest_terminal_nodes.append(
                terminal_nodes[int(np.argmin(dists))]
            )

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
        dist = nx.shortest_path_length(
            G,
            source=nearest_node,
            target=tuple(center_xyz),
            weight="weight",
        )
    except nx.NetworkXNoPath:
        dist = np.inf

    return float(dist)


# -----------------------------
# MAIN
# -----------------------------
df_ref = pd.read_csv(TRACE_CSV)

extra_df = load_napari_points_df(EXTRA_POINTS_CSV)
extra_xyz_arr = extra_df[["_x", "_y", "_z"]].to_numpy(dtype=float)

G10x, center10x = build_dendrite_tree_from_xyz_df(
    df_ref,
    branch_col="path" if "path" in df_ref.columns else "__no_path__",
    coord_cols=("x", "y", "z"),
    max_edge_length=MAX_EDGE_LENGTH,
    reference_point=REFERENCE_POINT_FOR_CENTER,
)

if center10x is None:
    raise RuntimeError("center10x is None. Check reference point or trace graph.")

node_coords = np.array(list(G10x.nodes), dtype=float)
kdtree = cKDTree(node_coords)

nearest_nodes = []
nearest_dists = []
dist_to_center = []

for p in extra_xyz_arr:
    nn, d = find_nearest_node(p, kdtree, node_coords)

    nearest_nodes.append(nn)
    nearest_dists.append(d)

    dist_to_center.append(
        graph_distance_to_center_from_xyz(
            p,
            center10x,
            G10x,
            kdtree,
            node_coords,
        )
    )

nearest_nodes = np.array(nearest_nodes, dtype=float)
nearest_dists = np.array(nearest_dists, dtype=float)
dist_to_center = np.array(dist_to_center, dtype=float)

matches = pd.DataFrame(
    {
        "label": extra_df["label"].values
            if "label" in extra_df.columns
            else np.array([None] * len(extra_xyz_arr)),
        "type": (
            extra_df["type"].values
            if "type" in extra_df.columns
            else np.array([None] * len(extra_xyz_arr))
        ),
        "axis-2": extra_xyz_arr[:, 0],#x
        "axis-1": extra_xyz_arr[:, 1], #y
        "axis-0": extra_xyz_arr[:, 2], #z
        "nearest10x_x": nearest_nodes[:, 0],
        "nearest10x_y": nearest_nodes[:, 1],
        "nearest10x_z": nearest_nodes[:, 2],
        "euclid_dist_to_graph_node": nearest_dists,
        "dendritic_dist_to_center": dist_to_center,
        # "bouton_overlap_fracr2": (
        #     extra_df["bouton_overlap_fracr2"].values
        #     if "type" in extra_df.columns
        #     else np.array([None] * len(extra_xyz_arr))
        # ),
    }
)

print(matches.head(10))

matches.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")


# -----------------------------
# NAPARI VISUALIZATION
# -----------------------------
assigned_nodes_xyz = np.unique(nearest_nodes, axis=0)
assigned_set = set(map(tuple, assigned_nodes_xyz))
all_nodes_set = set(map(tuple, node_coords))

unassigned_nodes_xyz = np.array(
    [n for n in all_nodes_set if n not in assigned_set],
    dtype=float,
)

counts = Counter(map(tuple, nearest_nodes))
assigned_sizes = np.array(
    [counts[tuple(n)] for n in assigned_nodes_xyz],
    dtype=float,
)

viewer = napari.Viewer(ndisplay=3)


def rand_color(name: str):
    seed = abs(hash(name)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(3)



# show extra points
viewer.add_points(
    extra_xyz_arr[:, [2, 1, 0]],
    name="extra_points",
    size=8,
    face_color=[1.0, 0.2, 0.2],
    opacity=1.0,
)


# show center
viewer.add_points(
    np.array(center10x, dtype=float)[None, [2, 1, 0]],
    name="center_node",
    size=12,
    face_color=[0.0, 1.0, 1.0],
    opacity=1.0,
)


# graph nodes
if len(unassigned_nodes_xyz) > 0:
    viewer.add_points(
        unassigned_nodes_xyz[:, [2, 1, 0]],
        name="graph_nodes_unassigned",
        size=1.5,
        face_color=[0.6, 0.6, 0.6],
        opacity=0.15,
    )

viewer.add_points(
    assigned_nodes_xyz[:, [2, 1, 0]],
    name="graph_nodes_assigned",
    size=4,
    face_color=[1.0, 0.85, 0.1],
    opacity=1.0,
)

viewer.add_points(
    assigned_nodes_xyz[:, [2, 1, 0]],
    name="graph_nodes_assigned_weighted",
    size=2 + 1.5 * assigned_sizes,
    face_color=[1.0, 0.4, 0.0],
    opacity=0.9,
)


# vectors from extra point to nearest graph node
extra_zyx = extra_xyz_arr[:, [2, 1, 0]]
nearest_zyx = nearest_nodes[:, [2, 1, 0]]

vecs = np.zeros((len(extra_zyx), 2, 3), dtype=float)
vecs[:, 0, :] = extra_zyx
vecs[:, 1, :] = nearest_zyx - extra_zyx

viewer.add_vectors(
    vecs,
    name="extra_to_nearest_trace_vectors",
    edge_width=2,
    opacity=0.7,
)

napari.run()

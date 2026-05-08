import numpy as np
import pandas as pd
import napari
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter


# -----------------------------
# USER SETTINGS
# -----------------------------
TRACE_CSV = '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell4/SNTTrace/Image0/SOM055_Image0_Trace_xyzCoordinates.csv'

# must be on the puncta scoring one 
EXTRA_POINTS_CSV = '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell4/PunctaScoring/branch1/SOM055_Image0_branch1.csv'

OUT_CSV = "/Users/amyzheng/Desktop/testing.csv"

REFERENCE_POINT_FOR_CENTER = (
    771.7282062274531,
    777.3587841588766,
    172.87154911171154,
)


IMAGE_NUM = 0
BASE_DIR = "/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell4"

MAX_EDGE_LENGTH = 5.0

for BRANCH_NUM in [1, 3, 4, 5,6,7, 8, 9, 10]:
    EXTRA_POINTS_CSV = (
        f"{BASE_DIR}/PunctaScoring/branch{BRANCH_NUM}/"
        f"SOM055_Image{IMAGE_NUM}_branch{BRANCH_NUM}.csv"
    )

    OUT_CSV = (
        f'/Users/amyzheng/Desktop/test/'
        f"SOM055_Image{IMAGE_NUM}_branch{BRANCH_NUM}_test.csv"
    )

    print(f"Running branch {BRANCH_NUM}")
    print(EXTRA_POINTS_CSV)


    def load_napari_points_df(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        cols_lower = {c.lower(): c for c in df.columns}

        if {"z", "y", "x"} <= set(cols_lower):
            zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
            df["_x"] = df[xc].astype(float)
            df["_y"] = df[yc].astype(float)
            df["_z"] = df[zc].astype(float) * 0.25
            return df

        axis_cols = [k for k in cols_lower if k.startswith("axis-")]
        if len(axis_cols) >= 3:
            axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
            c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))

            # napari axis order is usually z, y, x
            df["_x"] = df[c2].astype(float)
            df["_y"] = df[c1].astype(float)
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


    def graph_distance_between_points(coord1_xyz, coord2_xyz, G, kdtree, node_coords):
        node1, snap_dist1 = find_nearest_node(coord1_xyz, kdtree, node_coords)
        node2, snap_dist2 = find_nearest_node(coord2_xyz, kdtree, node_coords)

        try:
            path_dist = nx.shortest_path_length(
                G,
                source=node1,
                target=node2,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            path_dist = np.inf

        return float(path_dist), node1, node2, snap_dist1, snap_dist2


    def get_start_end_from_extra_df(extra_df: pd.DataFrame):
        if "type" not in extra_df.columns:
            raise ValueError(
                "EXTRA_POINTS_CSV must contain a 'type' column with StartBranch and EndBranch."
            )

        type_clean = extra_df["type"].astype(str).str.strip().str.lower()

        start_rows = extra_df[type_clean.isin(["startbranch", "start_branch"])]
        end_rows = extra_df[type_clean.isin(["endbranch", "end_branch"])]

        if len(start_rows) != 1 or len(end_rows) != 1:
            raise ValueError(
                f"Expected exactly one StartBranch and one EndBranch; "
                f"got {len(start_rows)} start and {len(end_rows)} end."
            )

        start_xyz = start_rows[["_x", "_y", "_z"]].iloc[0].to_numpy(dtype=float)
        end_xyz = end_rows[["_x", "_y", "_z"]].iloc[0].to_numpy(dtype=float)

        return start_xyz, end_xyz


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


    # -----------------------------
    # BRANCH START → END DISTANCE
    # -----------------------------
    start_xyz, end_xyz = get_start_end_from_extra_df(extra_df)

    branch_start_to_end_dist, start_node, end_node, start_snap_dist, end_snap_dist = (
        graph_distance_between_points(
            start_xyz,
            end_xyz,
            G10x,
            kdtree,
            node_coords,
        )
    )

    print("\nBranch Start → End distance")
    print(f"StartBranch xyz: {start_xyz}")
    print(f"EndBranch xyz:   {end_xyz}")
    print(f"Start snapped node: {start_node}, snap distance = {start_snap_dist:.3f}")
    print(f"End snapped node:   {end_node}, snap distance = {end_snap_dist:.3f}")
    print(f"Dendritic Start→End distance = {branch_start_to_end_dist:.3f}")


    # -----------------------------
    # DISTANCE FOR EACH EXTRA POINT
    # -----------------------------
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


    # -----------------------------
    # OUTPUT CSV
    # -----------------------------
    matches = pd.DataFrame(
        {
            "extra_i": np.arange(len(extra_xyz_arr)),
            "extra_type": (
                extra_df["type"].values
                if "type" in extra_df.columns
                else np.array([None] * len(extra_xyz_arr))
            ),
            "extra_x": extra_xyz_arr[:, 0],
            "extra_y": extra_xyz_arr[:, 1],
            "extra_z": extra_xyz_arr[:, 2],
            "nearest10x_x": nearest_nodes[:, 0],
            "nearest10x_y": nearest_nodes[:, 1],
            "nearest10x_z": nearest_nodes[:, 2],
            "euclid_dist_to_graph_node": nearest_dists,
            "dendritic_dist_to_center": dist_to_center,
            "dendritic_start_to_end_distance": branch_start_to_end_dist,
            "start_x": start_xyz[0],
            "start_y": start_xyz[1],
            "start_z": start_xyz[2],
            "end_x": end_xyz[0],
            "end_y": end_xyz[1],
            "end_z": end_xyz[2],
        }
    )

    print(matches.head(10))

    matches.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


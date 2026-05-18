import numpy as np
import pandas as pd
import napari
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter
from pathlib import Path


BASE_DIR = Path(
    "/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5"
)

TRACE_CSV = BASE_DIR / "SNTTrace/Full_Cell_SNT/SOM055_Image2_FullTrace_xyzCoordinates.csv"

TOPOLOGY_CSV = '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5/SNTTrace/Full_Cell_SNT/SOM055_Image2_SNT_BranchInfo.csv'
AFTER_MANUAL_EDITS_DIR = BASE_DIR / "Alignment_and_checking/AfterManualEdits"

OUT_DIR = Path("/Users/amyzheng/Desktop/testingdistance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCH_IDS = [1, 2, 3, 4, 5]

REFERENCE_POINT_FOR_CENTER = (
    382 / 4,
    324 / 4,
    27,
)

MAX_EDGE_LENGTH = 5.0
SHOW_NAPARI = False


def load_napari_points_df(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    if {"z", "y", "x"} <= set(cols_lower):
        zc, yc, xc = cols_lower["z"], cols_lower["y"], cols_lower["x"]
        df["_x"] = df[xc].astype(float) * 0.25
        df["_y"] = df[yc].astype(float) * 0.25
        df["_z"] = df[zc].astype(float) * 0.25
        return df

    axis_cols = [k for k in cols_lower if k.startswith("axis-")]

    if len(axis_cols) >= 3:
        axis_cols_sorted = sorted(axis_cols, key=lambda s: int(s.split("-")[1]))
        c0, c1, c2 = (cols_lower[axis_cols_sorted[i]] for i in range(3))

        df["_x"] = df[c2].astype(float) * 0.25
        df["_y"] = df[c1].astype(float) * 0.25
        df["_z"] = df[c0].astype(float) * 0.25
        return df

    raise ValueError(f"Unrecognized napari CSV columns: {list(df.columns)}")


def get_branch_run_endpoints_from_trace(
    full_trace_df: pd.DataFrame,
    branch_id: int,
    branch_col: str = "path",
    coord_cols=("x", "y", "z"),
):
    if branch_col not in full_trace_df.columns:
        raise ValueError(f"Branch column '{branch_col}' not found in trace CSV.")

    df = full_trace_df.copy()
    target_branch_name = f"branch{branch_id}"

    df = df[df[branch_col].astype(str).str.lower() == target_branch_name.lower()].copy()

    if df.empty:
        available = sorted(full_trace_df[branch_col].astype(str).unique())
        raise ValueError(
            f"No rows found for real branch '{target_branch_name}'. "
            f"Available path values include: {available[:20]}"
        )

    start = df.iloc[0].loc[list(coord_cols)].to_numpy(dtype=float)
    end = df.iloc[-1].loc[list(coord_cols)].to_numpy(dtype=float)

    start[0] *= 0.25
    start[1] *= 0.25

    end[0] *= 0.25
    end[1] *= 0.25

    return tuple(start), tuple(end)


def build_dendrite_tree_from_xyz_df(
    full_trace_df: pd.DataFrame,
    topology_df: pd.DataFrame,
    branch_col: str = "path",
    coord_cols=("x", "y", "z"),
    max_edge_length=5.0,
    reference_point=None,
):
    G = nx.Graph()

    if branch_col in full_trace_df.columns:
        groups = full_trace_df.groupby(branch_col, sort=False)
    else:
        groups = [(0, full_trace_df)]

    for path_name, group in groups:
        group = group.reset_index(drop=True)
        coords = group.loc[:, coord_cols].dropna().to_numpy(dtype=float)

        coords[:, 0] *= 0.25
        coords[:, 1] *= 0.25

        if len(coords) < 2:
            continue

        node_ids = []

        for i, xyz in enumerate(coords):
            node_id = (str(path_name), i)
            node_ids.append(node_id)

            G.add_node(
                node_id,
                xyz=tuple(xyz),
                path_name=str(path_name),
                index=i,
            )

        for i in range(len(node_ids) - 1):
            n1 = node_ids[i]
            n2 = node_ids[i + 1]

            xyz1 = np.array(G.nodes[n1]["xyz"], dtype=float)
            xyz2 = np.array(G.nodes[n2]["xyz"], dtype=float)

            dist = float(np.linalg.norm(xyz1 - xyz2))

            if dist < max_edge_length:
                G.add_edge(n1, n2, weight=dist)

    node_ids = list(G.nodes)
    node_coords = np.array([G.nodes[n]["xyz"] for n in node_ids], dtype=float)
    tree = cKDTree(node_coords)

    for _, row in topology_df.iterrows():
        if pd.isna(row["ChildPathIDs"]):
            continue

        parent_end_xyz = np.array(
            [
                float(row["EndX"]) * 0.25,
                float(row["EndY"]) * 0.25,
                float(row["EndZ"]),
            ],
            dtype=float,
        )

        _, parent_idx = tree.query(parent_end_xyz)
        parent_node = node_ids[parent_idx]

        child_ids = [
            int(x.strip())
            for x in str(row["ChildPathIDs"]).split(",")
            if x.strip()
        ]

        for child_id in child_ids:
            child_rows = topology_df.loc[
                topology_df["PathID"].astype(int) == child_id
            ]

            if child_rows.empty:
                continue

            child_row = child_rows.iloc[0]

            child_start_xyz = np.array(
                [
                    float(child_row["StartX"]) * 0.25,
                    float(child_row["StartY"]) * 0.25,
                    float(child_row["StartZ"]),
                ],
                dtype=float,
            )

            _, child_idx = tree.query(child_start_xyz)
            child_node = node_ids[child_idx]

            xyz1 = np.array(G.nodes[parent_node]["xyz"], dtype=float)
            xyz2 = np.array(G.nodes[child_node]["xyz"], dtype=float)

            dist = float(np.linalg.norm(xyz1 - xyz2))

            G.add_edge(parent_node, child_node, weight=dist)

    center = tuple(reference_point) if reference_point is not None else None

    return G, center


def find_nearest_node(coord_xyz, kdtree, node_ids, node_coords):
    dist, idx = kdtree.query(np.asarray(coord_xyz, dtype=float))
    return node_ids[idx], float(dist)


def graph_distance_between_xyz(coord_xyz, target_xyz, G, kdtree, node_ids, node_coords):
    source_node, _ = find_nearest_node(coord_xyz, kdtree, node_ids, node_coords)
    target_node, _ = find_nearest_node(target_xyz, kdtree, node_ids, node_coords)

    try:
        dist = nx.shortest_path_length(
            G,
            source=source_node,
            target=target_node,
            weight="weight",
        )
    except nx.NetworkXNoPath:
        dist = np.inf

    return float(dist)


def make_matches_for_branch(
    branch_id: int,
    G10x,
    center10x,
    kdtree,
    node_ids,
    node_coords,
    df_ref,
):
    extra_points_csv = (
        AFTER_MANUAL_EDITS_DIR
        / f"checked_Branch{branch_id}"
        / f"Image2_branch{branch_id}_snapped_bouton_overlap_with_empty.csv"
    )

    out_csv = OUT_DIR / f"branch{branch_id}_Image2_dendritic_distances.csv"

    if not extra_points_csv.exists():
        print(f"[skip] Missing file for branch {branch_id}: {extra_points_csv}")
        return None

    print("\n" + "=" * 80)
    print(f"Processing branch {branch_id}")
    print(f"Input: {extra_points_csv}")
    print(f"Output: {out_csv}")
    print("=" * 80)

    extra_df = load_napari_points_df(extra_points_csv)
    extra_xyz_arr = extra_df[["_x", "_y", "_z"]].to_numpy(dtype=float)

    branch_start_xyz, branch_end_xyz = get_branch_run_endpoints_from_trace(
        df_ref,
        branch_id=branch_id,
        branch_col="path",
        coord_cols=("x", "y", "z"),
    )

    nearest_node_ids = []
    nearest_node_coords = []
    nearest_dists = []
    dist_to_center = []
    dist_to_branch_start = []
    dist_to_branch_end = []

    for p in extra_xyz_arr:
        nn, d = find_nearest_node(p, kdtree, node_ids, node_coords)

        nearest_node_ids.append(nn)
        nearest_node_coords.append(G10x.nodes[nn]["xyz"])
        nearest_dists.append(d)

        dist_to_center.append(
            graph_distance_between_xyz(
                p,
                center10x,
                G10x,
                kdtree,
                node_ids,
                node_coords,
            )
        )

        dist_to_branch_start.append(
            graph_distance_between_xyz(
                p,
                branch_start_xyz,
                G10x,
                kdtree,
                node_ids,
                node_coords,
            )
        )

        dist_to_branch_end.append(
            graph_distance_between_xyz(
                p,
                branch_end_xyz,
                G10x,
                kdtree,
                node_ids,
                node_coords,
            )
        )

    nearest_node_coords = np.array(nearest_node_coords, dtype=float)
    nearest_dists = np.array(nearest_dists, dtype=float)
    dist_to_center = np.array(dist_to_center, dtype=float)
    dist_to_branch_start = np.array(dist_to_branch_start, dtype=float)
    dist_to_branch_end = np.array(dist_to_branch_end, dtype=float)

    matches = pd.DataFrame(
        {
            "branch_id": branch_id,
            "label": extra_df["label"].values
            if "label" in extra_df.columns
            else np.array([None] * len(extra_xyz_arr)),
            "type": extra_df["type"].values
            if "type" in extra_df.columns
            else np.array([None] * len(extra_xyz_arr)),
            "axis-2": extra_xyz_arr[:, 0],
            "axis-1": extra_xyz_arr[:, 1],
            "axis-0": extra_xyz_arr[:, 2],
            "nearest_node_id": [str(x) for x in nearest_node_ids],
            "nearest10x_x": nearest_node_coords[:, 0],
            "nearest10x_y": nearest_node_coords[:, 1],
            "nearest10x_z": nearest_node_coords[:, 2],
            "euclid_dist_to_graph_node": nearest_dists,
            "dendritic_dist_to_center": dist_to_center,
            "dendritic_dist_to_branch_start": dist_to_branch_start,
            "dendritic_dist_to_branch_end": dist_to_branch_end,
            "branch_start_x": branch_start_xyz[0],
            "branch_start_y": branch_start_xyz[1],
            "branch_start_z": branch_start_xyz[2],
            "branch_end_x": branch_end_xyz[0],
            "branch_end_y": branch_end_xyz[1],
            "branch_end_z": branch_end_xyz[2],
        }
    )

    if "bouton_overlap_fracr2" in extra_df.columns:
        matches["bouton_overlap_fracr2"] = extra_df["bouton_overlap_fracr2"].values

    if "bouton overlap percent" in extra_df.columns:
        matches["bouton overlap percent"] = extra_df["bouton overlap percent"].values

    matches.to_csv(out_csv, index=False)
    print(matches.head(10))
    print(f"Saved: {out_csv}")

    return matches, extra_xyz_arr, nearest_node_coords, node_coords


def show_napari_view(extra_xyz_arr, nearest_node_coords, node_coords, center10x):
    assigned_nodes_xyz = np.unique(nearest_node_coords, axis=0)
    assigned_set = set(map(tuple, assigned_nodes_xyz))
    all_nodes_set = set(map(tuple, node_coords))

    unassigned_nodes_xyz = np.array(
        [n for n in all_nodes_set if n not in assigned_set],
        dtype=float,
    )

    counts = Counter(map(tuple, nearest_node_coords))
    assigned_sizes = np.array(
        [counts[tuple(n)] for n in assigned_nodes_xyz],
        dtype=float,
    )

    viewer = napari.Viewer(ndisplay=3)

    viewer.add_points(
        extra_xyz_arr[:, [2, 1, 0]],
        name="extra_points",
        size=8,
        face_color=[1.0, 0.2, 0.2],
        opacity=1.0,
    )

    viewer.add_points(
        np.array(center10x, dtype=float)[None, [2, 1, 0]],
        name="center_reference_only",
        size=12,
        face_color=[0.0, 1.0, 1.0],
        opacity=1.0,
    )

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

    extra_zyx = extra_xyz_arr[:, [2, 1, 0]]
    nearest_zyx = nearest_node_coords[:, [2, 1, 0]]

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


def main():
    df_ref = pd.read_csv(TRACE_CSV)
    topology_df = pd.read_csv(TOPOLOGY_CSV)

    G10x, center10x = build_dendrite_tree_from_xyz_df(
        full_trace_df=df_ref,
        topology_df=topology_df,
        branch_col="path" if "path" in df_ref.columns else "__no_path__",
        coord_cols=("x", "y", "z"),
        max_edge_length=MAX_EDGE_LENGTH,
        reference_point=REFERENCE_POINT_FOR_CENTER,
    )

    if center10x is None:
        raise RuntimeError("center10x is None. Check reference point.")

    node_ids = list(G10x.nodes)
    node_coords = np.array(
        [G10x.nodes[n]["xyz"] for n in node_ids],
        dtype=float,
    )
    kdtree = cKDTree(node_coords)

    last_for_view = None

    for branch_id in BRANCH_IDS:
        result = make_matches_for_branch(
            branch_id=branch_id,
            G10x=G10x,
            center10x=center10x,
            kdtree=kdtree,
            node_ids=node_ids,
            node_coords=node_coords,
            df_ref=df_ref,
        )

        if result is not None:
            last_for_view = result

    if SHOW_NAPARI and last_for_view is not None:
        _, extra_xyz_arr, nearest_node_coords, node_coords = last_for_view
        show_napari_view(extra_xyz_arr, nearest_node_coords, node_coords, center10x)


if __name__ == "__main__":
    main()

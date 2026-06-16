import numpy as np
import pandas as pd
import napari
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter
from pathlib import Path
import logging
from datetime import datetime

#manually edit
BASE_DIR = Path(
    r"Z:\Joe\AnalysisCode_forSharing\ExampleData\FullyAnalyzed\Analyzed_Data"
)

#manually edit
TRACE_CSV = BASE_DIR / "SNTTrace/Full_Cell_SNT/SOM056_Image4_FullTrace_xyzCoordinates.csv"

#manually edit
TOPOLOGY_CSV = r'Z:\Joe\AnalysisCode_forSharing\ExampleData\FullyAnalyzed\Analyzed_Data\SNTTrace\Full_Cell_SNT\SOM056_Image4_SNT_BranchInfo.csv'
AFTER_MANUAL_EDITS_DIR = BASE_DIR / "AlignmentAndChecking/AfterCorrections"

#manually edit
OUT_DIR = Path(r"Z:\Joe\AnalysisCode_forSharing\ExampleData\FullyAnalyzed\Analyzed_Data\AlignmentAndChecking\AfterCorrections\DendriticDistances")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCH_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

MAX_EDGE_LENGTH = 5.0
SHOW_NAPARI = False


# ============================================================================
# SETUP LOGGING
# ============================================================================
def setup_logging():
    """Configure logging to write to a file in OUT_DIR."""
    log_file = OUT_DIR / f"dendritic_distances_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("DENDRITIC DISTANCE CALCULATION LOG")
    logger.info("=" * 80)
    
    return logger, log_file


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


# ============================================================================
# DIAGNOSTIC FUNCTION: Print topology structure
# ============================================================================
def print_topology_structure(logger, topology_df, branch_ids_to_check=[1, 2, 3, 4]):
    """Log the parent-child relationships from the topology."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TOPOLOGY STRUCTURE REPORT")
    logger.info("=" * 80)
    
    for branch_id in branch_ids_to_check:
        branch_name = f"branch{branch_id}"
        matching_rows = topology_df[
            topology_df["PathName"].astype(str).str.lower() == branch_name.lower()
        ]
        
        if matching_rows.empty:
            logger.info(f"\n{branch_name}: NOT FOUND")
            continue
        
        row = matching_rows.iloc[0]
        logger.info(f"\n{branch_name}:")
        logger.info(f"  PathID: {row['PathID']}")
        logger.info(f"  PrimaryPath: {row['PrimaryPath']}")
        logger.info(f"  StartsOnPath: {row['StartsOnPath']} (type: {type(row['StartsOnPath']).__name__})")
        logger.info(f"  ChildPathIDs: {row['ChildPathIDs']}")
        logger.info(f"  StartXYZ: ({row['StartX']}, {row['StartY']}, {row['StartZ']})")
        logger.info(f"  EndXYZ: ({row['EndX']}, {row['EndY']}, {row['EndZ']})")
        
        # Try to find parent
        if pd.notna(row['StartsOnPath']) and str(row['StartsOnPath']).strip() != "":
            try:
                parent_id = int(float(row['StartsOnPath']))
                parent_rows = topology_df[topology_df['PathID'] == parent_id]
                if not parent_rows.empty:
                    parent_name = parent_rows.iloc[0]['PathName']
                    logger.info(f"  Parent: {parent_name} (PathID {parent_id})")
            except (ValueError, TypeError):
                logger.info(f"  Parent: Could not parse StartsOnPath")
        else:
            logger.info(f"  Parent: NONE (this is a primary dendrite)")
    
    logger.info("\n" + "=" * 80 + "\n")


# ============================================================================
# ENHANCED: Get first-order dendrite endpoint with detailed debugging
# ============================================================================
def get_first_order_dendrite_end_node_from_graph(
    logger,
    branch_id: int,
    topology_df: pd.DataFrame,
    G: nx.Graph,
) -> tuple:
    """
    Trace backwards through the parent chain from branch_id until reaching
    a primary dendrite, then return the PROXIMAL END node (start) of that 
    first-order dendrite.
    """
    
    branch_name = f"branch{branch_id}"
    matching_rows = topology_df[
        topology_df["PathName"].astype(str).str.lower() == branch_name.lower()
    ]
    
    if matching_rows.empty:
        available_paths = sorted(topology_df["PathName"].unique())
        logger.error(f"ERROR: Available paths in topology: {available_paths}")
        raise ValueError(f"Branch '{branch_name}' not found in topology CSV.")
    
    current_path_id = int(matching_rows.iloc[0]["PathID"])
    current_row = matching_rows.iloc[0]
    current_path_name = current_row["PathName"]
    
    logger.info(f"  Starting trace from: {current_path_name} (PathID {current_path_id})")
    
    trace_chain = [current_path_name]
    
    iteration = 0
    while iteration < 100:
        iteration += 1
        is_primary = str(current_row["PrimaryPath"]).upper() == "TRUE"
        
        logger.info(f"    [{iteration}] {current_path_name}: PrimaryPath={is_primary}")
        
        if is_primary:
            logger.info(f"  ✓ Found primary dendrite: {current_path_name}")
            logger.info(f"    Full chain: {' → '.join(trace_chain)}")
            
            path_nodes = [node for node in G.nodes() 
                         if G.nodes[node]["path_name"].lower() == current_path_name.lower()]
            
            if not path_nodes:
                raise ValueError(f"No nodes found in graph for path '{current_path_name}'")
            
            # DIAGNOSTIC: Log both ends
            distal_node = path_nodes[-1]
            proximal_node = path_nodes[0]
            
            distal_xyz = G.nodes[distal_node]["xyz"]
            proximal_xyz = G.nodes[proximal_node]["xyz"]
            
            logger.info(f"    Available nodes in {current_path_name}: {len(path_nodes)} nodes")
            logger.info(f"      DISTAL (path_nodes[-1]): {distal_node} at {distal_xyz}")
            logger.info(f"      PROXIMAL (path_nodes[0]): {proximal_node} at {proximal_xyz}")
            logger.info(f"    >>> USING: PROXIMAL END <<<")
            
            return proximal_node, proximal_xyz
        
        starts_on_path = current_row["StartsOnPath"]
        
        logger.info(f"      StartsOnPath: '{starts_on_path}' (type: {type(starts_on_path).__name__})")
        
        if (pd.isna(starts_on_path) or 
            str(starts_on_path).strip() == "" or 
            str(starts_on_path).upper() == "NAN"):
            
            logger.error(f"  ✗ Reached end of parent chain without finding a primary dendrite!")
            logger.error(f"    Trace chain: {' → '.join(trace_chain)}")
            raise ValueError(
                f"Cannot find primary dendrite for branch {branch_id}. "
                f"Trace ended at '{current_path_name}' with no parent. "
                f"Trace chain was: {' → '.join(trace_chain)}"
            )
        
        try:
            parent_path_id = int(float(starts_on_path))
        except (ValueError, TypeError) as e:
            logger.error(f"  ✗ Could not parse StartsOnPath value: {starts_on_path}")
            raise ValueError(f"Invalid StartsOnPath value '{starts_on_path}' for {current_path_name}: {e}")
        
        parent_rows = topology_df[topology_df["PathID"] == parent_path_id]
        
        if parent_rows.empty:
            logger.error(f"  ✗ Parent PathID {parent_path_id} not found in topology!")
            raise ValueError(f"Parent path ID {parent_path_id} not found in topology CSV.")
        
        current_path_id = parent_path_id
        current_row = parent_rows.iloc[0]
        current_path_name = current_row["PathName"]
        trace_chain.append(current_path_name)
        logger.info(f"      Moving to parent: {current_path_name}")
    
    raise ValueError("Maximum iterations reached while tracing parent chain (possible circular reference)")


def build_dendrite_tree_from_xyz_df(
    logger,
    full_trace_df: pd.DataFrame,
    topology_df: pd.DataFrame,
    branch_col: str = "path",
    coord_cols=("x", "y", "z"),
    max_edge_length=5.0,
):
    """
    Build a graph with properly connected paths.
    """
    G = nx.Graph()
    
    path_to_nodes = {}

    if branch_col in full_trace_df.columns:
        groups = full_trace_df.groupby(branch_col, sort=False)
    else:
        groups = [(0, full_trace_df)]

    logger.info("  Building nodes from trace coordinates...")
    for path_name, group in groups:
        group = group.reset_index(drop=True)
        coords = group.loc[:, coord_cols].dropna().to_numpy(dtype=float)

        coords[:, 0] *= 0.25
        coords[:, 1] *= 0.25

        if len(coords) < 1:
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

        path_to_nodes[str(path_name)] = node_ids
        
        for i in range(len(node_ids) - 1):
            n1 = node_ids[i]
            n2 = node_ids[i + 1]

            xyz1 = np.array(G.nodes[n1]["xyz"], dtype=float)
            xyz2 = np.array(G.nodes[n2]["xyz"], dtype=float)

            dist = float(np.linalg.norm(xyz1 - xyz2))

            if dist < max_edge_length:
                G.add_edge(n1, n2, weight=dist)

    logger.info(f"  Created {len(path_to_nodes)} paths with total {len(G.nodes())} nodes")

    logger.info(f"  Connecting parent-child paths using topology...")
    
    connection_count = 0
    for _, row in topology_df.iterrows():
        if pd.isna(row["ChildPathIDs"]):
            continue

        parent_path_name = str(row["PathName"])
        
        if parent_path_name not in path_to_nodes:
            logger.warning(f"    WARNING: Parent path '{parent_path_name}' not in graph")
            continue
            
        parent_nodes = path_to_nodes[parent_path_name]
        if not parent_nodes:
            continue
            
        parent_end_node = parent_nodes[-1]

        child_id_str = str(row["ChildPathIDs"])
        child_ids = [
            int(x.strip())
            for x in child_id_str.split(",")
            if x.strip()
        ]

        for child_id in child_ids:
            child_rows = topology_df[topology_df["PathID"].astype(int) == child_id]

            if child_rows.empty:
                continue

            child_row = child_rows.iloc[0]
            child_path_name = str(child_row["PathName"])
            
            if child_path_name not in path_to_nodes:
                logger.warning(f"    WARNING: Child path '{child_path_name}' not in graph")
                continue
                
            child_nodes = path_to_nodes[child_path_name]
            if not child_nodes:
                continue
                
            child_start_node = child_nodes[0]

            parent_xyz = np.array(G.nodes[parent_end_node]["xyz"], dtype=float)
            child_xyz = np.array(G.nodes[child_start_node]["xyz"], dtype=float)
            dist = float(np.linalg.norm(parent_xyz - child_xyz))
            
            if dist < max_edge_length * 2:
                G.add_edge(parent_end_node, child_start_node, weight=dist)
                connection_count += 1
                logger.info(f"    Connected {parent_path_name}[-1] → {child_path_name}[0] (distance: {dist:.2f} µm)")

    logger.info(f"  Total parent-child connections made: {connection_count}")
    return G


def find_nearest_node(coord_xyz, kdtree, node_ids, node_coords):
    dist, idx = kdtree.query(np.asarray(coord_xyz, dtype=float))
    return node_ids[idx], float(dist)


def print_graph_connectivity_info(logger, G, node_ids, branch_id, synapse_nearest_node, first_order_end_node):
    """Log information about which connected components nodes belong to."""
    
    logger.info(f"\n--- Graph Connectivity Debug for Branch {branch_id} ---")
    
    components = list(nx.connected_components(G))
    logger.info(f"Total connected components: {len(components)}")
    
    if len(components) <= 20:
        logger.info(f"Component sizes: {sorted([len(c) for c in components], reverse=True)}")
    else:
        logger.info(f"Component sizes (top 20): {sorted([len(c) for c in components], reverse=True)[:20]}")
    
    synapse_component = None
    for i, component in enumerate(components):
        if synapse_nearest_node in component:
            synapse_component = i
            logger.info(f"Synapse nearest node {synapse_nearest_node} is in component {i} (size {len(component)})")
            sample_paths = set([G.nodes[n]['path_name'] for n in list(component)[:10]])
            logger.info(f"  Sample paths: {sample_paths}")
            break
    
    endpoint_component = None
    for i, component in enumerate(components):
        if first_order_end_node in component:
            endpoint_component = i
            logger.info(f"First-order endpoint node {first_order_end_node} is in component {i} (size {len(component)})")
            sample_paths = set([G.nodes[n]['path_name'] for n in list(component)[:10]])
            logger.info(f"  Sample paths: {sample_paths}")
            break
    
    if synapse_component is not None and endpoint_component is not None:
        if synapse_component == endpoint_component:
            logger.info(f"✓ Both nodes are in the SAME component (component {synapse_component})")
        else:
            logger.error(f"✗ NODES ARE IN DIFFERENT COMPONENTS! ({synapse_component} vs {endpoint_component})")
    
    logger.info("--- End Debug ---\n")


def graph_distance_between_nodes(source_node, target_node, G):
    """Calculate shortest path distance between two nodes in the graph."""
    try:
        dist = nx.shortest_path_length(G, source=source_node, target=target_node, weight="weight")
    except nx.NetworkXNoPath:
        dist = np.inf
    
    return float(dist)


def make_matches_for_branch(
    logger,
    branch_id: int,
    G10x,
    topology_df: pd.DataFrame,
    kdtree,
    node_ids,
    node_coords,
    df_ref,
):
    
    extra_points_csv = (
        AFTER_MANUAL_EDITS_DIR
        / f"Corrected_Branch{branch_id}"
        #manually edit image number here (P42 session image)
        / f"Image4_branch{branch_id}_snapped_bouton_overlap_with_empty.csv"
    )
    #manually edit image number here (P42 session image)
    out_csv = OUT_DIR / f"branch{branch_id}_Image4_dendritic_distances.csv"

    if not extra_points_csv.exists():
        logger.info(f"[skip] Missing file for branch {branch_id}: {extra_points_csv}")
        return None

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Processing branch {branch_id}")
    logger.info(f"Input: {extra_points_csv}")
    logger.info(f"Output: {out_csv}")
    logger.info("=" * 80)

    logger.info(f"Tracing branch {branch_id} back to first-order dendrite...")
    try:
        first_order_end_node, first_order_endpoint_xyz = get_first_order_dendrite_end_node_from_graph(
            logger, branch_id, topology_df, G10x
        )
        logger.info(f"  First-order dendrite endpoint: {first_order_endpoint_xyz}")
    except ValueError as e:
        logger.error(f"  ERROR: {e}")
        return None

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

    for idx, p in enumerate(extra_xyz_arr):
        nn, d = find_nearest_node(p, kdtree, node_ids, node_coords)

        nearest_node_ids.append(nn)
        nearest_node_coords.append(G10x.nodes[nn]["xyz"])
        nearest_dists.append(d)

        if idx == 0:
            print_graph_connectivity_info(logger, G10x, node_ids, branch_id, nn, first_order_end_node)

        dist_to_center.append(
            graph_distance_between_nodes(nn, first_order_end_node, G10x)
        )

        dist_to_branch_start.append(
            graph_distance_between_nodes(nn, find_nearest_node(branch_start_xyz, kdtree, node_ids, node_coords)[0], G10x)
        )

        dist_to_branch_end.append(
            graph_distance_between_nodes(nn, find_nearest_node(branch_end_xyz, kdtree, node_ids, node_coords)[0], G10x)
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
            "first_order_dendrite_endpoint_x": first_order_endpoint_xyz[0],
            "first_order_dendrite_endpoint_y": first_order_endpoint_xyz[1],
            "first_order_dendrite_endpoint_z": first_order_endpoint_xyz[2],
        }
    )

    if "bouton_overlap_frac_r2" in extra_df.columns:
        matches["bouton_overlap_frac_r2"] = extra_df["bouton_overlap_frac_r2"].values

    if "bouton overlap percent" in extra_df.columns:
        matches["bouton overlap percent"] = extra_df["bouton overlap percent"].values

    matches.to_csv(out_csv, index=False)
    logger.info("\nFirst 10 rows of results:")
    logger.info(matches[["label", "type", "dendritic_dist_to_center"]].to_string())
    logger.info(f"\nSaved: {out_csv}")

    return matches, extra_xyz_arr, nearest_node_coords, node_coords


def main():
    # Setup logging
    logger, log_file = setup_logging()
    logger.info(f"Log file: {log_file}")
    
    df_ref = pd.read_csv(TRACE_CSV)
    topology_df = pd.read_csv(TOPOLOGY_CSV)

    # Log topology structure for debugging
    print_topology_structure(logger, topology_df, branch_ids_to_check=BRANCH_IDS)

    logger.info("Building dendrite graph...")
    G10x = build_dendrite_tree_from_xyz_df(
        logger,
        full_trace_df=df_ref,
        topology_df=topology_df,
        branch_col="path" if "path" in df_ref.columns else "__no_path__",
        coord_cols=("x", "y", "z"),
        max_edge_length=MAX_EDGE_LENGTH,
    )

    logger.info(f"\nGraph Statistics:")
    logger.info(f"  Nodes: {len(G10x.nodes())}")
    logger.info(f"  Edges: {len(G10x.edges())}")
    logger.info(f"  Connected components: {nx.number_connected_components(G10x)}")

    node_ids = list(G10x.nodes)
    node_coords = np.array(
        [G10x.nodes[n]["xyz"] for n in node_ids],
        dtype=float,
    )
    kdtree = cKDTree(node_coords)

    last_for_view = None

    for branch_id in BRANCH_IDS:
        result = make_matches_for_branch(
            logger,
            branch_id=branch_id,
            G10x=G10x,
            topology_df=topology_df,
            kdtree=kdtree,
            node_ids=node_ids,
            node_coords=node_coords,
            df_ref=df_ref,
        )

        if result is not None:
            last_for_view = result

    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"All output files saved to: {OUT_DIR}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import networkx as nx
from numpy.ma.core import append
from scipy.spatial import cKDTree


synapses = pd.read_csv("synapses_ROI_PV4569.csv")


print(synapses.iloc[:, 10].unique())

PV_positive = synapses[synapses.iloc[:, 10].str.contains('PVPositive')].copy()

PV_negative = synapses[synapses.iloc[:, 10].str.contains('PVNegative')].copy()

spine = synapses[synapses.iloc[:, 10].str.contains('Spine')].copy()

shaft = synapses[synapses.iloc[:, 10].str.contains('Shaft')].copy()


def build_dendrite_tree(full_trace_df, max_edge_length=5.0):
    G = nx.Graph()
    counter = 0

    branch_col = full_trace_df.columns[0]
    coord_cols = full_trace_df.columns[1:4]

    for branch_id, group in full_trace_df.groupby(branch_col):
        coords = group[coord_cols].dropna().values
        if len(coords) < 2:
            continue
        counter += 1
        for i in range(len(coords) - 1):
            p1 = tuple(coords[i])
            p2 = tuple(coords[i + 1])
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if p1 != p2 and dist < max_edge_length:
                G.add_edge(p1, p2, weight=dist)

    print('Branch count:', counter)
    print('Connected components:', nx.number_connected_components(G))

    if nx.is_tree(G):
        print("Graph is a valid tree.")
    else:
        print("WARNING: Graph is NOT a tree. Check for disconnected components or extra links.")

    # use it for centroid calculation
    reference_point = (115.0, 105.0, 50)

    components = list(nx.connected_components(G))
    closest_terminal_nodes = []

    for comp in components:
        terminal_nodes = [node for node in comp if G.degree[node] == 1]
        if not terminal_nodes:
            continue

        terminal_coords = np.array(terminal_nodes)
        dists = np.linalg.norm(terminal_coords - np.array(reference_point), axis=1)
        closest_terminal = terminal_nodes[np.argmin(dists)]
        closest_terminal_nodes.append(closest_terminal)

    if closest_terminal_nodes:
        closest_coords = np.array(closest_terminal_nodes)
        centroid = np.mean(closest_coords, axis=0)
        print(f'Centroid of closest terminal nodes: {centroid}')
        centroid = tuple(centroid)
        G.add_node(centroid)
        for node in closest_terminal_nodes:
            dist = np.linalg.norm(np.array(node) - np.array(centroid))
            G.add_edge(node, centroid, weight=dist)
    else:
        centroid = None
        print("No terminal nodes found to compute centroid.")

    print('Final connected components:', nx.number_connected_components(G))
    return G, centroid


def find_nearest_node(coord, nodes_kdtree, node_coords):

    #Find the nearest node to the given synapse.

    dist, idx = nodes_kdtree.query(coord)
    return tuple(node_coords[idx])


def dendritic_distance(syn1, syn2, tree):

    G = tree

    # create tree
    node_coords = np.array(G.nodes)
    kdtree = cKDTree(node_coords)

    # find closest nodes in the tree for each synapse
    node1 = find_nearest_node(syn1, kdtree, node_coords)
    node2 = find_nearest_node(syn2, kdtree, node_coords)
    try:
        distance = nx.shortest_path_length(G, source=node1, target=node2, weight='weight')
    except nx.NetworkXNoPath:
        #print(" No path found")
        return np.inf
    return distance




full_trace = pd.read_csv("Full_trace_PV4569_2p.csv")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dendritic_graph_3d(G, node_size=1, edge_color='gray'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    for u, v in G.edges():
        xs = [u[0], v[0]]
        ys = [u[1], v[1]]
        zs = [u[2], v[2]]
        ax.plot(xs, ys, zs, color=edge_color, linewidth=0.5)


    nodes = np.array(G.nodes)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=node_size, color='black', alpha=0.3)

    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Z (µm)')
    ax.set_title('Dendritic Arbor Graph')
    plt.tight_layout()
    plt.show()

G,center = build_dendrite_tree(full_trace)
print(nx.number_connected_components(G))
plot_dendritic_graph_3d(G)

print('the center '+str(center))
syn1 = synapses.iloc[0][12:15].values
syn2 = synapses.iloc[110][12:15].values

dist = dendritic_distance(syn1, syn2, G)
print(f"Dendritic path distance: {dist:.2f} µm")









node_coords = np.array(G.nodes)
kdtree = cKDTree(node_coords)

# Extract XYZ positions
PV_pos_coords = PV_positive.iloc[:, 12:15].values
PV_neg_coords = PV_negative.iloc[:, 12:15].values



def graph_distance_to_center(coord, center, G, kdtree, node_coords):

    #Calculates the dendritic distance from a synapse to the center.

    nearest_node = find_nearest_node(coord, kdtree, node_coords)
    try:
        path = nx.shortest_path(G, source=nearest_node, target=center, weight='weight')
        distance = nx.shortest_path_length(G, source=nearest_node, target=tuple(center), weight='weight')
        branch_count = sum(1 for n in path if G.degree[n] >= 3)-1
    except nx.NetworkXNoPath:
        print('inf!!!!!!!!!!!!!!')
        distance = np.inf
        branch_count = np.nan
    return distance,branch_count




# Extract synapses coords
all_syn_coords = synapses.iloc[:, 12:15].values

# Compute distances from synapse to the center
distances_to_center = []
branched_point_from_soma=[]
for coord in all_syn_coords:
    dist,branched = graph_distance_to_center(coord, center, G, kdtree, node_coords)
    distances_to_center.append(dist)
    branched_point_from_soma.append(branched)

synapses['DistanceToCenter'] = distances_to_center
synapses['BranchPoints'] = branched_point_from_soma

synapses.to_csv("SynapseROIs_with_DistanceToCenter.csv", index=False)

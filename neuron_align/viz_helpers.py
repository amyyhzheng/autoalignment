# viz_helpers.py
# Tiny plotting helpers for neuron_align results (matplotlib only).
# Use after you have `res = compute(settings)`.

from typing import List, Optional
import matplotlib.pyplot as plt

# Types you’ll pass in
# - settings: neuron_align.config.Settings
# - res: neuron_align.computation.ComputationResult


def plot_branch_and_fiducials(res, tp: int = 0, show_markers: bool = True):
    """
    Plot normalized branch XY for a timepoint, with placed fiducials, mapped fiducials,
    and (optionally) marker→branch projections.
    """
    bx = [p[0] for p in res.normalized_branch[tp]]
    by = [p[1] for p in res.normalized_branch[tp]]

    # placed fiducials (normalized)
    fx = [p[0] for p in res.normalized_fiducials[tp]]
    fy = [p[1] for p in res.normalized_fiducials[tp]]

    # fiducials mapped to nearest branch point
    mapped_f = [res.normalized_branch[tp][idx] for idx in res.closest_branch_idx_fids[tp]]
    mfx = [p[0] for p in mapped_f]
    mfy = [p[1] for p in mapped_f]

    plt.figure(figsize=(6, 5))
    plt.plot(bx, by, lw=1.5, label="branch")
    if fx:
        plt.scatter(fx, fy, marker="o", s=36, label="fiducials (placed)")
    if mfx:
        plt.scatter(mfx, mfy, marker="X", s=64, label="fiducials (mapped)")

    if show_markers and res.closest_branch_idx_markers and res.closest_branch_idx_markers[tp]:
        mapped_m = [res.normalized_branch[tp][idx] for idx in res.closest_branch_idx_markers[tp]]
        mmx = [p[0] for p in mapped_m]
        mmy = [p[1] for p in mapped_m]
        plt.scatter(mmx, mmy, marker=".", s=20, label="markers→branch")

    plt.xlabel("X (µm)"); plt.ylabel("Y (µm)")
    plt.title(f"Timepoint {tp+1} — branch & fiducials")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_markers_along_branch(res, title: str = "Markers along branch (scaled)"):
    """
    Scatter markers by their final scaled cumulative distance (X) and timepoint (Y).
    Handy to eyeball cluster separation across sessions.
    """
    plt.figure(figsize=(7.5, 5))
    for tp in range(len(res.final_marker_distance)):
        xs = res.final_marker_distance[tp]
        ys = [tp] * len(xs)
        if xs:
            plt.scatter(xs, ys, s=16, label=f"TP{tp+1}")
    plt.yticks(range(len(res.final_marker_distance)), [f"TP{t+1}" for t in range(len(res.final_marker_distance))])
    plt.xlabel("Scaled cumulative distance (µm)")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_segment_scaling(res, tp: int = 0):
    """
    Bar plot comparing unscaled vs scaled segment lengths for a timepoint.
    Lets you confirm per-segment scale factors did what you expect.
    """
    import numpy as np

    unscaled = res.seg_lengths_all[tp] if res.seg_lengths_all else []
    scaled = []
    if res.seg_lengths_all and res.scale_factors_all:
        scaled = [l * sf for l, sf in zip(res.seg_lengths_all[tp], res.scale_factors_all[tp])]

    n = len(unscaled)
    if n == 0:
        print("No segments to plot for this timepoint.")
        return

    x = np.arange(n)
    w = 0.42

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - w/2, unscaled, width=w, label="unscaled")
    plt.bar(x + w/2, scaled,   width=w, label="scaled")
    plt.xlabel("Segment index")
    plt.ylabel("Length (µm)")
    plt.title(f"Timepoint {tp+1} — segment lengths (pre/post scaling)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fiducial_index_diagnostics(res, tp: int = 0):
    """
    Visual check of fiducial→branch index mapping. If two fiducials map to the same index,
    you’ll see duplicate dots on the same X position.
    """
    import numpy as np

    idxs = sorted([0, len(res.normalized_branch[tp]) - 1] + res.closest_branch_idx_fids[tp])
    y = [1] * len(idxs)
    plt.figure(figsize=(7, 1.6))
    plt.scatter(idxs, y, s=36)
    plt.yticks([])
    plt.xlabel("Branch index")
    plt.title(f"Timepoint {tp+1} — start/end + fiducial branch indices")
    plt.tight_layout()
    plt.show()


def plot_cluster_bins(cluster_list: List[List], title: str = "Clusters (one row per cluster)"):
    """
    Visualize cluster contents from export_grouping_csv() output (the 'cluster_list' it returns).
    Each row is a cluster; boxes show which timepoints have a marker vs 'NA'.
    """
    import numpy as np

    if not cluster_list:
        print("No clusters to visualize.")
        return

    # cluster_list is a list of clusters; each cluster is [(tp, idx, pos) or (tp,'NA','NA')]
    clusters = cluster_list
    tp_max = 1 + max(tp for cl in clusters for (tp, _, _) in cl)

    mat = []
    for cl in clusters:
        row = [0] * tp_max
        for (tp, idx, _pos) in cl:
            row[tp] = 0 if idx == "NA" else 1
        mat.append(row)

    mat = np.array(mat)  # shape: [n_clusters, n_timepoints]

    plt.figure(figsize=(max(6, tp_max * 0.8), max(2, len(clusters) * 0.25)))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="present (1) / missing (0)")
    plt.xlabel("Timepoint")
    plt.ylabel("Cluster")
    plt.title(title)
    plt.xticks(range(tp_max), [f"TP{t+1}" for t in range(tp_max)])
    plt.yticks(range(len(clusters)), [f"G{i+1}" for i in range(len(clusters))])
    plt.tight_layout()
    plt.show()

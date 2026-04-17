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

def plot_markers_along_branch(
    res,
    title: str = "Markers along branch (scaled)",
    show_landmarks: bool = True,
):
    """
    Scatter markers by their final scaled cumulative distance (X) and timepoint (Y).
    Optionally overlay vertical lines at landmark / segment-boundary positions.

    Landmarks are taken from res.cumdist_scaled[0], which contains the cumulative
    scaled distance at each segment boundary (0, landmark1, landmark2, ..., total).
    """
    plt.figure(figsize=(7.5, 5))

    # --- markers ---
    n_tp = len(res.final_marker_distance)
    for tp in range(n_tp):
        xs = res.final_marker_distance[tp]
        ys = [tp] * len(xs)
        if xs:
            plt.scatter(xs, ys, s=16, label=f"TP{tp+1}")

    # --- landmark lines (segment boundaries) ---
    if show_landmarks and getattr(res, "cumdist_scaled", None):
        if res.cumdist_scaled and len(res.cumdist_scaled[0]) >= 2:
            boundaries = res.cumdist_scaled[0]  # shared across TPs after scaling
            # internal boundaries only: skip 0 and total length
            internal_bounds = boundaries[1:-1]

            first = True
            for x in internal_bounds:
                plt.axvline(
                    x,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                    label="landmarks" if first else None,
                )
                first = False

    plt.yticks(
        range(n_tp),
        [f"TP{t+1}" for t in range(n_tp)],
    )
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
def plot_markers_along_branch_with_ids(
    res,
    cluster_list,
    title: str = "Markers along branch (scaled) with cluster IDs",
    show_landmarks: bool = True,
    text_fontsize: int = 7,
    text_dy: float = 0.15,
):
    """
    Scatter markers by their final scaled cumulative distance (X) and timepoint (Y),
    and annotate each marker with the cluster ID assigned by clustering.

    Parameters
    ----------
    res : ComputationResult
        Output of neuron_align.computation.compute().
    cluster_list : List[List[(tp, idx, pos)]]
        The cluster_list returned by export_grouping_csv().
        Each cluster is a list of (tp, idx, pos) or (tp, "NA", "NA").
        'idx' is the marker index within that timepoint.
    title : str
        Plot title.
    show_landmarks : bool
        If True, overlay vertical lines at segment boundaries (landmarks).
    text_fontsize : int
        Font size of cluster ID labels next to markers.
    text_dy : float
        Vertical offset for text labels (in y-units, i.e. timepoint index units).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Build mapping: (tp, marker_idx) -> cluster_id (1-based)
    # --------------------------------------------------------
    n_tp = len(res.final_marker_distance)
    marker_to_cluster = [dict() for _ in range(n_tp)]  # one dict per timepoint

    if cluster_list:
        for cid, cluster in enumerate(cluster_list, start=1):
            for (tp, idx, _pos) in cluster:
                # skip timepoints with no marker ("NA")
                if idx == "NA":
                    continue
                # tp is already 0-based in export_grouping_csv / plot_cluster_bins
                marker_to_cluster[tp][idx] = cid

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    plt.figure(figsize=(8.0, 5.0))

    # --- markers + text labels ---
    for tp in range(n_tp):
        xs = res.final_marker_distance[tp]
        if not xs:
            continue

        # Scatter points
        ys = [tp] * len(xs)
        plt.scatter(xs, ys, s=16, label=f"TP{tp+1}")

        # Add cluster ID as text where available
        for m_idx, x in enumerate(xs):
            cid = marker_to_cluster[tp].get(m_idx, None)
            if cid is None:
                continue  # marker not part of any cluster
            # Small vertical offset so the text isn't exactly on the point
            plt.text(
                x,
                tp + text_dy,
                str(cid),
                fontsize=text_fontsize,
                ha="center",
                va="bottom",
            )

    # --- landmark lines (segment boundaries) ---
    if show_landmarks and getattr(res, "cumdist_scaled", None):
        if res.cumdist_scaled and len(res.cumdist_scaled[0]) >= 2:
            boundaries = res.cumdist_scaled[0]  # shared across TPs after scaling
            internal_bounds = boundaries[1:-1]  # skip 0 and total length

            first = True
            for x in internal_bounds:
                plt.axvline(
                    x,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                    label="landmarks" if first else None,
                )
                first = False

    plt.yticks(
        range(n_tp),
        [f"TP{t+1}" for t in range(n_tp)],
    )
    plt.xlabel("Scaled cumulative distance (µm)")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_branch_with_marker_ids(
    res,
    cluster_list,
    tp: int = 0,
    title: str | None = None,
    show_fiducials: bool = True,
    show_unclustered: bool = True,
    text_fontsize: int = 8,
    text_dxy: tuple[float, float] = (0.0, 0.0),
):
    """
    Plot a single timepoint's normalized branch in XY, with all markers mapped
    onto the branch and annotated by their cluster IDs.

    Parameters
    ----------
    res : ComputationResult
        Output of neuron_align.computation.compute().
    cluster_list : list
        cluster_list returned by export_grouping_csv().
        Each entry is a cluster: [(tp, idx, pos), (tp, idx, pos), ...] or (tp,"NA","NA").
        'idx' is the marker index in that timepoint.
    tp : int
        Timepoint index (0-based; 0 == Timepoint 1).
    title : str or None
        Title for the plot. If None, a default will be used.
    show_fiducials : bool
        If True, also draw fiducials/landmarks (mapped to branch).
    show_unclustered : bool
        If False, only show markers that belong to at least one cluster.
    text_fontsize : int
        Font size for cluster ID labels.
    text_dxy : (float, float)
        Small (dx, dy) offset for text labels in branch coordinate units.
    """
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # 1) Build mapping: (tp, marker_idx) -> cluster_id (1-based)
    # --------------------------------------------------------
    n_tp = len(res.final_marker_distance)
    if tp < 0 or tp >= n_tp:
        raise ValueError(f"timepoint tp={tp} out of range [0, {n_tp-1}]")

    marker_to_cluster = [dict() for _ in range(n_tp)]
    if cluster_list:
        for cid, cluster in enumerate(cluster_list, start=1):
            for (tp_c, idx, _pos) in cluster:
                if idx == "NA":
                    continue
                marker_to_cluster[tp_c][idx] = cid

    mapping_tp = marker_to_cluster[tp]

    # --------------------------------------------------------
    # 2) Pull branch + marker coordinates for this TP
    # --------------------------------------------------------
    branch_xy = res.normalized_branch[tp]
    bx = [p[0] for p in branch_xy]
    by = [p[1] for p in branch_xy]

    # markers mapped to branch indices
    marker_branch_idx = res.closest_branch_idx_markers[tp]
    marker_coords = [branch_xy[idx] for idx in marker_branch_idx]

    # optionally filter unclustered markers
    plot_mask = []
    for m_idx in range(len(marker_coords)):
        cid = mapping_tp.get(m_idx, None)
        if cid is None and not show_unclustered:
            plot_mask.append(False)
        else:
            plot_mask.append(True)

    marker_coords_filtered = [
        c for c, keep in zip(marker_coords, plot_mask) if keep
    ]
    kept_indices = [i for i, keep in enumerate(plot_mask) if keep]

    # --------------------------------------------------------
    # 3) Fiducials (landmarks) for this TP, if desired
    # --------------------------------------------------------
    # normalized_fiducials = landmark coordinates (already on branch in our
    # DTW/curvature workflow), closest_branch_idx_fids = indices on branch.
    fx = fy = []
    if show_fiducials and res.normalized_fiducials:
        fx = [p[0] for p in res.normalized_fiducials[tp]]
        fy = [p[1] for p in res.normalized_fiducials[tp]]

    mapped_f = []
    if show_fiducials and res.closest_branch_idx_fids:
        mapped_f = [branch_xy[idx] for idx in res.closest_branch_idx_fids[tp]]

    # --------------------------------------------------------
    # 4) Plot
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))

    # branch backbone
    plt.plot(bx, by, "-k", lw=1.5, label="branch")

    # optional: polyline through fiducial/landmark positions along branch
    if show_fiducials and mapped_f:
        # make sure we plot them in branch-order
        order = sorted(
            range(len(res.closest_branch_idx_fids[tp])),
            key=lambda k: res.closest_branch_idx_fids[tp][k],
        )
        lf_x = [mapped_f[k][0] for k in order]
        lf_y = [mapped_f[k][1] for k in order]
        plt.plot(lf_x, lf_y, "--", lw=1.0, alpha=0.6, label="landmark line")

        # scatter the landmark points themselves
        lm_x = [mapped_f[k][0] for k in order]
        lm_y = [mapped_f[k][1] for k in order]
        plt.scatter(lm_x, lm_y, s=60, marker="s", facecolors="none",
                    label="landmarks")

    # markers
    if marker_coords_filtered:
        mx = [p[0] for p in marker_coords_filtered]
        my = [p[1] for p in marker_coords_filtered]
        plt.scatter(mx, my, s=30, marker="o", label="markers")

        # text labels: cluster IDs
        dx, dy = text_dxy
        for plot_idx, (x, y, z) in zip(kept_indices, marker_coords_filtered):
            cid = mapping_tp.get(plot_idx, None)
            if cid is None:
                continue
            plt.text(
                x + dx,
                y + dy,
                str(cid),
                fontsize=text_fontsize,
                ha="center",
                va="bottom",
            )

    if title is None:
        title = f"Timepoint {tp+1} — markers with cluster IDs on branch"
    plt.title(title)
    plt.xlabel("X (normalized)")
    plt.ylabel("Y (normalized)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_branch_with_cluster_ids(
    res,
    cluster_list,
    tp: int = 0,
    text_fontsize: int = 8,
    text_dxy: tuple[float, float] = (0.0, 0.0),
    show_unclustered: bool = True,
):
    """
    Plot normalized branch XY for a timepoint, with all markers projected onto
    the branch and each marker annotated with its cluster ID (from clustering).

    This is intentionally similar to plot_branch_and_fiducials, but instead of
    generic markers, we write the cluster ID next to each marker.

    Parameters
    ----------
    res : ComputationResult
        Output of neuron_align.computation.compute().
    cluster_list : list
        cluster_list returned by export_grouping_csv(). Each element is a
        cluster: [(tp, idx, pos), ...] where idx is the marker index at that TP
        or "NA".
    tp : int
        Timepoint index (0-based; tp=0 is Timepoint 1).
    text_fontsize : int
        Font size for cluster ID labels.
    text_dxy : (float, float)
        (dx, dy) in branch coordinate units to offset the text relative to
        the marker position.
    show_unclustered : bool
        If False, only markers that belong to some cluster are shown.
    """
    import matplotlib.pyplot as plt

    n_tp = len(res.final_marker_distance)
    if tp < 0 or tp >= n_tp:
        raise ValueError(f"timepoint tp={tp} out of range [0, {n_tp-1}]")

    # --------------------------------------------------------
    # 1) Build mapping (tp, marker_idx) -> cluster_id (1-based)
    #    marker_idx is the index in that TP's marker list
    # --------------------------------------------------------
    marker_to_cluster = [dict() for _ in range(n_tp)]
    if cluster_list:
        for cid, cluster in enumerate(cluster_list, start=1):
            for (tp_c, idx, _pos) in cluster:
                if idx == "NA":
                    continue
                marker_to_cluster[tp_c][idx] = cid

    mapping_tp = marker_to_cluster[tp]

    # --------------------------------------------------------
    # 2) Branch coordinates (normalized)
    # --------------------------------------------------------
    branch = res.normalized_branch[tp]
    bx = [p[0] for p in branch]
    by = [p[1] for p in branch]

    # --------------------------------------------------------
    # 3) Fiducials (same as plot_branch_and_fiducials)
    #     – no landmark line, just the points if you want to see them
    # --------------------------------------------------------
    fx = [p[0] for p in res.normalized_fiducials[tp]] if res.normalized_fiducials else []
    fy = [p[1] for p in res.normalized_fiducials[tp]] if res.normalized_fiducials else []

    mapped_f = []
    if res.closest_branch_idx_fids:
        mapped_f = [branch[idx] for idx in res.closest_branch_idx_fids[tp]]
    mfx = [p[0] for p in mapped_f]
    mfy = [p[1] for p in mapped_f]

    # --------------------------------------------------------
    # 4) Marker positions on the branch for this TP
    # --------------------------------------------------------
    # closest_branch_idx_markers[tp] is a list of branch indices (one per marker)
    marker_branch_idx = res.closest_branch_idx_markers[tp]
    marker_coords = [branch[idx] for idx in marker_branch_idx]

    # decide which markers to actually plot
    kept_indices = []
    kept_coords = []
    for m_idx, coord in enumerate(marker_coords):
        cid = mapping_tp.get(m_idx, None)
        if cid is None and not show_unclustered:
            continue
        kept_indices.append(m_idx)
        kept_coords.append(coord)

    # --------------------------------------------------------
    # 5) Plot
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))

    # branch
    plt.plot(bx, by, lw=1.5, label="branch")

    # fiducials (optional points, same as original plot)
    if fx:
        plt.scatter(fx, fy, marker="o", s=36, label="fiducials (placed)")
    if mfx:
        plt.scatter(mfx, mfy, marker="X", s=64, label="fiducials (mapped)")

    # markers
    if kept_coords:
        mx = [p[0] for p in kept_coords]
        my = [p[1] for p in kept_coords]
        plt.scatter(mx, my, marker=".", s=30, label="markers")

        # text labels with cluster IDs
        dx, dy = text_dxy
        for m_local_idx, coord in zip(kept_indices, kept_coords):
            x, y = coord[0], coord[1]
            cid = mapping_tp.get(m_local_idx, None)
            if cid is None:
                # unclustered marker: either skip label or label differently
                continue
            plt.text(
                x + dx,
                y + dy,
                str(cid),
                fontsize=text_fontsize,
                ha="center",
                va="bottom",
            )

    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.title(f"Timepoint {tp+1} — branch with marker cluster IDs")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_branches_with_cluster_ids(
    res,
    cluster_list,
    text_fontsize: int = 8,
    text_dxy: tuple[float, float] = (0.0, 0.0),
    show_unclustered: bool = True,
):
    """
    Plot all timepoints side-by-side, each showing:
      - branch
      - markers projected onto branch
      - cluster ID labels per marker

    This is a multi-panel version of plot_branch_with_cluster_ids().
    """

    import matplotlib.pyplot as plt

    n_tp = len(res.normalized_branch)

    # --------------------------------------------------------
    # Build mapping (tp, marker_idx) -> cluster_id (1-based)
    # --------------------------------------------------------
    marker_to_cluster = [dict() for _ in range(n_tp)]
    if cluster_list:
        for cid, cluster in enumerate(cluster_list, start=1):
            for (tp_c, idx, _pos) in cluster:
                if idx == "NA":
                    continue
                marker_to_cluster[tp_c][idx] = cid

    fig, axes = plt.subplots(
        1,
        n_tp,
        figsize=(5 * n_tp, 5),
        squeeze=False
    )

    for tp in range(n_tp):
        ax = axes[0, tp]

        branch = res.normalized_branch[tp]
        bx = [p[0] for p in branch]
        by = [p[1] for p in branch]

        # draw branch
        ax.plot(bx, by, lw=1.5, label="branch")

        # markers mapped to branch
        marker_branch_idx = res.closest_branch_idx_markers[tp]
        marker_coords = [branch[idx] for idx in marker_branch_idx]

        mapping_tp = marker_to_cluster[tp]

        kept_indices = []
        kept_coords = []

        for m_idx, coord in enumerate(marker_coords):
            cid = mapping_tp.get(m_idx, None)
            if cid is None and not show_unclustered:
                continue
            kept_indices.append(m_idx)
            kept_coords.append(coord)

        # plot markers
        if kept_coords:
            mx = [p[0] for p in kept_coords]
            my = [p[1] for p in kept_coords]
            ax.scatter(mx, my, marker=".", s=30, label="markers")

            dx, dy = text_dxy
            for m_local_idx, coord in zip(kept_indices, kept_coords):
                x, y = coord[0], coord[1]
                cid = mapping_tp.get(m_local_idx, None)
                if cid is None:
                    continue
                ax.text(
                    x + dx,
                    y + dy,
                    str(cid),
                    fontsize=text_fontsize,
                    ha="center",
                    va="bottom",
                )

        ax.set_title(f"Timepoint {tp+1}")
        ax.set_xlabel("X (µm)")
        if tp == 0:
            ax.set_ylabel("Y (µm)")
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle("Branch + marker cluster IDs across all timepoints", fontsize=14)
    plt.tight_layout()
    plt.show()

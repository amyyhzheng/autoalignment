from __future__ import annotations

import csv
import numpy as np

from config import Settings
from computation import ComputationResult


# --- helpers ---

def separate_shaft_spine(
    settings: "Settings",
    result: "ComputationResult",
    eps: float = 1e-6,
    delta: float = 1e-3,
):
    """
    Return per-timepoint lists of shaft markers and spine markers.

    Each marker is a dict with:
        - type
        - distance         : possibly shifted distance used for clustering
        - orig_distance    : original unshifted distance
        - tp               : timepoint index
        - orig_point_idx   : original index within result.final_marker_distance[tp]
        - point_idx        : stable ID used downstream

    Duplicate distances within a timepoint are made unique by adding `delta`
    repeatedly until they no longer collide within `eps`.
    """

    def dedupe_markers(marker_list, used_distances=None, next_idx=0, eps=1e-6, delta=1e-3):
        if used_distances is None:
            used_distances = []

        out = []
        for m in marker_list:
            m2 = dict(m)
            d = m2["distance"]

            while any(abs(d - u) <= eps for u in used_distances):
                print(f"[separate_shaft_spine] duplicate distance {d} -> shifting by {delta}")
                d += delta

            m2["distance"] = d
            m2["point_idx"] = next_idx
            next_idx += 1

            used_distances.append(d)
            out.append(m2)

        return out, used_distances, next_idx

    shaft, spine = [], []

    for tp_idx, (types, dists) in enumerate(zip(result.raw_marker_types_only, result.final_marker_distance)):
        shaft_tp = []
        spine_tp = []

        for orig_point_idx, (t, d) in enumerate(zip(types, dists)):
            marker = {
                "type": t,
                "distance": d,
                "orig_distance": d,
                "tp": tp_idx,
                "orig_point_idx": orig_point_idx,
                "point_idx": None,
            }

            if str(t).lower() == settings.inhibitory_shaft.lower():
                shaft_tp.append(marker)
            elif str(t).lower() == settings.inhibitory_spine.lower():
                spine_tp.append(marker)

        used = []
        next_idx = 0

        shaft_tp, used, next_idx = dedupe_markers(
            shaft_tp, used_distances=used, next_idx=next_idx, eps=eps, delta=delta
        )
        spine_tp, used, next_idx = dedupe_markers(
            spine_tp, used_distances=used, next_idx=next_idx, eps=eps, delta=delta
        )

        print(
            f"Timepoint {tp_idx}: "
            f"shaft distances after dedupe: {[m['distance'] for m in shaft_tp]}, "
            f"spine distances after dedupe: {[m['distance'] for m in spine_tp]}"
        )
        print(
            f"Timepoint {tp_idx}: "
            f"shaft point_idx: {[m['point_idx'] for m in shaft_tp]}, "
            f"spine point_idx: {[m['point_idx'] for m in spine_tp]}"
        )

        shaft.append(shaft_tp)
        spine.append(spine_tp)

    return shaft, spine


def _closest_centroid_idx(x, centroids):
    return int(np.argmin([abs(c - x) for c in centroids]))


def _get_pos_and_idx(marker, tp, local_i):
    """
    Supports both:
      - new marker dicts with stable point_idx
      - old float distances
    """
    if isinstance(marker, dict):
        return marker["distance"], marker["point_idx"]
    else:
        return marker, local_i


def _kmeans_once(centroids, distance_data, final_marker_distance=None):
    """
    distance_data can be either:
      - List[List[marker_dict]]
      - List[List[float]]
    """

    def assign(current):
        cmap = {}
        for tp, tp_list in enumerate(distance_data):
            for local_i, marker in enumerate(tp_list):
                pos, point_idx = _get_pos_and_idx(marker, tp, local_i)

                ci = _closest_centroid_idx(pos, current)
                if ci not in cmap:
                    cmap[ci] = []

                # store: (centroid_idx, shifted_position, tp, point_idx)
                cmap[ci].append((ci, pos, tp, point_idx))
        return cmap

    def reassign(cmap):
        keys = sorted(cmap.keys())
        newc = []
        for cid in keys:
            vals = [p for _, p, _, _ in cmap[cid]]
            newc.append(round(float(np.mean(vals)), 2))
        return newc

    current = centroids[:]
    for _ in range(10):
        cmap = assign(current)
        newc = reassign(cmap)
        if newc == current:
            break
        current = newc

    return current, cmap


def _split_if_needed(centroids, cmap, max_spread=3.0):
    """
    Enforce:
      - at most one point per timepoint per cluster
      - cluster spread <= max_spread

    If violated, split out the worst offender as a new centroid.
    """
    for cid, tuples in list(cmap.items()):
        seen = set()
        dup_tp = [t for _, _, t, _ in tuples if (t in seen or seen.add(t))]

        if dup_tp:
            worst = max(
                [t for t in tuples if t[2] in dup_tp],
                key=lambda T: abs(T[1] - centroids[cid])
            )
            pos = worst[1]

            vals = [p for _, p, _, _ in tuples]
            if len(vals) > 1:
                new_c = round((centroids[cid] * len(vals) - pos) / (len(vals) - 1), 2)
                return [new_c if i == cid else c for i, c in enumerate(centroids)] + [pos]

        vals = [p for _, p, _, _ in tuples]
        if vals and (max(vals) - min(vals) > max_spread):
            avg = float(np.mean(vals))
            furthest = max(vals, key=lambda v: abs(v - avg))
            if len(vals) > 1:
                new_c = round((avg * len(vals) - furthest) / (len(vals) - 1), 2)
                return [new_c if i == cid else c for i, c in enumerate(centroids)] + [furthest]

    return centroids


def _run_with_seed(seed_centroids, distance_data, final_marker_distance):
    centroids = seed_centroids[:]

    for _ in range(20):
        centroids, cmap = _kmeans_once(centroids, distance_data, final_marker_distance)
        updated = _split_if_needed(centroids, cmap)
        if updated == centroids:
            break
        centroids = updated

    stdevs = []
    for cid in sorted(cmap.keys()):
        vals = [p for _, p, _, _ in cmap[cid]]
        if vals:
            stdevs.append(float(np.std(vals)))

    avg_stdev = float(np.mean(stdevs)) if stdevs else 0.0
    return centroids, cmap, avg_stdev


def choose_best_clustering(distance_data, final_marker_distance):
    """
    Works with either:
      - List[List[marker_dict]]
      - List[List[float]]
    """
    stdevs, seeds, results = [], [], []

    for i, tp_list in enumerate(distance_data):
        if not tp_list:
            continue

        if isinstance(tp_list[0], dict):
            seed_centroids = [m["distance"] for m in tp_list]
        else:
            seed_centroids = tp_list[:]

        res = _run_with_seed(seed_centroids, distance_data, final_marker_distance)
        results.append(res)
        seeds.append(i)
        stdevs.append(res[2])

    if not results:
        return None

    avg = float(np.mean(stdevs)) if stdevs else 0.0
    best_i = int(np.argmin([abs(s - avg) for s in stdevs]))
    return results[best_i]


def export_grouping_csv(grouping, out_path, start_id=0, group_type=None, metadata_out=None):
    """
    Export grouped positions to CSV.

    Output CSV:
        columns = Group{N}
        rows    = timepoints
        values  = shifted clustering positions or 'NA'

    Optional metadata CSV includes:
        group_name, cluster_id, type, tp, point_idx, pos
    """
    centroids, cmap, _ = grouping

    raw_order = np.argsort(centroids)
    order = [cid for cid in raw_order if cid in cmap]

    columns = [f"Group{start_id + i + 1}" for i in range(len(order))]

    tp_max = 1 + max([t[2] for tuples in cmap.values() for t in tuples] or [0])
    grid = [["NA" for _ in order] for _ in range(tp_max)]
    cluster_list = []
    metadata_rows = []

    for col, cid in enumerate(order):
        tuples = cmap[cid]
        col_entries = []
        have_tp = set()
        group_name = f"Group{start_id + col + 1}"
        cluster_id = start_id + col + 1

        for _, pos, tp, point_idx in tuples:
            grid[tp][col] = pos
            col_entries.append((tp, point_idx, pos))
            have_tp.add(tp)

            metadata_rows.append({
                "group_name": group_name,
                "cluster_id": cluster_id,
                "type": group_type,
                "tp": tp,
                "point_idx": point_idx,
                "pos": pos,
            })

        for tp in range(tp_max):
            if tp not in have_tp:
                col_entries.append((tp, "NA", "NA"))
                metadata_rows.append({
                    "group_name": group_name,
                    "cluster_id": cluster_id,
                    "type": group_type,
                    "tp": tp,
                    "point_idx": "NA",
                    "pos": "NA",
                })

        col_entries.sort()
        cluster_list.append(col_entries)

    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(columns)
        for row in grid:
            wr.writerow(row)

    if metadata_out is not None:
        with open(metadata_out, "w", newline="") as f:
            wr = csv.DictWriter(
                f,
                fieldnames=["group_name", "cluster_id", "type", "tp", "point_idx", "pos"]
            )
            wr.writeheader()
            wr.writerows(metadata_rows)

    return cluster_list

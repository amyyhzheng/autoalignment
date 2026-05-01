from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv
import numpy as np

from config import Settings
from computation import ComputationResult

# --- helpers ---

# def separate_shaft_spine(settings: Settings, result: ComputationResult):
#     shaft, spine = [], []
#     for types, dists in zip(result.raw_marker_types_only, result.final_marker_distance):
#         zp = list(zip(types, dists))
#         shaft_tp = [d for t, d in zp if str(t).lower() == settings.inhibitory_shaft.lower()]
#         spine_tp = [d for t, d in zp if str(t).lower() == settings.inhibitory_spine.lower()]
#         print(f"Timepoint: shaft distances: {shaft_tp}, spine distances: {spine_tp}")
#         print(f"Timepoint: shaft types: {[t for t, _ in zp if str(t).lower() == settings.inhibitory_shaft.lower()]}, spine types: {[t for t, _ in zp if str(t).lower() == settings.inhibitory_spine.lower()]}")

#         if len(shaft_tp) != len(set(shaft_tp)):
#             raise ValueError("Duplicate distances for shafts within same timepoint")
#         if len(spine_tp) != len(set(spine_tp)):
#             raise ValueError("Duplicate distances for spines within same timepoint")
#         shaft.append(shaft_tp)
#         spine.append(spine_tp)
#     return shaft, spine

# def separate_shaft_spine(settings: Settings, result: ComputationResult, eps: float = 1e-6):
#     """
#     TEMPORARY SOLUTION
#     Return per-timepoint lists of shaft distances and spine distances.
#     If duplicate distances occur within a TP (common due to rounding/snapping),
#     de-duplicate them (keep first occurrence) instead of raising.
#     """
#     def dedupe_distances(dist_list, eps=1e-6):
#         # preserve original order, treat distances within eps as duplicates
#         kept = []
#         for d in dist_list:
#             if not any(abs(d - k) <= eps for k in kept):
#                 kept.append(d)
#         return kept

#     shaft, spine = [], []
#     for tp_idx, (types, dists) in enumerate(zip(result.raw_marker_types_only, result.final_marker_distance)):
#         zp = list(zip(types, dists))

#         shaft_tp = [d for t, d in zp if str(t).lower() == settings.inhibitory_shaft.lower()]
#         spine_tp = [d for t, d in zp if str(t).lower() == settings.inhibitory_spine.lower()]

#         print(f"Timepoint {tp_idx}: shaft distances: {shaft_tp}, spine distances: {spine_tp}")
#         print(
#             f"Timepoint {tp_idx}: shaft types: {[t for t, _ in zp if str(t).lower() == settings.inhibitory_shaft.lower()]}, "
#             f"spine types: {[t for t, _ in zp if str(t).lower() == settings.inhibitory_spine.lower()]}"
#         )

#         # de-dupe instead of raise
#         shaft_tp_dedup = dedupe_distances(shaft_tp, eps=eps)
#         spine_tp_dedup = dedupe_distances(spine_tp, eps=eps)

#         if len(shaft_tp_dedup) != len(shaft_tp):
#             print(f"[separate_shaft_spine] WARNING tp={tp_idx}: deduped shaft distances {len(shaft_tp)} -> {len(shaft_tp_dedup)} (eps={eps})")
#         if len(spine_tp_dedup) != len(spine_tp):
#             print(f"[separate_shaft_spine] WARNING tp={tp_idx}: deduped spine distances {len(spine_tp)} -> {len(spine_tp_dedup)} (eps={eps})")

#         shaft.append(shaft_tp_dedup)
#         spine.append(spine_tp_dedup)

#     return shaft, spine

def separate_shaft_spine(settings: Settings, result: ComputationResult, eps: float = 1e-6):
    """
    Return per-timepoint lists of shaft markers and spine markers.
    Each marker keeps its original type and distance.
    If duplicate distances occur within a TP, de-duplicate by distance
    while preserving the first matching marker entry.
    """

    def dedupe_markers(marker_list, used=None, eps=1e-6, delta=1e-3):
        if used is None:
            used = []

        for m in marker_list:
            d = m["distance"]

            # Keep shifting until it's unique
            while any(abs(d - u) <= eps for u in used):
                print('had to dedupe marker with distance', d, 'by shifting it by delta', delta)
                d += delta

            m["distance"] = d
            used.append(d)

        return marker_list, used
    

    shaft, spine = [], []

    for tp_idx, (types, dists) in enumerate(zip(result.raw_marker_types_only, result.final_marker_distance)):
        zp = list(zip(types, dists))
        #changed to include oriignal point index for better debugging and potential future use, but still de-dupe by distance

        shaft_tp = [
            {"type": t, "distance": d, "tp": tp_idx, "point_idx": point_idx}
            for point_idx, (t, d) in enumerate(zp)
            if str(t).lower() == settings.inhibitory_shaft.lower()
        ]
        spine_tp = [
            {"type": t, "distance": d, "tp": tp_idx, "point_idx": point_idx}
            for point_idx, (t, d) in enumerate(zp)
            if str(t).lower() == settings.inhibitory_spine.lower()
        ]

        print(f"Timepoint {tp_idx}: shaft distances: {[m['distance'] for m in shaft_tp]}, spine distances: {[m['distance'] for m in spine_tp]}")
        print(
            f"Timepoint {tp_idx}: shaft types: {[m['type'] for m in shaft_tp]}, "
            f"spine types: {[m['type'] for m in spine_tp]}"
        )

        used = []
        delta = 1e-3
        shaft_tp_dedup, used = dedupe_markers(shaft_tp, used=used, eps=eps, delta=delta)
        spine_tp_dedup, used = dedupe_markers(spine_tp, used=used, eps=eps, delta=delta)

        if len(shaft_tp_dedup) != len(shaft_tp):
            print(f"[separate_shaft_spine] WARNING tp={tp_idx}: deduped shaft distances {len(shaft_tp)} -> {len(shaft_tp_dedup)} (eps={eps})")
        if len(spine_tp_dedup) != len(spine_tp):
            print(f"[separate_shaft_spine] WARNING tp={tp_idx}: deduped spine distances {len(spine_tp)} -> {len(spine_tp_dedup)} (eps={eps})")

        shaft.append(shaft_tp_dedup)
        spine.append(spine_tp_dedup)

    return shaft, spine


# def _closest_centroid_idx(x, centroids):
#     return int(np.argmin([abs(c - x) for c in centroids]))
def _closest_valid_centroid_idx(pos, tp, current, cmap):
    ranked = sorted(
        range(len(current)),
        key=lambda ci: abs(current[ci] - pos)
    )

    for ci in ranked:
        tuples = cmap.get(ci, [])
        used_tps = {t for _, _, t, _ in tuples}
        if tp not in used_tps:
            return ci

    return None
def _closest_centroid_idx(x, centroids):
    return int(np.argmin([abs(c - x) for c in centroids]))

    return None

def _kmeans_once(centroids, distance_data, final_marker_distance):
    # def assign(current):
    #     cmap = {}
    #     for tp, tp_list in enumerate(distance_data):
    #         for pos in tp_list:
    #             ci = _closest_centroid_idx(pos, current)
    #             if ci not in cmap: cmap[ci] = []
    #             # store: (centroid_idx, position, tp, point_idx)
    #             point_idx = final_marker_distance[tp].index(pos)
    #             cmap[ci].append((ci, pos, tp, point_idx))
    #     return cmap
    # def assign(current):
    #     cmap = {}
    #     for tp, tp_list in enumerate(distance_data):
    #         for point_idx, pos in enumerate(tp_list):
    #             ci = _closest_valid_centroid_idx(pos, tp, current, cmap)

    #             if ci is None:
    #                 # no existing cluster can take this tp
    #                 ci = len(current)
    #                 current.append(pos)
    #                 cmap[ci] = []

    #             if ci not in cmap:
    #                 cmap[ci] = []

    #         for pos in tp_list:
    #             ci = _closest_centroid_idx(pos, current)
    #             if ci not in cmap: cmap[ci] = []
    #             # store: (centroid_idx, position, tp, point_idx)
    #             point_idx = final_marker_distance[tp].index(pos)
    #             cmap[ci].append((ci, pos, tp, point_idx))

    #     return cmap, current
    def assign(current):
        cmap = {}

        for tp, tp_list in enumerate(distance_data):
            for marker in tp_list:
                pos = marker["distance"]
                point_idx = marker["point_idx"]

                ci = _closest_valid_centroid_idx(pos, tp, current, cmap)

                if ci is None:
                    ci = len(current)
                    current.append(pos)
                    cmap[ci] = []

                if ci not in cmap:
                    cmap[ci] = []

                cmap[ci].append((ci, pos, tp, point_idx))

        return cmap, current

    def reassign(cmap):

        newc = []
        for i in range(len(cmap)):
            vals = [p for _, p, _, _ in cmap[i]]
            newc.append(round(float(np.mean(vals)), 2))
        return newc

    current = centroids[:]
    for _ in range(10):
        cmap, current = assign(current)

        newc = reassign(cmap)
        if newc == current: break
        current = newc
    return current, cmap


def _split_if_needed(centroids, cmap, max_spread=3.0):
    # Enforce: at most one point per timepoint per cluster; and cluster spread ≤ max_spread µm
    # if violated, split out the worst offender as new centroid
    for cid, tuples in list(cmap.items()):
        # one per timepoint
        seen = set()
        dup_tp = [t for _, _, t, _ in tuples if (t in seen or seen.add(t))]
        if dup_tp:
            # farthest from centroid
            worst = max([t for t in tuples if t[2] in dup_tp], key=lambda T: abs(T[1]-centroids[cid]))
            pos = worst[1]
            # recompute centroid if removing worst
            vals = [p for _, p, _, _ in tuples]
            new_c = round((centroids[cid]*len(vals) - pos)/(len(vals)-1), 2)
            return [new_c if i==cid else c for i,c in enumerate(centroids)] + [pos]
        # spread
        vals = [p for _, p, _, _ in tuples]
        if vals and (max(vals) - min(vals) > max_spread):
            avg = float(np.mean(vals))
            furthest = max(vals, key=lambda v: abs(v-avg))
            new_c = round((avg*len(vals) - furthest)/(len(vals)-1), 2)
            return [new_c if i==cid else c for i,c in enumerate(centroids)] + [furthest]
    return centroids


def _run_with_seed(seed_centroids, distance_data, final_marker_distance):

    
    centroids = seed_centroids[:]
    for _ in range(20):
        centroids, cmap = _kmeans_once(centroids, distance_data, final_marker_distance)
        updated = _split_if_needed(centroids, cmap)
        if updated == centroids:
            break
        centroids = updated
    # compute average stdev across clusters
    stdevs = []
    for cid in range(len(cmap)):
        vals = [p for _, p, _, _ in cmap[cid]]
        if vals:
            stdevs.append(float(np.std(vals)))
    avg_stdev = float(np.mean(stdevs)) if stdevs else 0.0
    return centroids, cmap, avg_stdev


def choose_best_clustering(distance_data, final_marker_distance):
    '''
    input is the output from the separate_shaft_spine function: a list of lists of distances, grouped by timepoint.
    '''
    stdevs, seeds, results = [], [], []
    for i, tp_list in enumerate(distance_data):
        if not tp_list: continue
        # res = _run_with_seed(tp_list[:], distance_data, final_marker_distance)


        seed_centroids = [m["distance"] for m in tp_list]
        res = _run_with_seed(seed_centroids, distance_data, final_marker_distance)
        results.append(res)
        seeds.append(i)
        stdevs.append(res[2])
    if not results:
        return None
    avg = float(np.mean(stdevs)) if stdevs else 0.0
    best_i = int(np.argmin([abs(s-avg) for s in stdevs]))
    return results[best_i]


# def export_grouping_csv(grouping, out_path):
#     centroids, cmap, _ = grouping

#     print("\n[export_grouping_csv] --- DIAGNOSTICS ---")
#     print("[export_grouping_csv] centroids len:", len(centroids))
#     print("[export_grouping_csv] cmap keys count:", len(cmap))

#     # keys summary
#     cmap_keys = sorted(cmap.keys())
#     print("[export_grouping_csv] cmap key min/max:", (cmap_keys[0], cmap_keys[-1]) if cmap_keys else None)
#     print("[export_grouping_csv] first 30 cmap keys:", cmap_keys[:30])

#     order = np.argsort(centroids)
#     order_list = order.tolist() if hasattr(order, "tolist") else list(order)
#     print("[export_grouping_csv] order len:", len(order_list))
#     print("[export_grouping_csv] order min/max:", (min(order_list), max(order_list)) if order_list else None)
#     print("[export_grouping_csv] first 30 order:", order_list[:30])

#     missing = [cid for cid in order_list if cid not in cmap]
#     print("[export_grouping_csv] missing cids (order not in cmap):", missing[:30], "count:", len(missing))

#     # optional: stop immediately to avoid a long crash loop
#     if missing:
#         raise KeyError(f"Missing {len(missing)} cluster ids in cmap. Example: {missing[:10]}")

#     # --- rest of your function ---
#     centroids, cmap, _ = grouping
#     # order clusters by centroid position
#     order = np.argsort(centroids)
#     columns = [f"Group{i+1}" for i in range(len(order))]
#     # make a grid rows=timepoints, cols=groups with positions or NA
#     # also return a list-of-clusters where each cluster is [(tp, idx, pos) or (tp,'NA','NA')]
#     tp_max = 1 + max([t[2] for tuples in cmap.values() for t in tuples] or [0])
#     grid = [["NA" for _ in order] for _ in range(tp_max)]
#     cluster_list = []
#     for col, cid in enumerate(order):
#         tuples = cmap[cid]
#         col_entries = []
#         have_tp = set()
#         for _, pos, tp, point_idx in tuples:
#             grid[tp][col] = pos
#             col_entries.append((tp, point_idx, pos))
#             have_tp.add(tp)
#         for tp in range(tp_max):
#             if tp not in have_tp:
#                 col_entries.append((tp, "NA", "NA"))
#         col_entries.sort()
#         cluster_list.append(col_entries)
#     with open(out_path, "w", newline="") as f:
#         wr = csv.writer(f)
#         wr.writerow(columns)
#         for row in grid:
#             wr.writerow(row)
#     return cluster_list

def export_grouping_csv(grouping, out_path, start_id=0, group_type=None, metadata_out=None):
    # temporary solution
    centroids, cmap, _ = grouping

    # order clusters by centroid position, BUT only keep ids that exist in cmap
    raw_order = np.argsort(centroids)
    order = [cid for cid in raw_order if cid in cmap]

    # shift group numbering by start_id
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

            # keep OLD format for downstream plotting/mapping compatibility
            col_entries.append((tp, point_idx, pos))
            have_tp.add(tp)

            # optional metadata export
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

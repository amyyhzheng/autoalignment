from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv
import numpy as np

from config import Settings
from computation import ComputationResult

# --- helpers ---

def separate_shaft_spine(settings: Settings, result: ComputationResult):
    shaft, spine = [], []
    for types, dists in zip(result.raw_marker_types_only, result.final_marker_distance):
        zp = list(zip(types, dists))
        shaft_tp = [d for t, d in zp if str(t).lower() == settings.inhibitory_shaft.lower()]
        spine_tp = [d for t, d in zp if str(t).lower() == settings.inhibitory_spine.lower()]
        if len(shaft_tp) != len(set(shaft_tp)):
            raise ValueError("Duplicate distances for shafts within same timepoint")
        if len(spine_tp) != len(set(spine_tp)):
            raise ValueError("Duplicate distances for spines within same timepoint")
        shaft.append(shaft_tp)
        spine.append(spine_tp)
    return shaft, spine


def _closest_centroid_idx(x, centroids):
    return int(np.argmin([abs(c - x) for c in centroids]))


def _kmeans_once(centroids, distance_data, final_marker_distance):
    def assign(current):
        cmap = {}
        for tp, tp_list in enumerate(distance_data):
            for pos in tp_list:
                ci = _closest_centroid_idx(pos, current)
                if ci not in cmap: cmap[ci] = []
                # store: (centroid_idx, position, tp, point_idx)
                point_idx = final_marker_distance[tp].index(pos)
                cmap[ci].append((ci, pos, tp, point_idx))
        return cmap

    def reassign(cmap):
        newc = []
        for i in range(len(cmap)):
            vals = [p for _, p, _, _ in cmap[i]]
            newc.append(round(float(np.mean(vals)), 2))
        return newc

    current = centroids[:]
    for _ in range(10):
        cmap = assign(current)
        newc = reassign(cmap)
        if newc == current: break
        current = newc
    return current, cmap


def _split_if_needed(centroids, cmap, max_spread=2.0):
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
    stdevs, seeds, results = [], [], []
    for i, tp_list in enumerate(distance_data):
        if not tp_list: continue
        res = _run_with_seed(tp_list[:], distance_data, final_marker_distance)
        results.append(res)
        seeds.append(i)
        stdevs.append(res[2])
    if not results:
        return None
    avg = float(np.mean(stdevs)) if stdevs else 0.0
    best_i = int(np.argmin([abs(s-avg) for s in stdevs]))
    return results[best_i]


def export_grouping_csv(grouping, out_path):
    centroids, cmap, _ = grouping
    # order clusters by centroid position
    order = np.argsort(centroids)
    columns = [f"Group{i+1}" for i in range(len(order))]
    # make a grid rows=timepoints, cols=groups with positions or NA
    # also return a list-of-clusters where each cluster is [(tp, idx, pos) or (tp,'NA','NA')]
    tp_max = 1 + max([t[2] for tuples in cmap.values() for t in tuples] or [0])
    grid = [["NA" for _ in order] for _ in range(tp_max)]
    cluster_list = []
    for col, cid in enumerate(order):
        tuples = cmap[cid]
        col_entries = []
        have_tp = set()
        for _, pos, tp, point_idx in tuples:
            grid[tp][col] = pos
            col_entries.append((tp, point_idx, pos))
            have_tp.add(tp)
        for tp in range(tp_max):
            if tp not in have_tp:
                col_entries.append((tp, "NA", "NA"))
        col_entries.sort()
        cluster_list.append(col_entries)
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(columns)
        for row in grid:
            wr.writerow(row)
    return cluster_list

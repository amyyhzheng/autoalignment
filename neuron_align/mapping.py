from __future__ import annotations
from typing import List, Tuple
import csv
import os

from config import Settings
from computation import ComputationResult
from io_utils import z_imagej_to_objectj
from geometry import distance_along_branch

Coord = Tuple[float, float, float]


def _average_cluster_distance(cluster):
    vals = [p[2] for p in cluster if p[2] != 'NA']
    return sum(vals)/len(vals) if vals else 0.0


def _estimate_coordinate(tp: int, avg_dist: float, result: ComputationResult) -> Coord:
    # find segment where the average falls for timepoint tp, then invert the scaling
    cum = result.cumdist_scaled[tp]
    for seg_id in range(len(cum)-1):
        if avg_dist <= cum[seg_id+1]:
            # unscale within this segment
            unscaled = (avg_dist - result.cumdist_scaled[tp][seg_id]) / result.scale_factors_all[tp][seg_id] + result.cumdist_unscaled[tp][seg_id]
            # find nearest branch index by distance along branch
            for i in range(len(result.normalized_branch[tp])):
                d = distance_along_branch(result.normalized_branch[tp], 0, i)
                if d >= unscaled - 1e-8:
                    return result.raw_branch[tp][i]
    raise ValueError("average distance out of bounds")


def _avg_translation(cluster, result: ComputationResult) -> Tuple[float, float]:
    # average vector from mapped branch point to raw marker point
    xs, ys, n = 0.0, 0.0, 0
    for tp, idx, _pos in cluster:
        if idx == 'NA':
            continue
        raw_xy = result.raw_marker_coords_only[tp][idx][:2]
        mapped_idx = result.closest_branch_idx_markers[tp][idx]
        mapped_xy = result.raw_branch[tp][mapped_idx][:2]
        xs += (raw_xy[0] - mapped_xy[0])
        ys += (raw_xy[1] - mapped_xy[1])
        n += 1
    return (round(xs/n, 0), round(ys/n, 0)) if n else (0.0, 0.0)


def clusters_to_csv_rows(cluster_list, marker_type: str, empty_type: str, start_id: int, translate: bool,
                         settings: Settings, result: ComputationResult):
    out = []
    next_id = start_id
    for cluster in cluster_list:
        avg_dist = _average_cluster_distance(cluster)
        dx, dy = _avg_translation(cluster, result) if translate else (0.0, 0.0)
        for tp, idx, _pos in cluster:
            if idx == 'NA':
                mtype = empty_type
                x, y, z = _estimate_coordinate(tp, avg_dist, result)
                x += dx; y += dy
            else:
                mtype = marker_type
                x, y, z = result.raw_marker_coords_only[tp][idx]
            out.append([f"Image{tp+1}", next_id, mtype, x, y, z_imagej_to_objectj(z, settings.num_channels)])
        next_id += 1
    return out, next_id


def export_all(settings: Settings, result: ComputationResult,
               shaft_clusters, spine_clusters,
               export_dir):
    auto_dir = os.path.join(export_dir, 'autoAlignment')
    os.makedirs(auto_dir, exist_ok=True)

    rows = []
    next_id = 1
    if shaft_clusters:
        part, next_id = clusters_to_csv_rows(shaft_clusters, "InhibitoryShaft", "Nothing", next_id, False, settings, result)
        rows.extend(part)
    if spine_clusters:
        part, next_id = clusters_to_csv_rows(spine_clusters, "SpinewithInhSynapse", "NudeSpine", next_id, True, settings, result)
        rows.extend(part)

    # Landmarks
    for tp, coords in enumerate(result.raw_fiducials):
        for k, (x, y, z_img) in enumerate(coords, start=1):
            rows.append([f"Image{tp+1}", f"Marker{next_id}", "Landmark", x, y, z_imagej_to_objectj(z_img, settings.num_channels)])
            next_id += 1

    out_csv = os.path.join(auto_dir, f"{settings.animal_id}_b{settings.branch_id}_alignmentMapping.csv")
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(["image","markerID","markerType","marker_X","marker_Y","marker_Z"])
        wr.writerows(rows)

    # Per-timepoint napari-friendly CSVs
    by_tp = {}
    for row in rows:
        tp = row[0].replace("Image", "")
        nap = [row[5], row[4], row[3], row[1], row[2]]  # z,y,x,label,type
        by_tp.setdefault(tp, []).append(nap)
    header = ["z","y","x","label","type"]
    for tp, nap_rows in by_tp.items():
        path = os.path.join(auto_dir, f"{settings.animal_id}_b{settings.branch_id}_timepoint{tp}_napari.csv")
        with open(path, 'w', newline='') as f:
            wr = csv.writer(f); wr.writerow(header); wr.writerows(nap_rows)

    return out_csv

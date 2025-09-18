
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import os
import numpy as np
import pandas as pd

from config import Settings
from io_utils import read_branch_csv, read_markers_csv_list, read_fiducials_csv, z_imagej_to_objectj
from geometry import fit_branch_spline, get_xyzs, euc_xy, distance_along_branch

Coord = Tuple[float, float, float]

@dataclass
class ComputationResult:
    normalized_branch: List[List[Coord]]
    normalized_markers: List[List[Coord]]
    normalized_fiducials: List[List[Coord]]
    raw_branch: List[List[Coord]]
    raw_markers: List[List[Tuple[str, Coord]]]
    raw_fiducials: List[List[Coord]]
    closest_branch_idx_markers: List[List[int]]
    closest_branch_idx_fids: List[List[int]]
    segments_all: List[List[Tuple[int, int]]]
    seg_lengths_all: List[List[float]]
    cumdist_unscaled: List[List[float]]
    cumdist_scaled: List[List[float]]
    scale_factors_all: List[List[float]]
    final_marker_distance: List[List[float]]
    raw_marker_coords_only: List[List[Coord]]
    raw_marker_types_only: List[List[str]]


def _normalize_and_scale(branch, markers, fiducials, scale):
    minx = min(pt[0] for pt in branch)
    miny = min(pt[1] for pt in branch)
    minz = min(pt[2] for pt in branch)
    def _xf(c):
        return ((c[0] - minx) * scale[0], (c[1] - miny) * scale[1], (c[2] - minz) * scale[2])
    return list(map(_xf, branch)), list(map(_xf, markers)), list(map(_xf, fiducials))


def _branch_points_from_csvs(settings: Settings) -> List[List[Coord]]:
    raw = []
    for i in range(settings.n_timepoints):
        df = read_branch_csv(settings.branch_csvs[f"Timepoint {i+1}"], settings.snt_branch_fmt % settings.branch_id)
        x, y, z = df["x"].values, df["y"].values, df["z"].values
        raw.append(fit_branch_spline(x, y, z, n_points=100))
    return raw


def compute(settings: Settings) -> ComputationResult:
    # Load raw data
    raw_branch = _branch_points_from_csvs(settings)
    raw_markers, raw_fids_typed = read_markers_csv_list(settings.marker_csvs, settings.num_channels)
    raw_fiducials = read_fiducials_csv(settings.fiducials_csv, settings.n_timepoints, settings.num_channels)

    # strip types for markers into parallel arrays
    raw_marker_coords_only = [[m[1] for m in tp] for tp in raw_markers]
    raw_marker_types_only = [[m[0] for m in tp] for tp in raw_markers]

    # Normalize & scale
    nb, nm, nf = [], [], []
    for i in range(settings.n_timepoints):
        b, m, f = _normalize_and_scale(raw_branch[i], raw_marker_coords_only[i], raw_fiducials[i], settings.scaling_factor)
        nb.append(b); nm.append(m); nf.append(f)

    # Mapping to nearest branch indices
    def nearest_indices(points, branch):
        idxs = []
        for p in points:
            dists = [euc_xy(p, q) for q in branch]
            idxs.append(int(np.argmin(dists)))
        return idxs

    cb_markers, cb_fids = [], []
    for i in range(settings.n_timepoints):
        cb_markers.append(nearest_indices(nm[i], nb[i]))
        fid_idx = nearest_indices(nf[i], nb[i])
        # avoid 0/last dup
        if fid_idx and fid_idx[0] == 0: fid_idx[0] = 1
        if len(fid_idx) >= 2 and fid_idx[-1] == len(nb[i]) - 1:
            fid_idx[-1] = max(0, fid_idx[-1] - 1)
        cb_fids.append(fid_idx)

    # Segments & lengths
    segments_all, seg_lengths_all = [], []
    for i in range(settings.n_timepoints):
        idxs = sorted([0, len(nb[i]) - 1] + cb_fids[i])
        if len(idxs) != len(set(idxs)):
            # caller will plot to help debug; keep going is unsafe
            raise RuntimeError(f"Duplicate fiducial-to-branch indices at timepoint {i+1}")
        segs, lens = [], []
        for a, b in zip(idxs[:-1], idxs[1:]):
            if a == b:
                lens.append(0.0); segs.append((a, b)); continue
            lens.append(distance_along_branch(nb[i], a, b))
            segs.append((a, b))
        segments_all.append(segs)
        seg_lengths_all.append(lens)

    # Max per segment across timepoints, per-timepoint scale factors
    max_per_seg = [max(vals) for vals in zip(*seg_lengths_all)] if seg_lengths_all else []
    scale_factors_all = []
    for i in range(settings.n_timepoints):
        sfs = []
        for seg_len, seg_max in zip(seg_lengths_all[i], max_per_seg):
            sfs.append(0.01 if seg_len == 0 else seg_max / seg_len)
        scale_factors_all.append(sfs)

    # Scaled segment lengths + cumulative distances (unscaled & scaled)
    seg_lengths_scaled = [
        [l * sf for l, sf in zip(seg_lengths_all[i], scale_factors_all[i])]
        for i in range(settings.n_timepoints)
    ]

    def cumdist(lengths):
        out = [0.0]
        c = 0.0
        for L in lengths:
            c += L
            out.append(c)
        return out

    cum_unscaled = [cumdist(seg_lengths_all[i]) for i in range(settings.n_timepoints)]
    cum_scaled   = [cumdist(seg_lengths_scaled[i]) for i in range(settings.n_timepoints)]

    # Final per-marker cumulative distances
    final_marker_distance: List[List[float]] = []
    for tp in range(settings.n_timepoints):
        fm = []
        for m_idx in cb_markers[tp]:
            # find containing segment
            segs = segments_all[tp]
            seg_id = next(j for j, (lo, hi) in enumerate(segs) if (m_idx >= lo and m_idx < hi) or (j == len(segs)-1 and m_idx == hi))
            lo, _ = segs[seg_id]
            d_in_seg = distance_along_branch(nb[tp], lo, m_idx)
            scaled_d = d_in_seg * scale_factors_all[tp][seg_id]
            fm.append(cum_scaled[tp][seg_id] + scaled_d)
        final_marker_distance.append(fm)

    return ComputationResult(
        normalized_branch=nb,
        normalized_markers=nm,
        normalized_fiducials=nf,
        raw_branch=raw_branch,
        raw_markers=raw_markers,
        raw_fiducials=raw_fiducials,
        closest_branch_idx_markers=cb_markers,
        closest_branch_idx_fids=cb_fids,
        segments_all=segments_all,
        seg_lengths_all=seg_lengths_all,
        cumdist_unscaled=cum_unscaled,
        cumdist_scaled=cum_scaled,
        scale_factors_all=scale_factors_all,
        final_marker_distance=final_marker_distance,
        raw_marker_coords_only=raw_marker_coords_only,
        raw_marker_types_only=raw_marker_types_only,
    )



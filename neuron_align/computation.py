
# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Tuple
# import csv
# import os
# import numpy as np
# import pandas as pd

# from config import Settings
# from io_utils import read_branch_csv, read_markers_csv_list, read_fiducials_csv, z_imagej_to_objectj
# from geometry import fit_branch_spline, get_xyzs, euc_xy, distance_along_branch

# Coord = Tuple[float, float, float]

# @dataclass
# class ComputationResult:
#     normalized_branch: List[List[Coord]]
#     normalized_markers: List[List[Coord]]
#     normalized_fiducials: List[List[Coord]]
#     raw_branch: List[List[Coord]]
#     raw_markers: List[List[Tuple[str, Coord]]]
#     raw_fiducials: List[List[Coord]]
#     closest_branch_idx_markers: List[List[int]]
#     closest_branch_idx_fids: List[List[int]]
#     segments_all: List[List[Tuple[int, int]]]
#     seg_lengths_all: List[List[float]]
#     cumdist_unscaled: List[List[float]]
#     cumdist_scaled: List[List[float]]
#     scale_factors_all: List[List[float]]
#     final_marker_distance: List[List[float]]
#     raw_marker_coords_only: List[List[Coord]]
#     raw_marker_types_only: List[List[str]]


# def _normalize_and_scale(branch, markers, fiducials, scale):
#     minx = min(pt[0] for pt in branch)
#     miny = min(pt[1] for pt in branch)
#     minz = min(pt[2] for pt in branch)
#     def _xf(c):
#         return ((c[0] - minx) * scale[0], (c[1] - miny) * scale[1], (c[2] - minz) * scale[2])
#     return list(map(_xf, branch)), list(map(_xf, markers)), list(map(_xf, fiducials))


# def _branch_points_from_csvs(settings: Settings) -> List[List[Coord]]:
#     raw = []
#     for i in range(settings.n_timepoints):
#         df = read_branch_csv(settings.branch_csvs[f"Timepoint {i+1}"], settings.snt_branch_fmt % settings.branch_id)
#         x, y, z = df["x"].values, df["y"].values, df["z"].values
#         raw.append(fit_branch_spline(x, y, z, n_points=100))
#     return raw


# def compute(settings: Settings) -> ComputationResult:
#     # Load raw data
#     raw_branch = _branch_points_from_csvs(settings)
#     raw_markers, raw_fids_typed = read_markers_csv_list(settings.marker_csvs, settings.num_channels)
#     raw_fiducials = read_fiducials_csv(settings.fiducials_csv, settings.n_timepoints, settings.num_channels)

#     # strip types for markers into parallel arrays
#     raw_marker_coords_only = [[m[1] for m in tp] for tp in raw_markers]
#     raw_marker_types_only = [[m[0] for m in tp] for tp in raw_markers]

#     # Normalize & scale
#     nb, nm, nf = [], [], []
#     for i in range(settings.n_timepoints):
#         b, m, f = _normalize_and_scale(raw_branch[i], raw_marker_coords_only[i], raw_fiducials[i], settings.scaling_factor)
#         nb.append(b); nm.append(m); nf.append(f)

#     # Mapping to nearest branch indices
#     def nearest_indices(points, branch):
#         idxs = []
#         for p in points:
#             dists = [euc_xy(p, q) for q in branch]
#             idxs.append(int(np.argmin(dists)))
#         return idxs

#     cb_markers, cb_fids = [], []
#     for i in range(settings.n_timepoints):
#         cb_markers.append(nearest_indices(nm[i], nb[i]))
#         fid_idx = nearest_indices(nf[i], nb[i])
#         # avoid 0/last dup
#         if fid_idx and fid_idx[0] == 0: fid_idx[0] = 1
#         if len(fid_idx) >= 2 and fid_idx[-1] == len(nb[i]) - 1:
#             fid_idx[-1] = max(0, fid_idx[-1] - 1)
#         cb_fids.append(fid_idx)

#     # Segments & lengths
#     segments_all, seg_lengths_all = [], []
#     for i in range(settings.n_timepoints):
#         idxs = sorted([0, len(nb[i]) - 1] + cb_fids[i])
#         if len(idxs) != len(set(idxs)):
#             # caller will plot to help debug; keep going is unsafe
#             raise RuntimeError(f"Duplicate fiducial-to-branch indices at timepoint {i+1}")
#         segs, lens = [], []
#         for a, b in zip(idxs[:-1], idxs[1:]):
#             if a == b:
#                 lens.append(0.0); segs.append((a, b)); continue
#             lens.append(distance_along_branch(nb[i], a, b))
#             segs.append((a, b))
#         segments_all.append(segs)
#         seg_lengths_all.append(lens)

#     # Max per segment across timepoints, per-timepoint scale factors
#     max_per_seg = [max(vals) for vals in zip(*seg_lengths_all)] if seg_lengths_all else []
#     scale_factors_all = []
#     for i in range(settings.n_timepoints):
#         sfs = []
#         for seg_len, seg_max in zip(seg_lengths_all[i], max_per_seg):
#             sfs.append(0.01 if seg_len == 0 else seg_max / seg_len)
#         scale_factors_all.append(sfs)

#     # Scaled segment lengths + cumulative distances (unscaled & scaled)
#     seg_lengths_scaled = [
#         [l * sf for l, sf in zip(seg_lengths_all[i], scale_factors_all[i])]
#         for i in range(settings.n_timepoints)
#     ]

#     def cumdist(lengths):
#         out = [0.0]
#         c = 0.0
#         for L in lengths:
#             c += L
#             out.append(c)
#         return out

#     cum_unscaled = [cumdist(seg_lengths_all[i]) for i in range(settings.n_timepoints)]
#     cum_scaled   = [cumdist(seg_lengths_scaled[i]) for i in range(settings.n_timepoints)]

#     # Final per-marker cumulative distances
#     final_marker_distance: List[List[float]] = []
#     for tp in range(settings.n_timepoints):
#         fm = []
#         for m_idx in cb_markers[tp]:
#             # find containing segment
#             segs = segments_all[tp]
#             seg_id = next(j for j, (lo, hi) in enumerate(segs) if (m_idx >= lo and m_idx < hi) or (j == len(segs)-1 and m_idx == hi))
#             lo, _ = segs[seg_id]
#             d_in_seg = distance_along_branch(nb[tp], lo, m_idx)
#             scaled_d = d_in_seg * scale_factors_all[tp][seg_id]
#             fm.append(cum_scaled[tp][seg_id] + scaled_d)
#         final_marker_distance.append(fm)

#     return ComputationResult(
#         normalized_branch=nb,
#         normalized_markers=nm,
#         normalized_fiducials=nf,
#         raw_branch=raw_branch,
#         raw_markers=raw_markers,
#         raw_fiducials=raw_fiducials,
#         closest_branch_idx_markers=cb_markers,
#         closest_branch_idx_fids=cb_fids,
#         segments_all=segments_all,
#         seg_lengths_all=seg_lengths_all,
#         cumdist_unscaled=cum_unscaled,
#         cumdist_scaled=cum_scaled,
#         scale_factors_all=scale_factors_all,
#         final_marker_distance=final_marker_distance,
#         raw_marker_coords_only=raw_marker_coords_only,
#         raw_marker_types_only=raw_marker_types_only,
#     )
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import os
import numpy as np
import pandas as pd

from config import Settings
from io_utils import read_branch_csv, read_markers_csv_list, z_imagej_to_objectj
from geometry import fit_branch_spline, get_xyzs, euc_xy, distance_along_branch

from scipy.signal import find_peaks  # NEW

Coord = Tuple[float, float, float]

# ============================================================
# Helpers for DTW + curvature landmarks
# ============================================================

def _to_xy(arr):
    """Ensure array is (N, 2) from (N,), (N,2), (N,3,...)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.stack([arr, np.zeros_like(arr)], axis=1)
    if arr.shape[1] >= 2:
        return arr[:, :2]
    pad = np.zeros((arr.shape[0], 1), dtype=arr.dtype)
    return np.hstack([arr, pad])

def compute_arclength(branch_xy: np.ndarray):
    """
    Given branch points (N,2), return:
      s: arclength at each vertex (length N)
      L: total length
      t_norm: normalized arclength in [0,1]
    """
    branch_xy = _to_xy(branch_xy)
    diffs = np.diff(branch_xy, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    L = float(s[-1]) if s[-1] > 0 else 1.0
    t_norm = s / L
    return s, L, t_norm

def resample_branch_equal_arclen(branch_xy: np.ndarray, n_samp: int = 200):
    """
    Resample the branch to n_samp points at equal arclength fractions in [0,1].
    Returns resampled points of shape (n_samp, 2) and the t grid.
    """
    branch_xy = _to_xy(branch_xy)
    s, L, t_norm = compute_arclength(branch_xy)

    t_grid = np.linspace(0.0, 1.0, n_samp)
    x = np.interp(t_grid, t_norm, branch_xy[:, 0])
    y = np.interp(t_grid, t_norm, branch_xy[:, 1])
    pts = np.stack([x, y], axis=1)
    return pts, t_grid

def compute_tangent_angles(branch_xy: np.ndarray,
                           n_samp: int = 200,
                           coord_smooth_window: int = 7,
                           theta_smooth_window: int = 11):
    """
    See your original definition: tangent angle signal θ(t) along branch.
    """
    pts, t_grid = resample_branch_equal_arclen(branch_xy, n_samp=n_samp)
    N = pts.shape[0]

    # Smooth coordinates
    if coord_smooth_window >= 3 and coord_smooth_window % 2 == 1:
        kernel = np.ones(coord_smooth_window, dtype=float) / coord_smooth_window
        x = np.convolve(pts[:, 0], kernel, mode="same")
        y = np.convolve(pts[:, 1], kernel, mode="same")
        pts_s = np.stack([x, y], axis=1)
    else:
        pts_s = pts

    # Finite differences for tangents
    diffs = np.zeros_like(pts_s)
    if N >= 2:
        diffs[0] = pts_s[1] - pts_s[0]
        diffs[-1] = pts_s[-1] - pts_s[-2]
    if N >= 3:
        diffs[1:-1] = (pts_s[2:] - pts_s[:-2]) * 0.5

    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = diffs / norms

    theta = np.arctan2(tangents[:, 1], tangents[:, 0])
    theta_unwrapped = np.unwrap(theta)

    # Smooth θ itself
    if theta_smooth_window >= 3 and theta_smooth_window % 2 == 1:
        k = np.ones(theta_smooth_window, dtype=float) / theta_smooth_window
        theta_smooth = np.convolve(theta_unwrapped, k, mode="same")
    else:
        theta_smooth = theta_unwrapped

    return theta_smooth, t_grid

def curvature_scores_along_branch(branch_xy: np.ndarray,
                                  n_samp: int = 200,
                                  coord_smooth_window: int = 9,
                                  curv_smooth_window: int = 9):
    """
    Curvature-like magnitude κ(t) along normalized arclength.
    """
    pts, t_grid = resample_branch_equal_arclen(branch_xy, n_samp=n_samp)
    N = pts.shape[0]

    # Smooth coordinates
    if coord_smooth_window >= 3 and coord_smooth_window % 2 == 1:
        kcoord = np.ones(coord_smooth_window, dtype=float) / coord_smooth_window
        x = np.convolve(pts[:, 0], kcoord, mode="same")
        y = np.convolve(pts[:, 1], kcoord, mode="same")
        pts_s = np.stack([x, y], axis=1)
    else:
        pts_s = pts

    # Second derivative
    d2 = np.zeros((N, 2), dtype=float)
    if N >= 3:
        core = pts_s[2:, :] - 2 * pts_s[1:-1, :] + pts_s[:-2, :]
        d2[1:-1, :] = core

    kappa = np.linalg.norm(d2, axis=1)

    # Smooth κ
    if curv_smooth_window >= 3 and curv_smooth_window % 2 == 1:
        kcurv = np.ones(curv_smooth_window, dtype=float) / curv_smooth_window
        kappa_smooth = np.convolve(kappa, kcurv, mode="same")
    else:
        kappa_smooth = kappa

    return kappa_smooth, t_grid

def dtw_path(seq1: np.ndarray, seq2: np.ndarray):
    """
    Classic DTW for 1D sequences.
    Returns a list of (i, j) indices giving the alignment path.
    """
    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)
    n, m = len(seq1), len(seq2)

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 = diag, 1 = up, 2 = left

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (seq1[i - 1] - seq2[j - 1]) ** 2
            choices = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(choices))
            D[i, j] = cost + choices[k]
            trace[i, j] = k

    # Backtrack
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = trace[i, j]
        if move == 0:   # diag
            i -= 1
            j -= 1
        elif move == 1: # up
            i -= 1
        else:           # left
            j -= 1

    path.reverse()
    return path

def build_warp_from_dtw(t_ref: np.ndarray,
                        t_src: np.ndarray,
                        path: List[tuple]):
    """
    Build a monotone, piecewise-linear warp f: t_src -> t_ref from DTW path.
    """
    t_src_samples = []
    t_ref_samples = []

    last_src = None
    for i_ref, j_src in path:
        src_t = float(t_src[j_src])
        ref_t = float(t_ref[i_ref])
        if last_src is None or src_t > last_src:
            t_src_samples.append(src_t)
            t_ref_samples.append(ref_t)
            last_src = src_t
        else:
            t_ref_samples[-1] = ref_t

    # Ensure endpoints
    if 0.0 not in t_src_samples:
        t_src_samples.insert(0, 0.0)
        t_ref_samples.insert(0, t_ref[0])
    if 1.0 not in t_src_samples:
        t_src_samples.append(1.0)
        t_ref_samples.append(t_ref[-1])

    t_src_arr = np.array(t_src_samples)
    t_ref_arr = np.array(t_ref_samples)

    def warp_func(t: float) -> float:
        return float(
            np.interp(
                t,
                t_src_arr,
                t_ref_arr,
                left=t_ref_arr[0],
                right=t_ref_arr[-1],
            )
        )

    return warp_func, t_src_arr, t_ref_arr

# ============================================================
# Original dataclass and helpers
# ============================================================

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
        return (
            (c[0] - minx) * scale[0],
            (c[1] - miny) * scale[1],
            (c[2] - minz) * scale[2],
        )

    return list(map(_xf, branch)), list(map(_xf, markers)), list(map(_xf, fiducials))


def _branch_points_from_csvs(settings: Settings) -> List[List[Coord]]:
    raw = []
    for i in range(settings.n_timepoints):
        df = read_branch_csv(
            settings.branch_csvs[f"Timepoint {i+1}"],
            settings.snt_branch_fmt % settings.branch_id,
        )
        x, y, z = df["x"].values, df["y"].values, df["z"].values
        raw.append(fit_branch_spline(x, y, z, n_points=100))
    return raw


def compute(settings: Settings) -> ComputationResult:
    # --------------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------------
    raw_branch = _branch_points_from_csvs(settings)
    raw_markers, raw_fids_typed = read_markers_csv_list(
        settings.marker_csvs, settings.num_channels
    )
    # No manual fiducials file anymore

    # Strip types for markers into parallel arrays
    raw_marker_coords_only = [[m[1] for m in tp] for tp in raw_markers]
    raw_marker_types_only = [[m[0] for m in tp] for tp in raw_markers]

    # --------------------------------------------------------
    # 2. Normalize & scale (branch + markers only)
    # --------------------------------------------------------
    nb, nm, nf = [], [], []
    for i in range(settings.n_timepoints):
        # Pass empty fiducials list; we'll fill nf later from landmarks
        b, m, f = _normalize_and_scale(
            raw_branch[i],
            raw_marker_coords_only[i],
            [],
            settings.scaling_factor,
        )
        nb.append(b)
        nm.append(m)
        nf.append(f)  # currently [], will be replaced with landmark coords

    # --------------------------------------------------------
    # 3. Map markers to nearest branch vertex (for arclength param)
    # --------------------------------------------------------
    def nearest_indices(points, branch):
        idxs = []
        for p in points:
            dists = [euc_xy(p, q) for q in branch]
            idxs.append(int(np.argmin(dists)))
        return idxs

    cb_markers: List[List[int]] = []
    for i in range(settings.n_timepoints):
        cb_markers.append(nearest_indices(nm[i], nb[i]))

    # --------------------------------------------------------
    # 4. DTW-based alignment of branches (tangent angle signals)
    #    and curvature-based landmarks
    # --------------------------------------------------------
    n_samp_theta = 200

    theta_all: List[np.ndarray] = []
    tgrid_all: List[np.ndarray] = []
    s_all: List[np.ndarray] = []
    L_all: List[float] = []
    t_norm_all: List[np.ndarray] = []

    for i in range(settings.n_timepoints):
        branch_xy = _to_xy(np.array(nb[i], dtype=float))
        s_i, L_i, t_norm_i = compute_arclength(branch_xy)
        s_all.append(s_i)
        L_all.append(L_i)
        t_norm_all.append(t_norm_i)

        theta_i, t_grid_i = compute_tangent_angles(
            branch_xy,
            n_samp=n_samp_theta,
            coord_smooth_window=7,
            theta_smooth_window=11,
        )
        theta_all.append(theta_i)
        tgrid_all.append(t_grid_i)

    ref_tp = 0
    theta_ref = theta_all[ref_tp]
    t_ref_grid = tgrid_all[ref_tp]

    warp_funcs: List = []
    warp_src_samples: List[np.ndarray] = []
    warp_ref_samples: List[np.ndarray] = []

    for tp in range(settings.n_timepoints):
        if tp == ref_tp:
            # identity warp
            warp_funcs.append(lambda t, _tg=t_ref_grid: float(t))
            warp_src_samples.append(t_ref_grid.copy())
            warp_ref_samples.append(t_ref_grid.copy())
            continue

        theta_src = theta_all[tp]
        t_src_grid = tgrid_all[tp]

        path = dtw_path(theta_ref, theta_src)
        f_tp, t_src_arr, t_ref_arr = build_warp_from_dtw(
            t_ref_grid, t_src_grid, path
        )
        warp_funcs.append(f_tp)
        warp_src_samples.append(t_src_arr)
        warp_ref_samples.append(t_ref_arr)

    # --------------------------------------------------------
    # 5. Curvature-based landmarks on reference branch
    # --------------------------------------------------------
    end_clip_frac = 0.08
    min_peak_frac = 0.25
    min_idx_sep_frac = 0.10
    max_landmarks = 3
    fallback_K = 2

    branch_ref_xy = _to_xy(np.array(nb[ref_tp], dtype=float))
    kappa_ref, t_ref_curv = curvature_scores_along_branch(
        branch_ref_xy,
        n_samp=200,
        coord_smooth_window=11,
        curv_smooth_window=11,
    )

    kappa_for_peaks = kappa_ref.copy()
    tip_mask = (t_ref_curv <= end_clip_frac) | (
        t_ref_curv >= 1.0 - end_clip_frac
    )
    kappa_for_peaks[tip_mask] = 0.0

    min_idx_sep = max(1, int(len(kappa_for_peaks) * min_idx_sep_frac))
    peaks_all, _ = find_peaks(kappa_for_peaks, distance=min_idx_sep)

    if len(peaks_all) == 0:
        # Fallback: nearly straight branch
        t_ref_landmarks = np.linspace(0.0, 1.0, fallback_K + 2)[1:-1]
    else:
        peak_vals = kappa_for_peaks[peaks_all]
        max_val = float(peak_vals.max())
        height_thresh = min_peak_frac * max_val

        keep_mask = peak_vals >= height_thresh
        peaks_kept = peaks_all[keep_mask]

        if len(peaks_kept) == 0:
            peaks_kept = peaks_all

        if len(peaks_kept) > max_landmarks:
            vals_kept = peak_vals[keep_mask] if keep_mask.any() else peak_vals
            order = np.argsort(vals_kept)[::-1]
            chosen_idx = order[:max_landmarks]
            peaks_chosen = peaks_kept[chosen_idx]
        else:
            peaks_chosen = peaks_kept

        peaks_chosen = np.sort(peaks_chosen)
        t_ref_landmarks = t_ref_curv[peaks_chosen]

    # --------------------------------------------------------
    # 6. Map those landmarks to every timepoint via DTW warp inverse
    #    -> landmark indices on each branch = "virtual fiducials"
    # --------------------------------------------------------
    landmarks_all: List[List[int]] = []

    for tp in range(settings.n_timepoints):
        branch_xy = _to_xy(np.array(nb[tp], dtype=float))
        s_i, L_i, t_norm_i = compute_arclength(branch_xy)

        if tp == ref_tp:
            t_src_landmarks = t_ref_landmarks
        else:
            t_src_arr = warp_src_samples[tp]
            t_ref_arr = warp_ref_samples[tp]
            t_src_landmarks = np.interp(
                t_ref_landmarks, t_ref_arr, t_src_arr
            )

        landmark_indices_tp = [
            int(np.argmin(np.abs(t_norm_i - t_src_l)))
            for t_src_l in t_src_landmarks
        ]
        landmark_indices_tp = sorted(set(landmark_indices_tp))
        landmarks_all.append(landmark_indices_tp)

    # Use these curvature/DTW landmarks as fiducial branch indices
    cb_fids: List[List[int]] = landmarks_all

    # Also define normalized/raw "fiducial" coordinates for completeness
    nf = [
        [nb[tp][idx] for idx in cb_fids[tp]]
        for tp in range(settings.n_timepoints)
    ]
    raw_fiducials = [
        [raw_branch[tp][idx] for idx in cb_fids[tp]]
        for tp in range(settings.n_timepoints)
    ]

    # --------------------------------------------------------
    # 7. Segments & lengths (unchanged logic, but using cb_fids)
    # --------------------------------------------------------
    segments_all: List[List[Tuple[int, int]]] = []
    seg_lengths_all: List[List[float]] = []

    for i in range(settings.n_timepoints):
        idxs = sorted([0, len(nb[i]) - 1] + cb_fids[i])
        if len(idxs) != len(set(idxs)):
            raise RuntimeError(
                f"Duplicate fiducial-to-branch indices at timepoint {i+1}"
            )
        segs, lens = [], []
        for a, b in zip(idxs[:-1], idxs[1:]):
            if a == b:
                lens.append(0.0)
                segs.append((a, b))
                continue
            lens.append(distance_along_branch(nb[i], a, b))
            segs.append((a, b))
        segments_all.append(segs)
        seg_lengths_all.append(lens)

    # --------------------------------------------------------
    # 8. Max per segment across timepoints, per-timepoint scale factors
    # --------------------------------------------------------
    max_per_seg = (
        [max(vals) for vals in zip(*seg_lengths_all)]
        if seg_lengths_all
        else []
    )
    scale_factors_all: List[List[float]] = []
    for i in range(settings.n_timepoints):
        sfs = []
        for seg_len, seg_max in zip(seg_lengths_all[i], max_per_seg):
            sfs.append(0.01 if seg_len == 0 else seg_max / seg_len)
        scale_factors_all.append(sfs)

    # Scaled segment lengths + cumulative distances
    seg_lengths_scaled = [
        [
            l * sf
            for l, sf in zip(seg_lengths_all[i], scale_factors_all[i])
        ]
        for i in range(settings.n_timepoints)
    ]

    def cumdist(lengths):
        out = [0.0]
        c = 0.0
        for L in lengths:
            c += L
            out.append(c)
        return out

    cum_unscaled = [
        cumdist(seg_lengths_all[i]) for i in range(settings.n_timepoints)
    ]
    cum_scaled = [
        cumdist(seg_lengths_scaled[i]) for i in range(settings.n_timepoints)
    ]

    # --------------------------------------------------------
    # 9. Final per-marker cumulative distances (same as before)
    # --------------------------------------------------------
    final_marker_distance: List[List[float]] = []
    for tp in range(settings.n_timepoints):
        fm = []
        for m_idx in cb_markers[tp]:
            segs = segments_all[tp]
            seg_id = next(
                j
                for j, (lo, hi) in enumerate(segs)
                if (m_idx >= lo and m_idx < hi)
                or (j == len(segs) - 1 and m_idx == hi)
            )
            lo, _ = segs[seg_id]
            d_in_seg = distance_along_branch(nb[tp], lo, m_idx)
            scaled_d = d_in_seg * scale_factors_all[tp][seg_id]
            fm.append(cum_scaled[tp][seg_id] + scaled_d)
        final_marker_distance.append(fm)

    # --------------------------------------------------------
    # 10. Return result
    # --------------------------------------------------------
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
        for plot_idx, (x, y) in zip(kept_indices, marker_coords_filtered):
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

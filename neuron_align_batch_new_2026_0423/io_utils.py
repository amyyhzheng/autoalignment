from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from config import Settings

# Z conversions
def z_objectj_to_imagej(z_obj: float, num_channels: int) -> float:
    return (int(z_obj) - 1) / num_channels

def z_imagej_to_objectj(z_img: float, num_channels: int) -> float:
    return z_img * num_channels + 1

# Readers
def read_branch_csv(path: Path, branch_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["path"] == branch_name].copy()

def extract_branch_bounds(raw_markers):
    """
    raw_markers: List[timepoint] of List[(label, (x,y,z))]

    Returns:
        bounds: List[timepoint] of {"start": coord, "end": coord}
    """
    bounds = []

    for tp_idx, tp_markers in enumerate(raw_markers):
        start_pts = [coord for label, coord in tp_markers if str(label).strip().lower() == "start_branch"]
        end_pts   = [coord for label, coord in tp_markers if str(label).strip().lower() == "end_branch"]

        if len(start_pts) != 1 or len(end_pts) != 1:
            raise ValueError(
                f"Timepoint {tp_idx+1} must contain exactly one start_branch and one end_branch "
                f"(got {len(start_pts)} start_branch, {len(end_pts)} end_branch)."
            )

        bounds.append({
            "start": start_pts[0],
            "end": end_pts[0],
        })

    return bounds

def nearest_index(point, branch):
    dists = [euc_xy(point, q) for q in branch]
    return int(np.argmin(dists))

def crop_branch_and_filter_markers(branch, markers, fiducials=None):
    """
    branch: List[Coord]
    markers: List[(label, coord)]
    fiducials: optional List[(label, coord)] or List[coord]

    Returns:
        cropped_branch
        filtered_markers
        filtered_fiducials
        meta = {"lo": lo, "hi": hi, "start_idx": i0, "end_idx": i1}
    """
    fiducials = fiducials or []

    start_pts = [coord for label, coord in markers if str(label).strip().lower() == "start_branch"]
    end_pts   = [coord for label, coord in markers if str(label).strip().lower() == "end_branch"]

    if len(start_pts) != 1 or len(end_pts) != 1:
        raise ValueError(
            f"Need exactly one start_branch and one end_branch "
            f"(got {len(start_pts)} start_branch, {len(end_pts)} end_branch)."
        )

    i0 = nearest_index(start_pts[0], branch)
    i1 = nearest_index(end_pts[0], branch)
    lo, hi = sorted((i0, i1))

    cropped_branch = branch[lo:hi+1]

    filtered_markers = []
    for label, coord in markers:
        label_l = str(label).strip().lower()

        # remove control markers from downstream biological marker set
        if label_l in {"start_branch", "end_branch"}:
            continue

        idx = nearest_index(coord, branch)
        if lo <= idx <= hi:
            filtered_markers.append((label, coord))

    filtered_fiducials = []
    for item in fiducials:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            label, coord = item
        else:
            coord = item

        idx = nearest_index(coord, branch)
        if lo <= idx <= hi:
            filtered_fiducials.append(item)

    meta = {"lo": lo, "hi": hi, "start_idx": i0, "end_idx": i1}
    return cropped_branch, filtered_markers, filtered_fiducials, meta

# def read_markers_csv_list(marker_files: List[Path], num_channels: int):
#     raw_markers = []
#     raw_fiducials = []
#     for tp_idx, fp in enumerate(marker_files):
#         df = pd.read_csv(fp)
#         tp_markers, tp_fids = [], []
#         if "type" in df.columns:
#             label_col = "type"
#         elif "label" in df.columns:
#             label_col = "label"
#         else:
#             raise ValueError("Markers CSV missing 'type' or 'label' column")
#         for _, row in df.iterrows():
#             mtype = row[label_col]
#             # Napari points convention (z, x, y) or similar — keep your original mapping
#             z = row.get("axis-0")
#             x = row.get("axis-1")
#             y = row.get("axis-2", 0)
#             z_img = z_objectj_to_imagej(z, num_channels)
#             tup = (mtype, (x, y, z_img))
#             if str(mtype).lower() == "landmark":
#                 tp_fids.append(tup)
#             else:
#                 tp_markers.append(tup)
#         raw_markers.append(tp_markers)
#         raw_fiducials.append(tp_fids)
#     return raw_markers, raw_fiducials


# def read_fiducials_csv(fiducials_csv: Path, n_timepoints: int, num_channels: int):
#     df = pd.read_csv(fiducials_csv)
#     if "timepoint" not in df.columns:
#         raise ValueError("Fiducials CSV must contain a 'timepoint' column")
#     out = []
#     for i in range(n_timepoints):
#         tp = df[df["timepoint"] == i]
#         coords = []
#         for _, r in tp.iterrows():
#             x = r.get("axis-0")
#             y = r.get("axis-1")
#             z = r.get("axis-2", 0)
#             z_img = z_objectj_to_imagej(z, num_channels)
#             coords.append((x, y, z_img))
#         out.append(coords)
#     return out

def _read_any_csv(fp) -> pd.DataFrame:
    return pd.read_csv(fp, sep=None, engine="python")

def _pick_first(df: pd.DataFrame, cols: List[str]) -> str:
    for c in cols:
        if c in df.columns:
            return c
    raise ValueError(f"None of the expected columns are present: {cols}")

# def read_markers_csv_list(marker_files: List[Path], num_channels: int):
#     """
#     ObjectJ CombinedResults.csv reader (list-aware).
#     Each entry in marker_files corresponds to one timepoint index (tp_idx starting at 0).
#     We filter that CSV to rows whose 'ojj File Name' contains '_Image{tp_idx+1}'.

#     Returns:
#         raw_markers:   List[timepoint] of List[(label, (x,y,z_img))] excluding landmarks
#         raw_fiducials: List[timepoint] of List[(label, (x,y,z_img))] where label == 'Landmark'
#     """
#     # allow caller to pass a single path as a string/Path
#     if not isinstance(marker_files, (list, tuple)):
#         marker_files = [marker_files]

#     raw_markers: List[List[Tuple[str, Tuple[float, float, float]]]] = []
#     raw_fiducials: List[List[Tuple[str, Tuple[float, float, float]]]] = []

#     for tp_idx, fp in enumerate(marker_files):
#         df = _read_any_csv(fp)

#         # Filter rows to this timepoint by matching ..._Image{tp} in 'ojj File Name'
#         if "ojj File Name" in df.columns:
#             mask = df["ojj File Name"].astype(str).str.contains(
#                 fr"_Image{tp_idx+1}\b", na=False
#             )
#             df = df.loc[mask].copy()

#         # Label + coordinates (ObjectJ narrow S1 columns)
#         # If you only want Final S1, keep the first option only.
#         label_col = _pick_first(df, ["Final S1", "Checked S1", "Original S1", "S 1", "label", "type"])
#         x_col = _pick_first(df, ["xpos S1", "x"])
#         y_col = _pick_first(df, ["ypos S1", "y"])
#         z_col = _pick_first(df, ["zpos S1", "z"])

#         tp_markers, tp_fids = [], []

#         for _, row in df.iterrows():
#             mtype = row.get(label_col)
#             if pd.isna(mtype) or str(mtype).strip() == "":
#                 continue
#             mtype = str(mtype).strip()

#             try:
#                 x = float(row.get(x_col))
#                 y = float(row.get(y_col))
#                 z_raw = float(row.get(z_col))
#             except (TypeError, ValueError):
#                 continue

#             # Use your existing converter
#             z_img = z_objectj_to_imagej(z_raw, num_channels)
#             item = (mtype, (x, y, z_img))

#             if mtype.lower() == "landmark":
#                 tp_fids.append(item)
#             else:
#                 tp_markers.append(item)

#         raw_markers.append(tp_markers)
#         raw_fiducials.append(tp_fids)

#         print(f"[markers] Timepoint {tp_idx+1}: {len(tp_markers)} markers, {len(tp_fids)} landmarks.")

#     return raw_markers, raw_fiducials
def read_markers_csv_list(marker_files, num_channels=4):
    """
    Napari Points CSV reader (list-aware).

    Supports each item in marker_files being either:
      - a CSV path
      - a dict with:
            {"markers": <path>, "landmarks": <path or None>}

    Expected columns:
        axis-0, axis-1, axis-2, type, notes, column

    Returns:
        raw_markers:   List[timepoint] of List[(label, (x,y,z))]
        raw_fiducials: List[timepoint] of List[(label, (x,y,z))] where type == "Landmark"
    """
    fiducial_type = "Landmark"

    if not isinstance(marker_files, (list, tuple)):
        marker_files = [marker_files]

    raw_markers: List[List[Tuple[str, Tuple[float, float, float]]]] = []
    raw_fiducials: List[List[Tuple[str, Tuple[float, float, float]]]] = []

    def _parse_csv(fp):
        df = _read_any_csv(fp)

        required = ["axis-0", "axis-1", "axis-2", "type"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Napari CSV missing columns {missing}. Found: {list(df.columns)}"
            )

        tp_markers = []
        tp_fids = []

        for _, row in df.iterrows():
            mtype = row.get("type")
            if pd.isna(mtype) or str(mtype).strip() == "":
                continue
            mtype = str(mtype).strip()

            try:
                z = float(row["axis-0"]) * 0.25
                y = float(row["axis-1"])
                x = float(row["axis-2"])
            except (TypeError, ValueError):
                continue

            item = (mtype, (x, y, z))

            if mtype.lower() == fiducial_type.lower():
                tp_fids.append(item)
            else:
                tp_markers.append(item)

        return tp_markers, tp_fids

    for tp_idx, item in enumerate(marker_files):
        if isinstance(item, dict):
            marker_fp = item["markers"]
            landmark_fp = item.get("landmarks", None)
        else:
            marker_fp = item
            landmark_fp = None

        tp_markers, tp_fids = _parse_csv(marker_fp)

        if landmark_fp is not None and Path(landmark_fp).exists():
            lm_markers, lm_fids = _parse_csv(landmark_fp)

            if lm_markers:
                print(
                    f"[warning] Timepoint {tp_idx+1}: landmark file {Path(landmark_fp).name} "
                    f"contains {len(lm_markers)} non-Landmark rows; ignoring them."
                )

            tp_fids.extend(lm_fids)

            print(
                f"[napari landmarks] Timepoint {tp_idx+1}: "
                f"added {len(lm_fids)} landmarks from {Path(landmark_fp).name}"
            )

        raw_markers.append(tp_markers)
        raw_fiducials.append(tp_fids)

        print(
            f"[napari markers] Timepoint {tp_idx+1}: "
            f"{len(tp_markers)} markers, {len(tp_fids)} {fiducial_type}s."
        )

    return raw_markers, raw_fiducials

# def read_fiducials_csv(fiducial_filename: str, number_of_timepoints = 6, num_channels: int = 4):
#     """
#     Reads ObjectJ CombinedResults.csv for fiducials.
#     Returns list[timepoint] of [(x,y,z_img)] for landmarks only.
#     """
#     file = pd.read_csv(fiducial_filename)
#     raw_fiducials = [[] for _ in range(number_of_timepoints)]

#     for _, row in file.iterrows():
#         for tp in range(1, number_of_timepoints+1):
#             label_cols = [f"Final S{tp}", f"Checked S{tp}", f"Original S{tp}", f"S {tp}"]
#             has_landmark = any(
#                 (col in row and pd.notna(row[col]) and str(row[col]).strip().lower() == "landmark")
#                 for col in label_cols
#             )
#             if not has_landmark:
#                 continue
#             x = row[f"xpos S{tp}"]
#             y = row[f"ypos S{tp}"]
#             z_obj = row[f"zpos S{tp}"]
#             z_img = z_imagej_to_objectj(z_obj, 4)
#             raw_fiducials[tp-1].append((int(x), int(y), z_img))

#     for tp, coords in enumerate(raw_fiducials, start=1):
#         print(f"Parsed {len(coords)} fiducials for Timepoint/Image {tp}")

#     return raw_fiducials

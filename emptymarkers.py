from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import csv
import re

import numpy as np
import pandas as pd

from config import Settings
from computation import compute, ComputationResult
from geometry import distance_along_branch
from io_utils import z_imagej_to_objectj, _read_any_csv


Coord = Tuple[float, float, float]
_IMAGE_RE = re.compile(r"Image(\d+)", re.IGNORECASE)


MARKER_BRANCH_DIR = Path(
   '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5/Alignment_and_checking/AfterManualEdits/checked_Branch1'
).expanduser().resolve()

TRACE_ROOT_DIR = Path(
    '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5/SNTTrace'
).expanduser().resolve()

EXPORT_DIR = Path(
    '/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM055_DOB051322_TT/Analysis_withAmyCode_cell5'
).expanduser().resolve()

BRANCH_ID = "1"
ANIMAL_ID = "SOM055"

TARGET_TP_ZERO_BASED = 2

OUTPUT_SUFFIX = "_with_backfilled_missing"


# ============================================================
# Marker CSV reader (supports separate landmark CSVs)
# ============================================================
def read_markers_csv_list(marker_files, num_channels=4):
    """
    Supports each item in marker_files being either:
      - a CSV path
      - a dict with:
            {"markers": <path>, "landmarks": <path or None>}

    Expected columns:
        axis-0, axis-1, axis-2, type, ...

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

            # Filter out anything containing "empty"
            if "empty" in mtype.lower():
                continue

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


# monkey-patch computation module if needed
import computation as computation_module
computation_module.read_markers_csv_list = read_markers_csv_list


# ============================================================
# Path / file collectors
# ============================================================
def extract_image_index(p: Path) -> Optional[int]:
    m = _IMAGE_RE.search(p.name)
    return int(m.group(1)) if m else None


def collect_marker_csvs_by_image(branch_dir: Path) -> dict[int, Path]:
    by_img: dict[int, Path] = {}
    for p in branch_dir.glob("*.csv"):
        if "landmark" in p.name.lower():
            continue

        idx = extract_image_index(p)
        if idx is None:
            continue

        if idx not in by_img or len(p.name) < len(by_img[idx].name):
            by_img[idx] = p
    return by_img


def collect_landmark_csvs_by_image(branch_dir: Path) -> dict[int, Path]:
    by_img: dict[int, Path] = {}
    for p in branch_dir.glob("*.csv"):
        if "landmark" not in p.name.lower():
            continue

        idx = extract_image_index(p)
        if idx is None:
            continue

        if idx not in by_img or len(p.name) < len(by_img[idx].name):
            by_img[idx] = p
    return by_img


def collect_trace_csv_by_image(trace_root_dir: Path) -> dict[int, Path]:
    by_img: dict[int, Path] = {}

    for img_dir in trace_root_dir.glob("Image*"):
        if not img_dir.is_dir():
            continue

        idx = extract_image_index(img_dir)
        if idx is None:
            continue

        candidates = list(img_dir.glob("*_Trace_xyzCoordinates.csv"))
        if not candidates:
            continue

        filtered = [p for p in candidates if ".traces" not in p.name]
        picks = filtered if filtered else candidates
        picks = sorted(picks, key=lambda p: (len(p.name), p.name))
        by_img[idx] = picks[0]

    return by_img


def build_aligned_inputs(marker_branch_dir: Path, trace_root_dir: Path):
    marker_by_img = collect_marker_csvs_by_image(marker_branch_dir)
    landmark_by_img = collect_landmark_csvs_by_image(marker_branch_dir)
    trace_by_img = collect_trace_csv_by_image(trace_root_dir)

    common_imgs = sorted(set(marker_by_img) & set(trace_by_img))
    if not common_imgs:
        raise FileNotFoundError(
            f"No overlapping Image# between markers in {marker_branch_dir} and traces in {trace_root_dir}.\n"
            f"Markers have: {sorted(marker_by_img.keys())}\n"
            f"Traces have: {sorted(trace_by_img.keys())}"
        )

    marker_csvs = []
    for img_idx in common_imgs:
        marker_csvs.append({
            "markers": marker_by_img[img_idx],
            "landmarks": landmark_by_img.get(img_idx, None),
        })

    branch_csvs: dict[str, Path] = {}
    for tp_idx, img_idx in enumerate(common_imgs):
        branch_csvs[f"Timepoint {tp_idx+1}"] = trace_by_img[img_idx]

    print(f"[build_aligned_inputs] Using Image indices: {common_imgs}")
    return marker_csvs, branch_csvs


# ============================================================
# Exported CSV parsing
# ============================================================
def read_exported_tp_csv(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def is_landmark_export_row(row: dict) -> bool:
    return str(row.get("type", "")).strip().lower() == "ambiguous"


def is_empty_type(type_str: str) -> bool:
    t = str(type_str).strip().lower()
    return t in {"empty_shaft", "empty_spine"}


def is_real_synapse_type(type_str: str) -> bool:
    t = str(type_str).strip().lower()
    return t in {"shaft_syntdnotscored", "spine_syntdnotscored"}


def infer_empty_type_from_rows(rows: List[dict]) -> str:
    real_types = [
        str(r["type"]).strip().lower()
        for r in rows
        if is_real_synapse_type(r.get("type", ""))
    ]
    if any(t == "spine_syntdnotscored" for t in real_types):
        return "Empty_spine"
    if any(t == "shaft_syntdnotscored" for t in real_types):
        return "Empty_shaft"

    all_types = [str(r["type"]).strip().lower() for r in rows]
    if any(t == "empty_spine" for t in all_types):
        return "Empty_spine"
    return "Empty_shaft"


def parse_float_or_none(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def normalize_label(label) -> str:
    return str(label).strip()


def collect_rows_by_id(export_dir: Path, branch_id: str) -> Dict[str, List[dict]]:
    files = sorted(export_dir.rglob(f"Image*_branch{branch_id}.csv"))
    if not files:
        raise FileNotFoundError(
            f"No exported timepoint CSVs found under {export_dir} for branch {branch_id}"
        )

    print("Found exported CSVs:")
    for p in files:
        print(f"  {p}")

    by_id: Dict[str, List[dict]] = defaultdict(list)

    for path in files:
        m = re.search(r"Image(\d+)_branch", path.name, re.IGNORECASE)
        if not m:
            continue
        tp_zero_based = int(m.group(1))

        rows = read_exported_tp_csv(path)
        for row in rows:
            row = dict(row)
            row["tp_zero_based"] = tp_zero_based

            if is_landmark_export_row(row):
                continue

            # Empty rows should not count as present
            if is_empty_type(row.get("type", "")):
                continue

            label = normalize_label(row.get("label", ""))
            if label == "":
                continue

            by_id[label].append(row)

    return by_id


def exported_csv_path(export_dir: Path, branch_id: str, tp_zero_based: int) -> Path:
    matches = list(export_dir.rglob(f"Image{tp_zero_based}_branch{branch_id}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find Image{tp_zero_based}_branch{branch_id}.csv under {export_dir}"
        )
    if len(matches) > 1:
        print("Multiple target CSV matches found; using first:")
        for m in matches:
            print(f"  {m}")
    return matches[0]


# ==============================
# Geometry / estimation
# ==============================
def estimate_coordinate_from_scaled_distance(
    tp: int,
    scaled_dist: float,
    result: ComputationResult,
) -> Coord:
    cum_scaled = result.cumdist_scaled[tp]
    cum_unscaled = result.cumdist_unscaled[tp]
    scale_factors = result.scale_factors_all[tp]
    norm_branch = result.normalized_branch[tp]
    raw_branch = result.raw_branch[tp]

    if scaled_dist < 0:
        scaled_dist = 0.0
    if scaled_dist > cum_scaled[-1]:
        scaled_dist = cum_scaled[-1]

    seg_id = None
    for j in range(len(cum_scaled) - 1):
        if scaled_dist <= cum_scaled[j + 1]:
            seg_id = j
            break
    if seg_id is None:
        seg_id = len(cum_scaled) - 2

    if scale_factors[seg_id] == 0:
        unscaled_dist = cum_unscaled[seg_id]
    else:
        unscaled_dist = (
            (scaled_dist - cum_scaled[seg_id]) / scale_factors[seg_id]
            + cum_unscaled[seg_id]
        )

    for i in range(len(norm_branch)):
        d = distance_along_branch(norm_branch, 0, i)
        if d >= unscaled_dist - 1e-8:
            return raw_branch[i]

    return raw_branch[-1]


def average_scaled_distance_from_rows(rows: List[dict]) -> Optional[float]:
    vals = []
    for r in rows:
        v = parse_float_or_none(r.get("Notes"))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def average_translation_from_rows(
    rows: List[dict],
    result: ComputationResult,
) -> Tuple[float, float]:
    dxs = []
    dys = []

    for r in rows:
        if not is_real_synapse_type(r.get("type", "")):
            continue

        tp = int(r["tp_zero_based"])
        scaled_dist = parse_float_or_none(r.get("Notes"))
        if scaled_dist is None:
            continue

        est_x, est_y, _ = estimate_coordinate_from_scaled_distance(tp, scaled_dist, result)

        x = parse_float_or_none(r.get("axis-2"))
        y = parse_float_or_none(r.get("axis-1"))
        if x is None or y is None:
            continue

        dxs.append(x - est_x)
        dys.append(y - est_y)

    if not dxs:
        return 0.0, 0.0

    return float(np.mean(dxs)), float(np.mean(dys))


# ============================================================
# Backfill logic
# ============================================================
def ids_present_in_target(by_id: Dict[str, List[dict]], target_tp_zero_based: int) -> set[str]:
    out = set()
    for label, rows in by_id.items():
        if any(int(r["tp_zero_based"]) == target_tp_zero_based for r in rows):
            out.add(label)
    return out


def ids_present_elsewhere(by_id: Dict[str, List[dict]], target_tp_zero_based: int) -> set[str]:
    out = set()
    for label, rows in by_id.items():
        if any(int(r["tp_zero_based"]) != target_tp_zero_based for r in rows):
            out.add(label)
    return out


def make_backfilled_rows_for_target(
    target_tp_zero_based: int,
    by_id: Dict[str, List[dict]],
    result: ComputationResult,
    settings: Settings,
) -> List[dict]:
    present_target = ids_present_in_target(by_id, target_tp_zero_based)
    present_elsewhere = ids_present_elsewhere(by_id, target_tp_zero_based)

    missing_ids = sorted(
        present_elsewhere - present_target,
        key=lambda x: (int(x) if str(x).isdigit() else x)
    )

    print(f"Target TP {target_tp_zero_based}: found {len(missing_ids)} missing IDs")

    new_rows = []

    for label in missing_ids:
        source_rows = [
            r for r in by_id[label]
            if int(r["tp_zero_based"]) != target_tp_zero_based
        ]

        avg_scaled_dist = average_scaled_distance_from_rows(source_rows)
        if avg_scaled_dist is None:
            print(f"Skipping label {label}: no usable Notes/scaled distance")
            continue

        empty_type = infer_empty_type_from_rows(source_rows)
        dx, dy = average_translation_from_rows(source_rows, result)

        x, y, z_img = estimate_coordinate_from_scaled_distance(
            target_tp_zero_based,
            avg_scaled_dist,
            result,
        )

        axis_0 = z_imagej_to_objectj(z_img, settings.num_channels) - 1
        axis_1 = y + dy
        axis_2 = x + dx

        new_rows.append({
            "axis-0": axis_0,
            "axis-1": axis_1,
            "axis-2": axis_2,
            "label": label,
            "type": empty_type,
            "Notes": avg_scaled_dist,
        })

    return new_rows


def write_augmented_target_csv(
    original_path: Path,
    new_rows: List[dict],
    output_suffix: str,
) -> Path:
    original_rows = read_exported_tp_csv(original_path)

    combined = []
    for row in original_rows:
        # Remove previously existing empty rows from the target export
        if is_empty_type(row.get("type", "")):
            continue

        combined.append({
            "axis-0": row["axis-0"],
            "axis-1": row["axis-1"],
            "axis-2": row["axis-2"],
            "label": row["label"],
            "type": row["type"],
            "Notes": row.get("Notes", ""),
        })

    combined.extend(new_rows)

    out_path = original_path.with_name(
        original_path.stem + output_suffix + original_path.suffix
    )

    header = ["axis-0", "axis-1", "axis-2", "label", "type", "index", "Notes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(combined, start=1):
            writer.writerow([
                row["axis-0"],
                row["axis-1"],
                row["axis-2"],
                row["label"],
                row["type"],
                i,
                row["Notes"],
            ])

    return out_path


# ============================================================
# Main
# ============================================================
def main():
    if not MARKER_BRANCH_DIR.exists():
        raise FileNotFoundError(f"Missing marker branch directory: {MARKER_BRANCH_DIR}")

    if not TRACE_ROOT_DIR.exists():
        raise FileNotFoundError(f"Missing trace root directory: {TRACE_ROOT_DIR}")

    if not EXPORT_DIR.exists():
        raise FileNotFoundError(f"Missing export directory: {EXPORT_DIR}")

    marker_csvs, branch_csvs = build_aligned_inputs(
        MARKER_BRANCH_DIR,
        TRACE_ROOT_DIR,
    )
    n_timepoints = len(marker_csvs)

    settings = Settings(
        animal_id=ANIMAL_ID,
        branch_id=str(BRANCH_ID),
        n_timepoints=n_timepoints,
        branch_csvs=branch_csvs,
        marker_csvs=marker_csvs,
        fiducials_csv=Path(""),
        export_dir=str(EXPORT_DIR),
        scaling_factor=[1, 1, 1],
        num_channels=4,
        inhibitory_shaft="Shaft_SyntdNotScored",
        inhibitory_spine="Spine_SynTdNotScored",
        ojj_tif_key="Image",
        snt_branch_fmt="branch%s",
    )

    print("Recomputing alignment/computation result...")
    result = compute(settings)

    print("Reading exported timepoint CSVs...")
    by_id = collect_rows_by_id(EXPORT_DIR, BRANCH_ID)

    target_path = exported_csv_path(EXPORT_DIR, BRANCH_ID, TARGET_TP_ZERO_BASED)

    print(f"Backfilling missing markers into target timepoint {TARGET_TP_ZERO_BASED}")
    new_rows = make_backfilled_rows_for_target(
        TARGET_TP_ZERO_BASED,
        by_id,
        result,
        settings,
    )

    print(f"Will append {len(new_rows)} generated empty markers")

    out_path = write_augmented_target_csv(
        target_path,
        new_rows,
        OUTPUT_SUFFIX,
    )

    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()

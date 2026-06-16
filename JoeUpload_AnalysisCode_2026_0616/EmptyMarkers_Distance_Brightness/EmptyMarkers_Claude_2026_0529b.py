from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import csv
import re

import numpy as np

from geometry import euc_xy, fit_branch_spline
from io_utils import z_imagej_to_objectj, read_branch_csv


Coord = Tuple[float, float, float]
_IMAGE_RE = re.compile(r"Image(\d+)", re.IGNORECASE)

# ── configuration ────────────────────────────────────────────────────────────
BASE_BRANCH_DIR = Path(
    r"Z:\Joe\AnalysisCode_forSharing\ExampleData\FullyAnalyzed\Analyzed_Data"
    r"\AlignmentAndChecking\AfterCorrections"
).expanduser().resolve()

TRACE_ROOT_DIR = Path(
    r"Z:\Joe\AnalysisCode_forSharing\ExampleData\FullyAnalyzed\Analyzed_Data"
    r"\SNTTrace"
).expanduser().resolve()

ANIMAL_ID             = "SOM056"
OUTPUT_SUFFIX         = "_with_empty"
BRANCH_IDS            = ["1","2","3","4","5","6","7","8","9","10"]
TARGET_TPS_ZERO_BASED = range(6)          # Image0 … Image5
NUM_CHANNELS          = 4
N_SPLINE_POINTS       = 1000


# ── type predicates ──────────────────────────────────────────────────────────

def is_real_synapse_type(t: str) -> bool:
    return str(t).strip().lower() in {
        "shaft_syntdnotscored", "spine_syntdnotscored",
        "spine_syntd", "shaft_syntd",
    }


def is_empty_type(t: str) -> bool:
    return str(t).strip().lower() in {"empty_shaft", "empty_spine"}


def infer_empty_type_from_rows(rows: List[dict]) -> str:
    types = [str(r.get("type", "")).strip().lower() for r in rows]
    return "Empty_spine" if any("spine" in t for t in types) else "Empty_shaft"


def parse_float_or_none(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        return None if s in ("", "NA") else float(s)
    except Exception:
        return None


def normalize_label(label) -> str:
    return str(label).strip()


def extract_image_index(p: Path) -> Optional[int]:
    m = _IMAGE_RE.search(p.name)
    return int(m.group(1)) if m else None


# ── file discovery ───────────────────────────────────────────────────────────

def collect_marker_csvs_by_image(branch_dir: Path) -> dict[int, Path]:
    by_img: dict[int, Path] = {}
    for p in branch_dir.glob("*.csv"):
        if "landmark" in p.name.lower():
            continue
        if not p.name.endswith("snapped_bouton_overlap.csv"):
            continue
        idx = extract_image_index(p)
        if idx is not None:
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
        candidates = [p for p in img_dir.glob("*_Trace_xyzCoordinates.csv")
                      if ".traces" not in p.name]
        if not candidates:
            candidates = list(img_dir.glob("*_Trace_xyzCoordinates.csv"))
        if candidates:
            by_img[idx] = sorted(candidates,
                                  key=lambda p: (len(p.name), p.name))[0]
    return by_img


# ── branch spline ─────────────────────────────────────────────────────────────

def load_branch_spline_for_image(
    trace_csv: Path,
    branch_id: str,
    n_points: int = N_SPLINE_POINTS,
) -> Optional[List[Coord]]:
    df = read_branch_csv(trace_csv, f"branch{branch_id}")
    if df.empty:
        print(f"  [warn] branch{branch_id} absent from {trace_csv.name}")
        return None
    x, y, z = df["x"].values, df["y"].values, df["z"].values
    return fit_branch_spline(x, y, z, n_points=n_points)


# ── arclength utilities ───────────────────────────────────────────────────────

def build_cumulative_arclength(spline: List[Coord]) -> List[float]:
    """
    Precompute cumulative 2-D arclength at every spline vertex.
    cum[0] = 0;  cum[i] = sum of segment lengths from vertex 0 to vertex i.
    """
    cum = [0.0]
    for i in range(len(spline) - 1):
        cum.append(cum[-1] + euc_xy(spline[i], spline[i + 1]))
    return cum


def compute_branch_start_offset(
    target_rows: List[dict],
    spline:      List[Coord],
    cum_arc:     List[float],
) -> float:
    """
    Estimate the arclength from spline index 0 to the StartBranch position
    by self-calibrating against real synapses already annotated in the
    target session.

    Each real synapse with a valid Notes value contributes one estimate:

        offset_i = arc(spline_start → snapped_position_i) − Notes_i

    Because Notes_i is the arclength from StartBranch to synapse i, offset_i
    equals the arclength from the spline start to StartBranch.

    Averaging over all qualifying synapses gives a robust estimate that
    requires no external annotation file.

    Returns the mean offset, or 0.0 when no calibration points are available
    (with a warning).
    """
    offsets: List[float] = []

    for r in target_rows:
        if not is_real_synapse_type(r.get("type", "")):
            continue

        notes = parse_float_or_none(r.get("Notes"))
        if notes is None or notes < 0:
            continue

        # Use branch-snapped coordinates (shaft position) for calibration.
        # Fall back to raw marker coordinates if snapped_axis columns are absent.
        snap_x = parse_float_or_none(r.get("snapped_axis-2"))
        snap_y = parse_float_or_none(r.get("snapped_axis-1"))
        if snap_x is None or snap_y is None:
            snap_x = parse_float_or_none(r.get("axis-2"))
            snap_y = parse_float_or_none(r.get("axis-1"))
        if snap_x is None or snap_y is None:
            continue

        # Nearest spline vertex (2-D distance)
        snap_pt = (snap_x, snap_y, 0.0)
        nearest_idx = int(np.argmin([euc_xy(pt, snap_pt) for pt in spline]))

        # offset = arc(spline_start → synapse) − Notes
        offsets.append(cum_arc[nearest_idx] - notes)

    if not offsets:
        print("  [warn] No calibration synapses found; "
              "arclength will be measured from spline index 0.")
        return 0.0

    mean_off = float(np.mean(offsets))
    std_off  = float(np.std(offsets))
    print(f"  [calibration] {len(offsets)} synapse(s) → "
          f"start_offset = {mean_off:.3f} px  (σ = {std_off:.3f})")
    return mean_off


def estimate_coord_from_arclength(
    target_arclength: float,
    spline:  List[Coord],
    cum_arc: List[float],
) -> Coord:
    """
    Return the spline vertex at which the cumulative 2-D arclength from
    vertex 0 first reaches *target_arclength*.
    """
    if not spline:
        raise ValueError("spline is empty")
    if target_arclength <= 0.0:
        return spline[0]
    for i in range(len(spline) - 1):
        if cum_arc[i + 1] >= target_arclength - 1e-8:
            return spline[i]
    return spline[-1]


# ── Notes / offset helpers ───────────────────────────────────────────────────

def average_notes_from_source_rows(rows: List[dict]) -> Optional[float]:
    """
    Average the pre-computed arclength (Notes field) over source rows that
    carry a real-synapse type and a valid numeric Notes value.
    """
    vals = [
        parse_float_or_none(r.get("Notes"))
        for r in rows
        if is_real_synapse_type(r.get("type", ""))
    ]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def average_spine_offset_from_source_rows(
    rows: List[dict],
) -> Tuple[float, float]:
    """
    Return mean (dx, dy) displacement between the raw spine-head position
    (axis-2, axis-1) and the branch-snapped shaft position (snapped_axis-*)
    across source rows.  Returns (0.0, 0.0) when no valid pairs exist.
    """
    dxs: List[float] = []
    dys: List[float] = []
    for r in rows:
        rx = parse_float_or_none(r.get("axis-2"))
        ry = parse_float_or_none(r.get("axis-1"))
        sx = parse_float_or_none(r.get("snapped_axis-2"))
        sy = parse_float_or_none(r.get("snapped_axis-1"))
        if all(v is not None for v in (rx, ry, sx, sy)):
            dxs.append(rx - sx)
            dys.append(ry - sy)
    if not dxs:
        return 0.0, 0.0
    return float(np.mean(dxs)), float(np.mean(dys))


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def read_exported_tp_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _is_non_synapse_row(row: dict) -> bool:
    """True for Ambiguous / Landmark rows that should not be grouped by label."""
    return str(row.get("type", "")).strip().lower() in {"ambiguous", "landmark"}


def collect_rows_by_id(
    export_dir: Path,
    branch_id:  str,
) -> Dict[str, List[dict]]:
    """
    Read every ``Image*_branch{branch_id}_snapped_bouton_overlap.csv`` under
    *export_dir*, keep only real-synapse rows (non-empty, non-ambiguous), and
    group them by label.  Each row gains a ``tp_zero_based`` key set to the
    image index extracted from the filename.
    """
    files = sorted(export_dir.rglob(
        f"Image*_branch{branch_id}_snapped_bouton_overlap.csv"))
    if not files:
        raise FileNotFoundError(
            f"No snapped-bouton CSVs for branch {branch_id} under {export_dir}")

    print("\n=== Input marker CSVs ===")
    for p in files:
        print(f"  {p}")

    by_id: Dict[str, List[dict]] = defaultdict(list)

    for path in files:
        m = re.search(r"Image(\d+)_branch", path.name, re.IGNORECASE)
        if not m:
            continue
        tp = int(m.group(1))

        for row in read_exported_tp_csv(path):
            row = dict(row)
            row["tp_zero_based"] = tp
            row["_source_csv"]   = str(path)

            if _is_non_synapse_row(row):
                continue
            if is_empty_type(row.get("type", "")):
                continue

            label = normalize_label(row.get("label", ""))
            if label:
                by_id[label].append(row)

    return by_id


def exported_csv_path(
    export_dir: Path, branch_id: str, tp: int
) -> Path:
    direct = (export_dir /
              f"Image{tp}_branch{branch_id}_snapped_bouton_overlap.csv")
    if direct.exists():
        return direct
    matches = list(export_dir.rglob(
        f"Image{tp}_branch{branch_id}_snapped_bouton_overlap.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find Image{tp}_branch{branch_id}_snapped_bouton_overlap.csv"
            f" under {export_dir}")
    return matches[0]


def ids_present_in_target(by_id: Dict[str, List[dict]], tp: int) -> set:
    return {lbl for lbl, rows in by_id.items()
            if any(int(r["tp_zero_based"]) == tp for r in rows)}


def ids_present_elsewhere(by_id: Dict[str, List[dict]], tp: int) -> set:
    return {lbl for lbl, rows in by_id.items()
            if any(int(r["tp_zero_based"]) != tp for r in rows)}


def write_augmented_target_csv(
    original_path: Path,
    new_rows: List[dict],
    output_suffix: str,
) -> Path:
    original_rows = read_exported_tp_csv(original_path)
    header = list(original_rows[0].keys()) if original_rows else []
    for col in ["axis-0","axis-1","axis-2","label","type","index","Notes"]:
        if col not in header:
            header.append(col)

    combined = [dict(r) for r in original_rows
                if not is_empty_type(r.get("type", ""))]
    for nr in new_rows:
        full = {col: "" for col in header}
        full.update(nr)
        combined.append(full)

    out_path = original_path.with_name(
        original_path.stem + output_suffix + original_path.suffix)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for i, row in enumerate(combined, start=1):
            row = dict(row)
            row["index"] = i
            writer.writerow(row)
    return out_path


# ── main backfill function ────────────────────────────────────────────────────

def make_backfilled_rows_for_target(
    target_img_idx:  int,
    by_id:           Dict[str, List[dict]],
    branch_splines:  Dict[int, List[Coord]],
    target_csv_path: Path,
    num_channels:    int = NUM_CHANNELS,
) -> List[dict]:
    """
    Generate Empty_shaft / Empty_spine rows for every synapse label that
    is present as a real synapse in at least one other session but absent
    from *target_img_idx*.

    Key design (Bug 2 fix)
    ----------------------
    The ``Notes`` field in each source CSV row is the pre-computed arclength
    from the session's ``StartBranch`` to that synapse along the splined
    branch.

    To reproduce the same structural position on the TARGET session's branch
    we need to know where ``StartBranch`` falls on the TARGET spline.
    ``compute_branch_start_offset`` derives this automatically:

        start_offset = mean over real synapses in target of
                       [ arc(spline_start → snapped_pos) − Notes ]

    The empty marker is then placed at:

        total_arclength = start_offset + avg_notes_across_sessions
    """
    target_spline = branch_splines.get(target_img_idx)
    if target_spline is None:
        print(f"[skip] No spline available for Image{target_img_idx}.")
        return []

    # Pre-compute cumulative arclength once for efficiency
    cum_arc = build_cumulative_arclength(target_spline)

    # Self-calibrate: find where StartBranch sits on this session's spline
    target_rows  = read_exported_tp_csv(target_csv_path)
    start_offset = compute_branch_start_offset(target_rows, target_spline,
                                               cum_arc)

    present_target    = ids_present_in_target(by_id, target_img_idx)
    present_elsewhere = ids_present_elsewhere(by_id, target_img_idx)
    missing_ids = sorted(
        present_elsewhere - present_target,
        key=lambda lbl: (int(lbl) if str(lbl).isdigit() else lbl),
    )

    print(f"\nImage{target_img_idx}: {len(missing_ids)} label(s) to backfill"
          f" – {missing_ids}")

    new_rows: List[dict] = []

    for label in missing_ids:
        source_rows = [r for r in by_id[label]
                       if int(r["tp_zero_based"]) != target_img_idx]

        # Average pre-computed arclength from StartBranch across source sessions
        avg_notes = average_notes_from_source_rows(source_rows)
        if avg_notes is None:
            print(f"  [skip] label {label!r}: no valid Notes in source rows")
            continue

        empty_type = infer_empty_type_from_rows(source_rows)

        # Spine-head offset (shaft → head displacement)
        dx, dy = (average_spine_offset_from_source_rows(source_rows)
                  if empty_type == "Empty_spine" else (0.0, 0.0))

        # Place the marker at start_offset + avg_notes along the spline
        total_arc = start_offset + avg_notes
        x, y, z   = estimate_coord_from_arclength(total_arc,
                                                   target_spline, cum_arc)

        axis_0 = z_imagej_to_objectj(z, num_channels) - 1
        axis_1 = y + dy
        axis_2 = x + dx

        src_imgs = sorted({int(r["tp_zero_based"]) for r in source_rows})
        print(f"  label {label!r}: avg_notes={avg_notes:.3f} px "
              f"(Images {src_imgs})  type={empty_type}"
              f"  total_arc={total_arc:.3f}"
              f"  → x={axis_2:.2f}, y={axis_1:.2f}, axis-0={axis_0:.2f}")

        new_rows.append({
            "axis-0": axis_0,
            "axis-1": axis_1,
            "axis-2": axis_2,
            "label":  label,
            "type":   empty_type,
            "Notes":  avg_notes,
        })

    return new_rows


# ── per-branch / per-timepoint runner ────────────────────────────────────────

def run_one_branch_one_tp(
    marker_branch_dir:    Path,
    branch_id:            str,
    target_tp_zero_based: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"BRANCH {branch_id}  |  TARGET Image{target_tp_zero_based}")
    print("=" * 80)

    trace_by_img = collect_trace_csv_by_image(TRACE_ROOT_DIR)
    if not trace_by_img:
        print(f"[skip] No trace CSVs under {TRACE_ROOT_DIR}.")
        return

    # Load branch splines for every available imaging session
    branch_splines: Dict[int, List[Coord]] = {}
    for img_idx, trace_csv in sorted(trace_by_img.items()):
        spline = load_branch_spline_for_image(trace_csv, branch_id)
        if spline is not None:
            branch_splines[img_idx] = spline
            print(f"  Image{img_idx}: {len(spline)}-pt spline  "
                  f"({trace_csv.name})")

    # Collect all real-synapse rows from the input marker CSVs
    by_id = collect_rows_by_id(marker_branch_dir, branch_id)

    # Locate the target session's input CSV
    target_path = exported_csv_path(
        marker_branch_dir, branch_id, target_tp_zero_based)
    print(f"\nTarget CSV: {target_path.name}")

    # Generate backfill rows
    new_rows = make_backfilled_rows_for_target(
        target_img_idx  = target_tp_zero_based,
        by_id           = by_id,
        branch_splines  = branch_splines,
        target_csv_path = target_path,
        num_channels    = NUM_CHANNELS,
    )
    print(f"\nAppending {len(new_rows)} empty marker(s).")

    out_path = write_augmented_target_csv(target_path, new_rows, OUTPUT_SUFFIX)
    print(f"Wrote: {out_path}")


# ── batch entry point ─────────────────────────────────────────────────────────

def main() -> None:
    for branch_id in BRANCH_IDS:
        marker_branch_dir = BASE_BRANCH_DIR / f"Corrected_branch{branch_id}"
        if not marker_branch_dir.exists():
            print(f"[skip] Missing dir: {marker_branch_dir}")
            continue
        for target_tp in TARGET_TPS_ZERO_BASED:
            try:
                run_one_branch_one_tp(
                    marker_branch_dir    = marker_branch_dir,
                    branch_id            = branch_id,
                    target_tp_zero_based = target_tp,
                )
            except Exception as exc:
                print(f"[ERROR] Branch {branch_id}, Image{target_tp}: {exc}")


if __name__ == "__main__":
    main()
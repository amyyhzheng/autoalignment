# neuron_align/main.py
from pathlib import Path
import re
import csv
import numpy as np
import matplotlib

# Save-only backend (no GUI windows)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Settings
from computation import compute
from clustering import separate_shaft_spine, choose_best_clustering, export_grouping_csv
from mapping import export_all

from viz_helpers import (
    plot_cluster_bins,
    plot_markers_along_branch_with_ids,
    plot_branch_with_cluster_ids,
)

DO_PLOTS = True

_IMAGE_RE = re.compile(r"Image(\d+)", re.IGNORECASE)


def extract_image_index(p: Path) -> int | None:
    """
    Extract Image# from a file/dir name. Returns None if not found.
    Examples:
      SOM026_Image0_foo.csv -> 0
      Image5 -> 5
    """
    m = _IMAGE_RE.search(p.name)
    return int(m.group(1)) if m else None


def save_current_fig(outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def collect_marker_csvs_by_image(branch_dir: Path) -> dict[int, Path]:
    """
    Collect marker CSVs in branch_dir keyed by Image#.
    If multiple CSVs map to the same Image#, keep the shortest filename (usually canonical).
    """
    by_img: dict[int, Path] = {}
    for p in branch_dir.glob("*.csv"):
        idx = extract_image_index(p)
        if idx is None:
            continue
        if idx not in by_img or len(p.name) < len(by_img[idx].name):
            by_img[idx] = p

    # Changed to only bouton overlap csv
    # for p in branch_dir.glob("*bouton*.csv"):
    #     idx = extract_image_index(p)
    #     if idx is None:
    #         continue

    #     if idx not in by_img or len(p.name) < len(by_img[idx].name):
    #         by_img[idx] = p

    return by_img



def collect_trace_csv_by_image(snt_root: Path) -> dict[int, Path]:
    """
    Collect trace CSVs under SNTTrace/Image*/ keyed by Image#.
    If there are multiple candidates, prefer ones NOT containing '.traces' in the filename.
    """
    by_img: dict[int, Path] = {}

    for img_dir in snt_root.glob("Image*"):
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


def build_aligned_inputs(parent_dir: Path, branch_dir: Path):
    """
    Build marker_csvs list + branch_csvs dict using the intersection of Image indices
    present in BOTH:
      - parent_dir/SNTTrace/Image{idx}/*_Trace_xyzCoordinates.csv
      - branch_dir/*.csv (marker files)
    This prevents crashes when a branch folder has extra marker CSVs (e.g., Image6)
    but SNTTrace only has Image0..Image5.
    """
    snt_root = parent_dir / "SNTTrace"
    marker_by_img = collect_marker_csvs_by_image(branch_dir)
    trace_by_img = collect_trace_csv_by_image(snt_root)

    common_imgs = sorted(set(marker_by_img) & set(trace_by_img))
    if not common_imgs:
        raise FileNotFoundError(
            f"No overlapping Image# between markers in {branch_dir} and traces in {snt_root}.\n"
            f"Markers have: {sorted(marker_by_img.keys())}\n"
            f"Traces have: {sorted(trace_by_img.keys())}"
        )

    print(f"[build_aligned_inputs] Using Image indices: {common_imgs}")

    # marker list ordered by image index
    marker_csvs = [marker_by_img[i] for i in common_imgs]

    # branch_csvs: Timepoint 1..N mapped to the chosen trace csv for each image index
    branch_csvs: dict[str, Path] = {}
    for tp_idx, img_idx in enumerate(common_imgs):
        tp_name = f"Timepoint {tp_idx+1}"
        branch_csvs[tp_name] = trace_by_img[img_idx]
        print(f"[build_branch_csvs] {tp_name}: {trace_by_img[img_idx].name}")

    return marker_csvs, branch_csvs


def run_one_branch(parent_dir: Path, branch_dir: Path, output_root: Path) -> None:
    # branch_dir looks like: parent/PunctaScoring/branch3
    branch_id = branch_dir.name.replace("branch", "").strip()

    marker_csvs, branch_csvs = build_aligned_inputs(parent_dir, branch_dir)
    n_timepoints = len(marker_csvs)

    # per-run output structure
    run_name = f"{parent_dir.name}_branch{branch_id}"
    run_out = output_root / run_name
    plots_out = run_out / "plots"
    run_out.mkdir(parents=True, exist_ok=True)
    plots_out.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        animal_id="SOM022",
        branch_id=str(branch_id),
        n_timepoints=n_timepoints,
        branch_csvs=branch_csvs,
        marker_csvs=marker_csvs,
        fiducials_csv=Path(""),
        export_dir=str(run_out),
        scaling_factor=[1, 1, 1],
        num_channels=4,
        inhibitory_shaft="Shaft_SyntdNotScored",
        inhibitory_spine="Spine_SyntdNotScored",
        ojj_tif_key="Image",
        snt_branch_fmt="branch%s",
    )

    # ---------------------------
    # computation
    # ---------------------------
    res = compute(settings)

    # ---------------------------
    # clustering
    shaft_markers, spine_markers = separate_shaft_spine(settings, res)

    shaft_d = [[m["distance"] for m in tp] for tp in shaft_markers]
    spine_d = [[m["distance"] for m in tp] for tp in spine_markers]

    shaft_clusters = []
    spine_clusters = []

    next_cluster_id = 0

    if any(shaft_d):
        shaft_grouping = choose_best_clustering(shaft_d, res.final_marker_distance)
        if shaft_grouping:
            shaft_clusters = export_grouping_csv(
                shaft_grouping,
                str(run_out / "inhibitory_shaft_grouping.csv"),
                start_id=next_cluster_id,
                group_type=settings.inhibitory_shaft,
                metadata_out=str(run_out / "inhibitory_shaft_grouping_metadata.csv"),
            )
            next_cluster_id += len(shaft_clusters)

    if any(spine_d):
        spine_grouping = choose_best_clustering(spine_d, res.final_marker_distance)
        if spine_grouping:
            spine_clusters = export_grouping_csv(
                spine_grouping,
                str(run_out / "inhibitory_spine_grouping.csv"),
                start_id=next_cluster_id,
                group_type=settings.inhibitory_spine,
                metadata_out=str(run_out / "inhibitory_spine_grouping_metadata.csv"),
            )
    # ---------------------------
    # plots -> save to disk
    # ---------------------------
    def export_markers_for_branch_plot(res, cluster_list, out_csv):
        '''
            probably move this function to viz_helpers just export so can plot later

        '''
        import pandas as pd

        n_tp = len(res.final_marker_distance)

        # --- build mapping exactly like the plot ---
        marker_to_cluster = [dict() for _ in range(n_tp)]
        if cluster_list:
            for cid, cluster in enumerate(cluster_list, start=1):
                for (tp, idx, _pos) in cluster:
                    if idx == "NA":
                        continue
                    marker_to_cluster[tp][idx] = cid

        # --- export rows ---
        rows = []

        for tp in range(n_tp):
            xs = res.final_marker_distance[tp]

            for m_idx, x in enumerate(xs):
                rows.append({
                    "timepoint": tp,
                    "marker_index": m_idx,
                    "distance_scaled": x,
                    "cluster_id": marker_to_cluster[tp].get(m_idx, None),
                })

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return df
    def export_landmarks_for_branch_plot(res, out_csv):
        import pandas as pd

        rows = []

        if getattr(res, "cumdist_scaled", None):
            if res.cumdist_scaled and len(res.cumdist_scaled[0]) >= 2:
                boundaries = res.cumdist_scaled[0]
                internal_bounds = boundaries[1:-1]  # skip 0 and total length

                for i, x in enumerate(internal_bounds, start=1):
                    rows.append({
                        "landmark_id": i,
                        "distance_scaled": x,
                    })

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return df
        
    if DO_PLOTS:
        if shaft_clusters:
            export_markers_for_branch_plot(
                res,
                shaft_clusters,
                plots_out / "shaft_markers_along_branch.csv"
            )
            export_landmarks_for_branch_plot(
                res,
                plots_out / "shaft_landmarks.csv"
            )

            plot_markers_along_branch_with_ids(
                res, shaft_clusters, title="Markers along branch (scaled) — shaft clusters"
            )

        if spine_clusters:
            export_markers_for_branch_plot(
                res,
                spine_clusters,
                plots_out / "spine_markers_along_branch.csv"
            )
            export_landmarks_for_branch_plot(
                res,
                plots_out / "spine_landmarks.csv"
            )

            plot_markers_along_branch_with_ids(
                res, spine_clusters, title="Markers along branch (scaled) — spine clusters"
            )
        for tp_to_show in range(settings.n_timepoints):
            if shaft_clusters:
                plot_branch_with_cluster_ids(
                    res, shaft_clusters, tp=tp_to_show, text_fontsize=8, text_dxy=(0.2, 0.2)
                )
                save_current_fig(plots_out / f"branch_with_shaft_cluster_ids_tp{tp_to_show+1}.png")

            if spine_clusters:
                plot_branch_with_cluster_ids(
                    res, spine_clusters, tp=tp_to_show, text_fontsize=8, text_dxy=(0.2, 0.2)
                )
                save_current_fig(plots_out / f"branch_with_spine_cluster_ids_tp{tp_to_show+1}.png")

    # ---------------------------
    # mapping + exports
    # ---------------------------
    out_csv = export_all(settings, res, shaft_clusters, spine_clusters, settings.export_dir)
    print(f"[{branch_dir.name}] Finished pipeline → {out_csv}")
    print(f"[{branch_dir.name}] Outputs in → {run_out}")


def main():
    parent_dir = Path(
        "/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode"
    ).expanduser().resolve()

    puncta_root = parent_dir / "PunctaScoring"
    output_root = parent_dir / "AutoalignmentOutputs2"

    branch_dirs = sorted([p for p in puncta_root.glob("branch*") if p.is_dir()])
    if not branch_dirs:
        raise FileNotFoundError(f"No branch folders found under {puncta_root}")

    for bd in branch_dirs:
        run_one_branch(parent_dir=parent_dir, branch_dir=bd, output_root=output_root)


if __name__ == "__main__":
    main()
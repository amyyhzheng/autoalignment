# scripts/run_pipeline.py
from pathlib import Path

# core pipeline
from config import Settings
from computation import compute
from clustering import separate_shaft_spine, choose_best_clustering, export_grouping_csv
from mapping import export_all

# quick 2D plots
from viz_helpers import (
    plot_branch_and_fiducials,
    plot_markers_along_branch,
    plot_segment_scaling,
    plot_fiducial_index_diagnostics,
    plot_cluster_bins,
)

DO_PLOTS = True  # set False to skip plotting

def main():
    # ---------------------------
    # 1) configure your run here
    # ---------------------------
    settings = Settings(
        animal_id="SOM022",
        branch_id="2",
        n_timepoints=6,
        branch_csvs={
            "Timepoint 1": Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image1/SOM022Image1FullTrace_withbrancheslabeled_xyzCoordinates.csv'),
            "Timepoint 2": Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image2/SOM022_Image2_fulltrace_withbrancheslabeled_xyzCoordinates.csv'),
            "Timepoint 3": Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image3/SOM022Image3_fulltrace_withbrancheslabeled_xyzCoordinates.csv'),
            "Timepoint 4": Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image4/SOM022Image4_fulltrace_withbrancheslabeled_xyzCoordinates.csv'),
            "Timepoint 5": Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image5/SOM022Image5_fulltrace_withbrancheslabeled_xyzCoordinates.csv'),
            "Timepoint 6": Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/SNT_Tracing/Image6/SOM022Image6fulltrace_withbrancheslabeled_xyzCoordinates.csv'),
        },
        marker_csvs=[
            Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/SOM022_b2_Image1.csv'),
            Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/SOM022_b2_Image2.csv'),
            Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/SOM022_b2_Image3.csv'),
            Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/SOM022_b2_Image4.csv'),
            Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/SOM022_b2_Image5.csv'),
            Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/beforeAlignment/SOM022_b2_Image6.csv'),
        ],
        fiducials_csv=Path('/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/Fiducials/CombinedResults.csv'),
        export_dir=Path("exports"),
        scaling_factor=[0.25, 0.25, 1],
        num_channels=4,
        inhibitory_shaft="InhibitoryShaft",
        inhibitory_spine="spinewithInhsynapse",
        ojj_tif_key="Image",
        snt_branch_fmt="b%s",
    )

    # ---------------------------
    # 2) computation
    # ---------------------------
    res = compute(settings)

    # quick sanity plots (pre-clustering)
    if DO_PLOTS:
        # pick a TP to inspect (0-based; 0 == Timepoint 1)
        tp_to_show = 0
        plot_branch_and_fiducials(res, tp=tp_to_show, show_markers=True)
        plot_markers_along_branch(res, title="Markers along branch (scaled) across TPs")
        plot_segment_scaling(res, tp=tp_to_show)
        plot_fiducial_index_diagnostics(res, tp=tp_to_show)

    # ---------------------------
    # 3) clustering
    # ---------------------------
    shaft_d, spine_d = separate_shaft_spine(settings, res)

    shaft_clusters = spine_clusters = None

    if any(shaft_d):
        shaft_grouping = choose_best_clustering(shaft_d, res.final_marker_distance)
        if shaft_grouping:
            shaft_clusters = export_grouping_csv(
                shaft_grouping,
                settings.export_dir / "autoAlignment" / f"{settings.animal_id}_b{settings.branch_id}_inhibitoryshaft_grouping.csv",
            )

    if any(spine_d):
        spine_grouping = choose_best_clustering(spine_d, res.final_marker_distance)
        if spine_grouping:
            spine_clusters = export_grouping_csv(
                spine_grouping,
                settings.export_dir / "autoAlignment" / f"{settings.animal_id}_b{settings.branch_id}_inhibitoryspine_grouping.csv",
            )

    # visualize cluster coverage (one row per cluster; columns=TPs)
    if DO_PLOTS:
        if shaft_clusters:
            plot_cluster_bins(shaft_clusters, title="Shaft clusters (TP presence)")
        if spine_clusters:
            plot_cluster_bins(spine_clusters, title="Spine clusters (TP presence)")

    # ---------------------------
    # 4) mapping + exports
    # ---------------------------
    out_csv = export_all(settings, res, shaft_clusters, spine_clusters, settings.export_dir)
    print(f"Finished pipeline → {out_csv}")

if __name__ == "__main__":
    main()

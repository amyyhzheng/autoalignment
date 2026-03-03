# scripts/run_pipeline.py
from pathlib import Path

# core pipeline
from config import Settings
from computation import compute
from clustering import separate_shaft_spine, choose_best_clustering, export_grouping_csv
from mapping import export_all
from geometry import plot_branch_and_synapses

# quick 2D plots
from viz_helpers import (
    plot_branch_and_fiducials,
    plot_markers_along_branch,
    plot_segment_scaling,
    plot_fiducial_index_diagnostics,
    plot_cluster_bins,
    plot_markers_along_branch_with_ids, 
    plot_branch_with_cluster_ids, 
)

DO_PLOTS = True  # set False to skip plotting

def main():
    # ---------------------------
    # 1) configure your run here
    # ---------------------------
    settings = Settings(
        animal_id="SOM022",
        branch_id="3",
        n_timepoints=5,
        branch_csvs={
            "Timepoint 1": Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/SNTTrace/Image1/SOM026_Image1_Trace_xyzCoordinates.csv'),
            "Timepoint 2": Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/SNTTrace/Image2/SOM026_Image2_Trace_xyzCoordinates.csv'),
            "Timepoint 3": Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/SNTTrace/Image3/SOM026_Image3_Trace_xyzCoordinates.csv'),
            "Timepoint 4": Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/SNTTrace/Image4/SOM026_Image4_Trace_xyzCoordinates.csv'),
            "Timepoint 5": Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/SNTTrace/Image5/SOM026_Image5_Trace_xyzCoordinates.csv')
        },
        marker_csvs=[
            Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/PunctaScoring/branch3/SOM026_Image1_branch3.csv'),
            Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/PunctaScoring/branch3/SOM026_Image2_branch3.csv'),
            Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/PunctaScoring/branch3/SOM026_Image3_branch3.csv'),
            Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/PunctaScoring/branch3/SOM026_Image4_branch3.csv'),
            Path('/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/PunctaScoring/branch3/SOM026_Image5_branch3.csv'),
        ],
        fiducials_csv=Path(''),
        export_dir='/Users/amyzheng/Desktop/autoalignment-main 6/neuron_align/what',
        #Joe scaling is 1,1,4 but the other scalin g is 0.250.25 1
        scaling_factor=[1, 1, 1],
        num_channels=4,
        inhibitory_shaft="Shaft_SyntdNotScored",
        inhibitory_spine="Spine_SyntdNotScored",
        ojj_tif_key="Image",
        snt_branch_fmt="branch%s",
    )

    # ---------------------------
    # 2) computation
    # ---------------------------
    res = compute(settings)

    # Plot branch and raw synapse coordinates for a timepoint before clustering
    tp = 0  # Change this to visualize other timepoints
    branch_points = res.raw_branch[tp]

    synapse_points = [coord for _type, coord in res.raw_markers[tp]]
    for i in range(200):
        print('hi')
    print(f'{synapse_points}synapse_points')
    plot_branch_and_synapses(branch_points, synapse_points)


    # quick sanity plots (pre-clustering)
    # if DO_PLOTS:
    #     # pick a TP to inspect (0-based; 0 == Timepoint 1)
    #     for tp_to_show in range(0, settings.n_timepoints):
    #         plot_branch_and_fiducials(    from neuron_align.ge    import matplotlib.pyplot as plt



    #         # plot_markers_along_branch_with_ids()
    #         plot_markers_along_branch(res, title="Markers along branch (scaled) across TPs")
    #         plot_segment_scaling(res, tp=tp_to_show)
    #         plot_fiducial_index_diagnostics(res, tp=tp_to_show)

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
                '/Users/amyzheng/Desktop/autoalignment-main 6/neuron_align/what/inhibitoryshaft_grouping.csv',
            )
    if any(spine_d):
        spine_grouping = choose_best_clustering(spine_d, res.final_marker_distance)
        if spine_grouping:
            spine_clusters = export_grouping_csv(
                spine_grouping,
                '/Users/amyzheng/Desktop/autoalignment-main 6/neuron_align/what/inhibitoryshaft_grouping.csv',
            )


    if DO_PLOTS:
        if shaft_clusters:
            plot_cluster_bins(shaft_clusters, title="Shaft clusters (TP presence)")
            plot_markers_along_branch_with_ids(
                res,
                shaft_clusters,
                title="Markers along branch (scaled) — shaft clusters"
            )

        if spine_clusters:
            plot_cluster_bins(spine_clusters, title="Spine clusters (TP presence)")
            plot_markers_along_branch_with_ids(
                res,
                spine_clusters,
                title="Markers along branch (scaled) — spine clusters"
            )

    if DO_PLOTS:
        for tp_to_show in range(0, 6): # Timepoint 1
            if shaft_clusters:
                plot_branch_with_cluster_ids(
                    res,
                    shaft_clusters,
                    tp=tp_to_show,
                    text_fontsize=8,
                    text_dxy=(0.2, 0.2),  # tweak if labels overlap
                )

            if spine_clusters:
                plot_branch_with_cluster_ids(
                    res,
                    spine_clusters,
                    tp=tp_to_show,
                    text_fontsize=8,
                    text_dxy=(0.2, 0.2),
                )

    # from viz_helpers import plot_all_branches_with_cluster_ids

    # if DO_PLOTS:
    #     if shaft_clusters:
    #         plot_all_branches_with_cluster_ids(
    #             res,
    #             shaft_clusters,
    #             text_fontsize=9,
    #             text_dxy=(0.15, 0.15),
    #         )

    #     if spine_clusters:
    #         plot_all_branches_with_cluster_ids(
    #             res,
    #             spine_clusters,
    #             text_fontsize=9,
    #             text_dxy=(0.15, 0.15),
    #     )

    # # ---------------------------
    # 4) mapping + exports
    # ---------------------------
    out_csv = export_all(settings, res, shaft_clusters, spine_clusters, settings.export_dir)
    print(f"Finished pipeline → {out_csv}")

if __name__ == "__main__":
    main()

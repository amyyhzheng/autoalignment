from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def plot_markers_along_branch_from_napari_notes(
    napari_dir,
    branch_number,
    landmarks_csv_path=None,
    title="Markers along branch from napari Notes",
    text_fontsize=7,
    text_dy=0.15,
    show_unclustered=True,
    show_landmarks=True,
    label_real_only=False,
):

    print("\n========== DEBUG START ==========\n")

    print("Napari dir:", napari_dir)
    print("Branch number:", branch_number)

    napari_dir = Path(napari_dir)
    branch_number = str(branch_number)

    pat = re.compile(
        rf".*Image(\d+)_branch{re.escape(branch_number)}.*\.csv$",
        re.IGNORECASE,
    )

    rows = []

    print("\n--- Scanning napari directory recursively ---\n")

    all_csvs = sorted(napari_dir.rglob("*.csv"))
    print(f"Total CSV files found under napari_dir: {len(all_csvs)}")

    for p in all_csvs:
        print("Found CSV:", p)

    print("\n--- Matching branch/timepoint pattern ---\n")

    for p in all_csvs:
        # skip hidden / AppleDouble files
        if p.name.startswith(".") or p.name.startswith("._"):
            print("Skipping hidden file:", p.name)
            continue

        name_lower = p.name.lower()

        # keep only bouton files if that is still what you want
        if "bouton" not in name_lower:
            print("Skipping non-bouton file:", p.name)
            continue

        if "snapped_bouton_overlap" not in name_lower:
            print("Skipping non-snapped bouton file:", p.name)
            continue

        m = pat.match(p.name)
        print("Checking file:", p.name)
        print("  matched regex?", bool(m))

        if not m:
            continue

        tp = int(m.group(1))
        print("  extracted timepoint:", tp)

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  skipped: could not read CSV ({e})")
            continue

        print("  columns:", list(df.columns))
        print("  rows:", len(df))

        required_cols = {"label", "type", "Notes"}
        if not required_cols.issubset(df.columns):
            print("  skipped: missing one of label/type/Notes")
            continue

        tmp = df.copy()
        tmp = tmp[tmp["label"].notna()].copy()

        # remove ambiguous landmarks if you do not want them as clustered markers
        if not show_unclustered:
            tmp = tmp[tmp["label"].notna()].copy()

        added = 0
        for _, row in tmp.iterrows():
            label = row["label"]
            marker_type = row["type"]
            pos = row["Notes"]

            # skip empty Notes
            if pd.isna(pos) or str(pos).strip() == "":
                continue

            try:
                pos = float(pos)
            except Exception:
                continue

            # try to coerce label to numeric cluster id if possible
            cluster_id = None
            try:
                cluster_id = int(float(label))
            except Exception:
                cluster_id = label

            rows.append({
                "timepoint": tp,
                "cluster_id": cluster_id,
                "distance_scaled": pos,
                "type": marker_type,
            })
            added += 1

        print(f"  added {added} plotting rows")

    markers = pd.DataFrame(rows)

    if markers.empty:
        print("No marker rows to plot.")
        return

    markers = markers.sort_values(
        ["timepoint", "distance_scaled", "cluster_id"]
    ).reset_index(drop=True)

    print("\n--- BUILT MARKER TABLE FROM NAPARI ---")
    print(markers.head(20))
    print("\nType counts:\n", markers["type"].value_counts(dropna=False))


    type_to_color = {
        "Shaft_Geph+Bsn_NoSynTd": "tab:blue",
        "Empty_shaft": "lightblue",
        "Spine_Geph+Bsn_NoSynTd": "tab:red",
        "Empty_spine": "lightcoral",
        "Ambiguous": "gray",
        "Shaft_SyntdNotScored": "tab:cyan",
        "Spine_SynTdNotScored": "tab:pink",
        "Shaft_SynTd": "navy",
        "Spine_SynTd": "darkred",
        None: "black",
    }

    timepoints = sorted(markers["timepoint"].dropna().astype(int).unique())

    plt.figure(figsize=(8.0, 5.0))

    unique_types = list(markers["type"].drop_duplicates())
    for t in unique_types:
        if pd.isna(t):
            sub = markers[markers["type"].isna()].copy()
            label = "unmapped"
            color = type_to_color.get(None, "black")
        else:
            sub = markers[markers["type"] == t].copy()
            label = str(t)
            color = type_to_color.get(t, "black")

        if sub.empty:
            continue

        xs = sub["distance_scaled"].tolist()
        ys = sub["timepoint"].astype(int).tolist()

        plt.scatter(xs, ys, s=16, color=color, label=label)

        for _, row in sub.iterrows():
            cid = row["cluster_id"]
            if pd.isna(cid):
                continue

            if label_real_only and isinstance(row["type"], str) and "empty" in row["type"].lower():
                continue

            cid_str = str(cid)
            plt.text(
                row["distance_scaled"],
                int(row["timepoint"]) + text_dy,
                cid_str,
                fontsize=text_fontsize,
                ha="center",
                va="bottom",
                color=color,
            )

    # --------------------------------------------------
    # landmark lines
    # --------------------------------------------------
    if show_landmarks and landmarks_csv_path is not None:
        landmarks = pd.read_csv(landmarks_csv_path)
        if not landmarks.empty:
            if "distance_scaled" not in landmarks.columns:
                raise ValueError("Landmark CSV must contain 'distance_scaled' column.")

            first = True
            for x in landmarks["distance_scaled"].tolist():
                plt.axvline(
                    x,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                    color="gray",
                    label="landmarks" if first else None,
                )
                first = False

    plt.yticks(timepoints, [f"TP{tp+1}" for tp in timepoints])
    plt.xlabel("Scaled cumulative distance (µm)")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


plot_markers_along_branch_from_napari_notes(
    napari_dir='/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM022_DOB073020LT/Analysis/Analysis_withAmyCode/AutoalignmentOutputs2/Analysis_withAmyCode_branch1',
    landmarks_csv_path="/Volumes/nedividata/Joe/2p_data/SOM/ThirdRound/SOM026_DOB081520_RV/Analysis/Analysis_withAmyCode/AutoalignmentOutputs/Analysis_withAmyCode_branch2/plots/shaft_landmarks.csv",
    branch_number=2,
    title="Markers along branch from napari Notes — spine clusters",
)

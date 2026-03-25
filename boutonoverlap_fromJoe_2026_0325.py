from pathlib import Path
import re
import pandas as pd
import numpy as np
import tifffile as tif

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.draw import disk


images_root = Path(r'Z:\Joe\2p_data\SOM\ThirdRound\SOM026_DOB081520_RV\Analysis\Analysis_withAmyCode\SNTTrace')
puncta_root = Path(r'Z:\Joe\2p_data\SOM\ThirdRound\SOM026_DOB081520_RV\Analysis\Analysis_withAmyCode\PunctaScoring')

CHANNEL_INDEX = 0          # channel used for peak finding / snapping
BOUTON_CHANNEL_INDEX = 2   # channel used for bouton overlap

# CSV coordinate columns
COORD_COLS = ["axis-0", "axis-1", "axis-2"]

# optional label columns
LABEL_COL_CANDIDATES = ["label"]

# folder names like Image0, Image1, ...
image_dir_pat = re.compile(r"^Image(\d+)$", re.IGNORECASE)

# filenames like something_Image0_branch1.csv
csv_img_pat = re.compile(r"_Image(\d+)_branch\d+", re.IGNORECASE)


def extract_image_number_from_dir(d: Path):
    m = image_dir_pat.match(d.name)
    return int(m.group(1)) if m else None


def find_one_tif(image_dir: Path) -> Path:
    tifs = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))
    if not tifs:
        raise FileNotFoundError(f"No .tif/.tiff found in {image_dir}")
    if len(tifs) > 1:
        print(f"[WARN] Multiple tifs in {image_dir}; using {tifs[0].name}")
    return tifs[0]


def load_full_image(tif_path: Path) -> np.ndarray:
    img = tif.imread(str(tif_path))
    if img.ndim != 4:
        raise ValueError(f"Expected 4D tif (Z,C,Y,X). Got shape {img.shape} for {tif_path}")
    return img


def load_image_channel(img: np.ndarray, channel_index: int = 0) -> np.ndarray:
    if channel_index < 0 or channel_index >= img.shape[1]:
        raise IndexError(f"channel_index={channel_index} out of range for shape {img.shape}")
    return img[:, channel_index, :, :].astype(np.float32)  # (Z,Y,X)


def load_branch_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} missing coord columns {missing}. Has columns: {list(df.columns)}")

    pts = df[COORD_COLS].to_numpy(np.float32)

    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    labels = df[label_col].to_numpy() if label_col is not None else None

    return df, pts, labels


def find_branch_csvs_for_image(img_num: int, puncta_root: Path) -> list[Path]:
    hits = []
    for p in puncta_root.rglob("*.csv"):
        m = csv_img_pat.search(p.name)
        if m and int(m.group(1)) == img_num:
            hits.append(p)
    return sorted(hits)


def run_one_branch(
    img_zycx: np.ndarray,
    df: pd.DataFrame,
    pts_zyx: np.ndarray,
    labels,
    *,
    img_num: int,
    csv_path: Path,
    channel_index: int = CHANNEL_INDEX,
    bouton_channel_index: int = BOUTON_CHANNEL_INDEX,
    z_scale: int = 4,
    z_pad: int = 1,
    #default was 2, Joe changed z_tol to 0 so it'll stay with the same Z plane as the marker
    z_tol_planes: int = 0,
    #default was 100, Joe changed max_snap to 3 so it'll stay within 3 xy pixels of the marker
    max_snap_px: float = 3,
    disk_r: int = 2,
    otsu_offset: float = 30,
):
    other_channel = load_image_channel(img_zycx, channel_index)
    boutons = load_image_channel(img_zycx, bouton_channel_index)

    pts = pts_zyx.copy().astype(np.float32)

    Z, H, W = other_channel.shape

    # 1) assign each point to nearest image z slice
    z_img_idx = np.clip(
        np.rint(pts[:, 0] / z_scale).astype(int),
        0,
        Z - 1
    )

    z_planes = np.unique(z_img_idx)

    if len(z_planes) == 0:
        raise ValueError(f"No points map to valid image z planes for {csv_path}")

    # optional z padding
    if z_pad > 0:
        z_planes = np.unique(
            np.clip(
                np.concatenate([z_planes + dz for dz in range(-z_pad, z_pad + 1)]),
                0,
                Z - 1
            )
        )

    # 2) restricted volume
    vol = other_channel[z_planes].astype(np.float32)

    # 3) smooth
    vol_s = gaussian_filter(vol, sigma=(0.0, 1.0, 1.0))

    # 4) peaks
    coords3d = peak_local_max(
        vol_s,
        min_distance=1,
        threshold_abs=0,
        exclude_border=False
    )

    print(f"3D peaks (restricted planes) for {csv_path.name}: {len(coords3d)}")
    if len(coords3d) == 0:
        raise ValueError(f"No 3D peaks found for {csv_path}")

    # 5) map peak z back to scaled point coordinates
    coords3d_global = coords3d.copy()
    coords3d_global[:, 0] = z_planes[coords3d[:, 0]] * z_scale

    # assign each point to nearest scaled plane
    z_scaled = np.rint(pts[:, 0] / z_scale).astype(int) * z_scale

    pts_snapped_all = pts.copy()
    z_tol_scaled = z_tol_planes * z_scale

    for z in np.unique(z_scaled):
        idx_p = np.where(z_scaled == z)[0]

        # allow peaks within ± z_tol_planes
        mask_peaks = np.abs(coords3d_global[:, 0] - z) <= z_tol_scaled

        if not np.any(mask_peaks):
            print(f"z={z:4d} | peaks=0 in ±{z_tol_planes} planes | snapped 0 / {len(idx_p)} (kept original)")
            continue

        peaks = coords3d_global[mask_peaks]              # (P,3) z,y,x
        peaks_xy = peaks[:, 1:3].astype(np.float32)     # y,x
        query_xy = pts[idx_p][:, 1:3].astype(np.float32)

        tree = cKDTree(peaks_xy)
        dist, nn = tree.query(query_xy, k=1)

        snapped_xy = peaks_xy[nn].copy()
        too_far = dist > max_snap_px
        snapped_xy[too_far] = query_xy[too_far]

        pts_snapped_all[idx_p, 1:3] = snapped_xy

        print(
            f"z={z:4d} | peaks={mask_peaks.sum():4d} (±{z_tol_planes} planes) | "
            f"snapped {(~too_far).sum():4d} / {len(idx_p):4d}"
        )

    # bouton overlap
    sn = pts_snapped_all.astype(np.float32)

    z_idx = np.clip(np.rint(sn[:, 0] / z_scale).astype(int), 0, Z - 1)
    yy = np.clip(np.rint(sn[:, 1]).astype(int), 0, H - 1)
    xx = np.clip(np.rint(sn[:, 2]).astype(int), 0, W - 1)

    disk_mask = np.zeros((Z, H, W), dtype=np.uint8)
    for zi, y, x in zip(z_idx, yy, xx):
        rr, cc = disk((y, x), disk_r, shape=(H, W))
        disk_mask[zi, rr, cc] = 1

    #this said t = threshold_otsu(boutons) + otsu_offset. Joe changed to t = 10 for constant threshold
    t = 10
    boutons_mask = boutons > t
    coloc_mask = boutons_mask & disk_mask.astype(bool)

    print(f"boutons threshold = {t:.2f} | boutons_mask fraction = {boutons_mask.mean():.6f}")
    print(f"coloc voxels = {coloc_mask.sum()}")

    overlap_counts = np.zeros(len(df), dtype=np.int32)
    disk_areas = np.zeros(len(df), dtype=np.int32)

    for i, (zi, y, x) in enumerate(zip(z_idx, yy, xx)):
        rr, cc = disk((y, x), disk_r, shape=(H, W))
        disk_areas[i] = len(rr)
        overlap_counts[i] = int(boutons_mask[zi, rr, cc].sum())


    out_df = df.copy()
    out_df["snapped_axis-0"] = pts_snapped_all[:, 0]
    out_df["snapped_axis-1"] = pts_snapped_all[:, 1]
    out_df["snapped_axis-2"] = pts_snapped_all[:, 2]
    out_df[f"disk_area_px_r{disk_r}"] = disk_areas
    out_df[f"bouton_overlap_px_r{disk_r}"] = overlap_counts
    out_df[f"bouton_overlap_frac_r{disk_r}"] = overlap_counts / np.clip(disk_areas, 1, None)
    
    type_col = "type"
    #this previously said overlap_col = f"bouton_overlap_px_r{disk_r}", joe changed so that it uses the fraction overlap rather than number of pixels
    overlap_col = f"bouton_overlap_frac_r{disk_r}"
    
    if type_col in out_df.columns:
        #if more than 50% overlap (fraction overlap 0.5), score as syntd. otherwise leave as the original marker type that includes the words "syntdnotscored"
        has_overlap = out_df[overlap_col] > 0.5
    
        out_df.loc[
            has_overlap & (out_df[type_col] == "Shaft_SyntdNotScored"),
            type_col
        ] = "Shaft_SynTd"
    
        out_df.loc[
            has_overlap & (out_df[type_col] == "Spine_SyntdNotScored"),
            type_col
        ] = "Spine_SynTd"
    else:
        print(f"[WARN] No '{type_col}' column found in {csv_path.name}; skipped type update")
    
    out_csv = csv_path.with_name(csv_path.stem + "_snapped_bouton_overlap.csv")
    out_df.to_csv(out_csv, index=False)
    print("wrote:", out_csv)
    return out_df


def run_all_images_and_branches(images_root: Path, puncta_root: Path):
    image_dirs = [
        d for d in images_root.iterdir()
        if d.is_dir() and extract_image_number_from_dir(d) is not None
    ]
    image_dirs = sorted(image_dirs, key=lambda d: extract_image_number_from_dir(d))

    print(f"Found {len(image_dirs)} image directories under {images_root}")

    for image_dir in image_dirs:
        img_num = extract_image_number_from_dir(image_dir)
        assert img_num is not None

        tif_path = find_one_tif(image_dir)
        img = load_full_image(tif_path)

        branch_csvs = find_branch_csvs_for_image(img_num, puncta_root)

        print(
            f"\nImage{img_num}: tif={tif_path.name} | full shape={img.shape} | "
            f"branches found={len(branch_csvs)}"
        )

        if not branch_csvs:
            print(f"  [WARN] No branch CSVs found for Image{img_num} under {puncta_root}")
            continue

        for csv_path in branch_csvs:
            try:
                df, pts, labels = load_branch_csv(csv_path)
                run_one_branch(
                    img,
                    df,
                    pts,
                    labels,
                    img_num=img_num,
                    csv_path=csv_path,
                )
            except Exception as e:
                print(f"[ERROR] Failed on {csv_path}: {e}")


# Run it
run_all_images_and_branches(images_root, puncta_root)

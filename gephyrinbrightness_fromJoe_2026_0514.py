from pathlib import Path
import re
import pandas as pd
import numpy as np
import tifffile as tif


images_root = Path(r'Z:\Joe\2p_data\SOM\ThirdRound\SOM055_DOB051322_TT\Analysis_withAmyCode_cell4\SNTTrace')
puncta_root = Path(r'Z:\Joe\2p_data\SOM\ThirdRound\SOM055_DOB051322_TT\Analysis_withAmyCode_cell4\AlignmentAndChecking\AfterManualEdits')

Gephyrin_CHANNEL_INDEX = 0   # channel used for gephyrin brightness

COORD_COLS = ["axis-0", "axis-1", "axis-2"]

image_dir_pat = re.compile(r"^Image(\d+)$", re.IGNORECASE)
#the line below is from Joe:
csv_img_pat = re.compile(r"Image(\d+)_branch(\d+)_snapped_bouton_overlap_with_empty\.csv$", re.IGNORECASE)
#previous line from Amy is below
#csv_img_pat = re.compile(r"_Image(\d+)_branch\d+", re.IGNORECASE)


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


def load_image_channel(img: np.ndarray, channel_index: int) -> np.ndarray:
    if channel_index < 0 or channel_index >= img.shape[1]:
        raise IndexError(f"channel_index={channel_index} out of range for shape {img.shape}")
    return img[:, channel_index, :, :].astype(np.float32)  # Z,Y,X


def load_branch_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} missing coord columns {missing}. Has columns: {list(df.columns)}")

    pts = df[COORD_COLS].to_numpy(np.float32)
    return df, pts


def find_branch_csvs_for_image(img_num: int, puncta_root: Path) -> list[Path]:
    hits = []
    for p in puncta_root.rglob("*.csv"):
        m = csv_img_pat.search(p.name)
        if m and int(m.group(1)) == img_num:
            hits.append(p)
    return sorted(hits)

#code below written by Claude Haiku 4.5 5/14/26 as instructed by Joe to find mean brightness of 3 x 3 pixel ROI rather than just the marker itself
def run_one_branch(
    img_zycx: np.ndarray,
    df: pd.DataFrame,
    pts_zyx: np.ndarray,
    *,
    csv_path: Path,
    gephyrin_channel_index: int = Gephyrin_CHANNEL_INDEX,
    z_scale: int = 4,
):
    gephyrin = load_image_channel(img_zycx, gephyrin_channel_index)

    Z, H, W = gephyrin.shape

    pts = pts_zyx.astype(np.float32)

    # Convert original marker coordinates directly to image pixel indices
    z_idx = np.clip(np.rint(pts[:, 0] / z_scale).astype(int), 0, Z - 1)
    yy = np.clip(np.rint(pts[:, 1]).astype(int), 0, H - 1)
    xx = np.clip(np.rint(pts[:, 2]).astype(int), 0, W - 1)

    # Mean brightness of 3x3 square centered on each marker point
    gephyrin_brightness = np.zeros(len(pts))
    for i in range(len(pts)):
        z = z_idx[i]
        y = yy[i]
        x = xx[i]
        
        # Define 3x3 neighborhood bounds (clipped to image boundaries)
        y_min, y_max = max(0, y - 1), min(H - 1, y + 1)
        x_min, x_max = max(0, x - 1), min(W - 1, x + 1)
        
        # Extract 3x3 region and compute mean
        neighborhood = gephyrin[z, y_min:y_max+1, x_min:x_max+1]
        gephyrin_brightness[i] = neighborhood.mean()

    out_df = df.copy()
    out_df["gephyrin_z_idx"] = z_idx
    out_df["gephyrin_y_idx"] = yy
    out_df["gephyrin_x_idx"] = xx
    out_df["gephyrin_brightness"] = gephyrin_brightness

    # ADD THESE MISSING LINES:
    out_csv = csv_path.with_name(csv_path.stem + "_gephyrin_brightness.csv")
    out_df.to_csv(out_csv, index=False)

    print("wrote:", out_csv)
    return out_df

#Amy's original code that finds brightness of the pixel where the marker is located, but not a 3 x 3 square ROI surrounding the marker
# def run_one_branch(
#     img_zycx: np.ndarray,
#     df: pd.DataFrame,
#     pts_zyx: np.ndarray,
#     *,
#     csv_path: Path,
#     gephyrin_channel_index: int = Gephyrin_CHANNEL_INDEX,
#     z_scale: int = 4,
# ):
#     gephyrin = load_image_channel(img_zycx, gephyrin_channel_index)

#     Z, H, W = gephyrin.shape

#     pts = pts_zyx.astype(np.float32)

#     # Convert original marker coordinates directly to image pixel indices
#     z_idx = np.clip(np.rint(pts[:, 0] / z_scale).astype(int), 0, Z - 1)
#     yy = np.clip(np.rint(pts[:, 1]).astype(int), 0, H - 1)
#     xx = np.clip(np.rint(pts[:, 2]).astype(int), 0, W - 1)

#     # Brightness at the exact original point pixel
#     gephyrin_brightness = gephyrin[z_idx, yy, xx]

#     out_df = df.copy()
#     out_df["gephyrin_z_idx"] = z_idx
#     out_df["gephyrin_y_idx"] = yy
#     out_df["gephyrin_x_idx"] = xx
#     out_df["gephyrin_brightness"] = gephyrin_brightness

#     out_csv = csv_path.with_name(csv_path.stem + "_gephyrin_brightness.csv")
#     out_df.to_csv(out_csv, index=False)

#     print("wrote:", out_csv)
#     return out_df


def run_all_images_and_branches(images_root: Path, puncta_root: Path):
    image_dirs = [
        d for d in images_root.iterdir()
        if d.is_dir() and extract_image_number_from_dir(d) is not None
    ]
    image_dirs = sorted(image_dirs, key=lambda d: extract_image_number_from_dir(d))

    print(f"Found {len(image_dirs)} image directories under {images_root}")

    for image_dir in image_dirs:
        img_num = extract_image_number_from_dir(image_dir)

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
                df, pts = load_branch_csv(csv_path)
                run_one_branch(
                    img,
                    df,
                    pts,
                    csv_path=csv_path,
                )
            except Exception as e:
                print(f"[ERROR] Failed on {csv_path}: {e}")



run_all_images_and_branches(images_root, puncta_root)

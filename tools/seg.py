"""
Point-seeded 3D gephyrin puncta segmentation (Napari CSV seeds)

What this does:
1) Load a 3-channel TIF stack shaped like (Z, C, Y, X)
2) Compute normch4 = (ch2 * multiplier) / ch1
3) Restrict to a dendrite mask based on ch1 intensity range
4) Build a puncta foreground mask by thresholding normch4_dendrites
5) Use Napari points CSV as watershed markers (optionally snapped to local maxima)
6) Run watershed on -distance_transform to force splits around your provided centers
7) Visualize in napari

Notes:
- Napari points CSV commonly stores coords as z,y,x or axis-0/1/2. This script tries both.
- If your TIF dimension order differs, adjust in load_and_preprocess_image().
"""

import tifffile as tiff
import numpy as np
import pandas as pd

from skimage.segmentation import clear_border, watershed
from skimage.morphology import remove_small_objects, binary_dilation, ball
from scipy.ndimage import distance_transform_edt

import napari
from pathlib import Path
from typing import List, Tuple, Union, Optional

def z_objectj_to_imagej(z_obj: float, num_channels: int) -> float:
    return (int(z_obj) - 1) / num_channels
def _read_any_csv(fp: Union[str, Path]) -> pd.DataFrame:
    fp = str(fp)
    # try comma, then tab
    try:
        df = pd.read_csv(fp)
        if df.shape[1] == 1:
            df = pd.read_csv(fp, sep="\t")
    except Exception:
        df = pd.read_csv(fp, sep="\t")
    df.columns = df.columns.astype(str).str.strip()
    return df

def _pick_first(df: pd.DataFrame, options: List[str]) -> str:
    cols = {c.strip(): c for c in df.columns}
    for opt in options:
        if opt in cols:
            return cols[opt]
    raise ValueError(f"None of these columns found: {options}. Have: {list(df.columns)}")

def load_and_preprocess_image(image_path, min_intensity, max_intensity, multiplier=100.0, eps=1e-6):
    """
    Assumes image shape is (Z, C, Y, X) with at least 3 channels.
    ch1: cell fill / dendrite
    ch2: gephyrin
    ch3: other
    """
    image = tiff.imread(image_path)

    if image.ndim != 4:
        raise ValueError(f"Expected a 4D array (Z,C,Y,X). Got shape {image.shape}")

    if image.shape[1] < 3:
        raise ValueError(f"Expected >=3 channels in axis=1. Got shape {image.shape}")

    ch1 = image[:, 0, :, :].astype(np.float32)
    ch2 = image[:, 1, :, :].astype(np.float32)
    ch3 = image[:, 2, :, :].astype(np.float32)

    # Normalize channel 2 by channel 1 (avoid divide-by-zero)
    normch4 = (ch2 * float(multiplier)) / (ch1 + float(eps))

    # Dendrite mask from ch1 intensity
    dendrite_mask = (ch1 >= float(min_intensity)) & (ch1 <= float(max_intensity))

    # Restrict normalized image to dendrites
    normch4_dendrites = np.where(dendrite_mask, normch4, 0.0).astype(np.float32)

    return ch1, ch2, ch3, normch4, normch4_dendrites, dendrite_mask


# ----------------------------
# Napari CSV points handling
def load_points(
    marker_files: Union[str, Path, List[Union[str, Path]]],
    num_channels: int,
    tp_idx: Optional[int] = None,
    include_landmarks: bool = False,
    label_priority: List[str] = None,
    x_candidates: List[str] = None,
    y_candidates: List[str] = None,
    z_candidates: List[str] = None,
):
    """
    Reads ObjectJ CombinedResults-like CSV(s) and returns points_zyx array for seeding.

    - If marker_files is a list, you can select one timepoint via tp_idx (0-based),
      or concatenate all if tp_idx is None.
    - If tp_idx is not None and 'ojj File Name' exists, filters rows containing '_Image{tp_idx+1}'.
    - Uses label column priority: Final S1 -> Checked S1 -> Original S1 -> S 1 ...
    - Reads coordinates from xpos/ypos/zpos S1 (or fallback candidates).
    - Converts z via z_objectj_to_imagej(z_raw, num_channels).
    - Returns numpy array (N,3) in (z,y,x).
    """

    if label_priority is None:
        label_priority = ["Final S1", "Checked S1", "Original S1", "S 1", "label", "type"]
    if x_candidates is None:
        x_candidates = ["xpos S1", "x"]
    if y_candidates is None:
        y_candidates = ["ypos S1", "y"]
    if z_candidates is None:
        z_candidates = ["zpos S1", "z"]

    if not isinstance(marker_files, (list, tuple)):
        marker_files = [marker_files]

    all_pts = []

    # If caller gave a list and a specific tp_idx, only read that file.
    files_to_read = marker_files
    if tp_idx is not None and len(marker_files) > 1:
        files_to_read = [marker_files[tp_idx]]

    for file_index, fp in enumerate(files_to_read):
        df = _read_any_csv(fp)

        # Filter rows by timepoint tag if requested and possible
        # If user passed tp_idx and only a single file, we still filter by _Image{tp} within that file.
        if tp_idx is not None and "ojj File Name" in df.columns:
            target = tp_idx + 1
            mask = df["ojj File Name"].astype(str).str.contains(fr"_Image{target}\b", na=False)
            df = df.loc[mask].copy()

        label_col = _pick_first(df, label_priority)
        x_col = _pick_first(df, x_candidates)
        y_col = _pick_first(df, y_candidates)
        z_col = _pick_first(df, z_candidates)

        for _, row in df.iterrows():
            mtype = row.get(label_col)
            if pd.isna(mtype):
                continue
            mtype = str(mtype).strip()
            if mtype == "":
                continue

            if (not include_landmarks) and (mtype.lower() == "landmark"):
                continue

            try:
                x = float(row.get(x_col))
                y = float(row.get(y_col))
                z_raw = float(row.get(z_col))
            except (TypeError, ValueError):
                continue

            z_img = float(z_objectj_to_imagej(z_raw, num_channels))

            # XYZ -> ZYX
            all_pts.append((z_img, y, x))

    pts_zyx = np.array(all_pts, dtype=float)
    pts_zyx = np.atleast_2d(pts_zyx)
    if pts_zyx.size == 0:
        return pts_zyx.reshape(0, 3)

    if pts_zyx.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got {pts_zyx.shape}")

    return pts_zyx


def _in_bounds_int_points(points_zyx_int, shape_zyx):
    Z, Y, X = shape_zyx
    z = points_zyx_int[:, 0]
    y = points_zyx_int[:, 1]
    x = points_zyx_int[:, 2]
    return (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)


def snap_points_to_local_max(image, points_zyx, win_zyx=(1, 3, 3), require_positive=True):
    """
    For each point, search a local window in `image` and move to the brightest voxel.

    win_zyx: half-window sizes (dz, dy, dx). Total window size is (2*dz+1, 2*dy+1, 2*dx+1).
    require_positive: if True, will drop snapped points whose local max value <= 0.
    """
    Z, Y, X = image.shape
    pts = np.rint(points_zyx).astype(int)

    keep = _in_bounds_int_points(pts, image.shape)
    pts = pts[keep]

    dz, dy, dx = win_zyx
    snapped = []
    for z, y, x in pts:
        z0, z1 = max(0, z - dz), min(Z, z + dz + 1)
        y0, y1 = max(0, y - dy), min(Y, y + dy + 1)
        x0, x1 = max(0, x - dx), min(X, x + dx + 1)

        patch = image[z0:z1, y0:y1, x0:x1]
        if patch.size == 0:
            continue

        flat = int(np.argmax(patch))
        zz, yy, xx = np.unravel_index(flat, patch.shape)
        z_s, y_s, x_s = z0 + zz, y0 + yy, x0 + xx

        if require_positive and image[z_s, y_s, x_s] <= 0:
            continue

        snapped.append((z_s, y_s, x_s))

    return np.array(snapped, dtype=float)


def markers_from_points(points_zyx, shape, seed_radius_vox=1):
    """
    Build an int32 marker volume where each seed point has a unique positive integer label.

    seed_radius_vox:
      - 0 => single-voxel seed
      - >0 => dilate as a small 3D ball (more stable)
    """
    markers = np.zeros(shape, dtype=np.int32)
    pts = np.rint(points_zyx).astype(int)

    keep = _in_bounds_int_points(pts, shape)
    pts = pts[keep]

    # Assign unique labels
    for i, (z, y, x) in enumerate(pts, start=1):
        markers[z, y, x] = i

    if seed_radius_vox > 0 and pts.shape[0] > 0:
        # Dilate each label independently via repeated ball stamping.
        # (Efficient enough for typical seed counts; for huge counts, optimize.)
        selem = ball(seed_radius_vox)
        out = np.zeros_like(markers)
        for i, (z, y, x) in enumerate(pts, start=1):
            # stamp the ball
            dz, dy, dx = selem.shape
            rz, ry, rx = dz // 2, dy // 2, dx // 2
            z0, z1 = max(0, z - rz), min(shape[0], z + rz + 1)
            y0, y1 = max(0, y - ry), min(shape[1], y + ry + 1)
            x0, x1 = max(0, x - rx), min(shape[2], x + rx + 1)

            # corresponding selem crop
            sz0 = 0 if z - rz >= 0 else -(z - rz)
            sy0 = 0 if y - ry >= 0 else -(y - ry)
            sx0 = 0 if x - rx >= 0 else -(x - rx)
            sz1 = dz if z + rz + 1 <= shape[0] else dz - ((z + rz + 1) - shape[0])
            sy1 = dy if y + ry + 1 <= shape[1] else dy - ((y + ry + 1) - shape[1])
            sx1 = dx if x + rx + 1 <= shape[2] else dx - ((x + rx + 1) - shape[2])

            stamp = selem[sz0:sz1, sy0:sy1, sx0:sx1]
            roi = out[z0:z1, y0:y1, x0:x1]
            roi[stamp] = i
            out[z0:z1, y0:y1, x0:x1] = roi

        markers = out

    return markers


def seeded_gephyrin_segmentation(
    image_norm_dendrites,
    dendrite_mask,
    points_csv_path,
    num_stddevs=1.0,
    min_puncta_size=3,
    seed_radius_vox=1,
    snap_to_local_maxima=True,
    snap_win_zyx=(1, 3, 3),
    ensure_seeds_inside_foreground=True,
    foreground_dilate_vox=0,
):
    """
    image_norm_dendrites: float32 (Z,Y,X) gephyrin-like signal restricted to dendrites
    dendrite_mask: bool (Z,Y,X)
    points_csv_path: Napari CSV points path

    Returns:
      labels (int32), puncta_mask (bool), markers (int32), dist (float32)
    """



    img = image_norm_dendrites.astype(np.float32)

    # Foreground threshold using only dendrite pixels
    vals = img[dendrite_mask]
    if vals.size == 0:
        raise ValueError("dendrite_mask has no True voxels; adjust min/max intensity range.")

    thr = float(vals.mean() + num_stddevs * vals.std())

    puncta_mask = clear_border(img > thr)
    puncta_mask = remove_small_objects(puncta_mask, min_size=int(min_puncta_size))

    if foreground_dilate_vox > 0:
        puncta_mask = binary_dilation(puncta_mask, footprint=ball(int(foreground_dilate_vox)))

    # Load points
    pts_zyx = load_points(
    marker_files=points_csv_path,
    num_channels=3,      # <-- set to your stack’s num channels used in ObjectJ z indexing
    tp_idx=None,         # or 0 if you want _Image1 only
    include_landmarks=False
    )

    # Optionally snap to local maxima of the image (usually improves stability)
    if snap_to_local_maxima:
        pts_zyx = snap_points_to_local_max(img, pts_zyx, win_zyx=snap_win_zyx, require_positive=True)

    # Build markers (unique labels per point)
    markers = markers_from_points(pts_zyx, shape=img.shape, seed_radius_vox=int(seed_radius_vox))

    # Optionally force markers to lie inside the foreground
    if ensure_seeds_inside_foreground:
        markers = markers * puncta_mask.astype(np.int32)

    # If all seeds got removed (e.g. threshold too strict), fail fast
    if markers.max() == 0:
        raise ValueError(
            "No valid seeds remain after (optional) snapping / foreground restriction. "
            "Try: lower num_stddevs, increase seed_radius_vox, increase foreground_dilate_vox, "
            "or disable ensure_seeds_inside_foreground."
        )

    # Watershed on -distance_transform (classic blob splitting)
    dist = distance_transform_edt(puncta_mask).astype(np.float32)
    labels = watershed(-dist, markers, mask=puncta_mask).astype(np.int32)

    # Remove tiny regions after watershed
    labels = remove_small_objects(labels, min_size=int(min_puncta_size)).astype(np.int32)
    print("img shape (Z,Y,X):", img.shape)
    print("thr:", thr)
    print("puncta_mask voxels:", int(puncta_mask.sum()))

    # if you have pts_zyx at this point:
    print("seeds count:", int(len(pts_zyx)))
    if len(pts_zyx) > 0:
        pts_i = np.rint(pts_zyx).astype(int)
        pts_i[:,0] = np.clip(pts_i[:,0], 0, img.shape[0]-1)
        pts_i[:,1] = np.clip(pts_i[:,1], 0, img.shape[1]-1)
        pts_i[:,2] = np.clip(pts_i[:,2], 0, img.shape[2]-1)
        frac = puncta_mask[pts_i[:,0], pts_i[:,1], pts_i[:,2]].mean()
        print("fraction of seeds inside puncta_mask:", float(frac))

    return labels, puncta_mask, markers, dist, thr


import matplotlib.pyplot as plt

def plot_seed_debug_xy(
    puncta_mask,
    pts_raw,
    pts_local,
    pts_fg,
    title="Seed alignment (XY projection)",
    figsize=(6, 6),
):
    """
    Fast matplotlib diagnostic plot.
    - puncta_mask: (Z,Y,X) bool
    - pts_*: (N,3) arrays in (z,y,x)
    """

    # XY projection of puncta foreground
    proj = puncta_mask.max(axis=0)  # shape (Y,X)

    plt.figure(figsize=figsize)
    plt.imshow(proj, cmap="gray", origin="upper")
    
    if pts_raw is not None and len(pts_raw) > 0:
        plt.scatter(
            pts_raw[:, 2], pts_raw[:, 1],
            s=15, c="red", label="raw", alpha=0.8
        )

    if pts_local is not None and len(pts_local) > 0:
        plt.scatter(
            pts_local[:, 2], pts_local[:, 1],
            s=15, c="orange", label="local max", alpha=0.8
        )

    if pts_fg is not None and len(pts_fg) > 0:
        plt.scatter(
            pts_fg[:, 2], pts_fg[:, 1],
            s=15, c="lime", label="foreground", alpha=0.9
        )

    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend(loc="upper right", markerscale=1.5)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Napari display helpers
# ----------------------------
def display_images(viewer, images, names):
    for image, name in zip(images, names):
        viewer.add_image(image, name=name)


def main(
    image_path,
    points_csv_path,
    min_intensity=10,
    max_intensity=80,
    multiplier=100.0,
    num_stddevs=0,
    min_puncta_size=3,
    seed_radius_vox=1,
    snap_to_local_maxima=True,
    snap_win_zyx=(1, 3, 3),
    ensure_seeds_inside_foreground=True,
    foreground_dilate_vox=0,
    num_channels=3,          # for your ObjectJ z converter
    tp_idx=None,             # set 0 to force _Image1 filtering if needed
    include_landmarks=False,
):
    """
    Matplotlib-only debug main:
    - runs preprocessing + segmentation
    - plots fast XY projection with raw/local/foreground-snapped seeds over puncta_mask
    - (optional) plots z histograms
    """

    # --- load image & preprocess ---
    ch1, ch2, ch3, normch4, normch4_dendrites, dendrite_mask = load_and_preprocess_image(
        image_path=image_path,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        multiplier=multiplier,
    )

    # --- load seeds (raw) ---
    # NOTE: assumes you have load_points(marker_files, num_channels, tp_idx, include_landmarks)
    pts_raw = load_points(
        marker_files=points_csv_path,
        num_channels=num_channels,
        tp_idx=tp_idx,
        include_landmarks=include_landmarks,
    )
    pts_raw = np.asarray(pts_raw, dtype=float)
    pts_raw = np.atleast_2d(pts_raw)
    if pts_raw.size == 0:
        pts_raw = pts_raw.reshape(0, 3)

    # --- run segmentation, but also get the intermediate point states ---
    # For this to work, seeded_gephyrin_segmentation should return:
    # labels, puncta_mask, markers, dist, thr, pts_raw_used, pts_local, pts_fg
    out = seeded_gephyrin_segmentation(
        image_norm_dendrites=normch4_dendrites,
        dendrite_mask=dendrite_mask,
        points_csv_path=points_csv_path,
        num_stddevs=num_stddevs,
        min_puncta_size=min_puncta_size,
        seed_radius_vox=seed_radius_vox,
        snap_to_local_maxima=snap_to_local_maxima,
        snap_win_zyx=snap_win_zyx,
        ensure_seeds_inside_foreground=ensure_seeds_inside_foreground,
        foreground_dilate_vox=foreground_dilate_vox,
        # if your seeded_gephyrin_segmentation needs these, add them there too:
        # num_channels=num_channels,
        # tp_idx=tp_idx,
        # include_landmarks=include_landmarks,
    )

    if len(out) == 5:
        # fallback if you haven't updated seeded_gephyrin_segmentation to return points
        labels, puncta_mask, markers, dist, thr = out

        pts_local = pts_raw.copy()
        if snap_to_local_maxima and pts_local.shape[0] > 0:
            pts_local = snap_points_to_local_max(
                normch4_dendrites, pts_local, win_zyx=snap_win_zyx, require_positive=True
            )

        pts_fg = pts_local.copy()
        if ensure_seeds_inside_foreground and pts_fg.shape[0] > 0:
            # only if you have this function; otherwise set ensure_seeds_inside_foreground=False
            pts_fg = snap_points_to_foreground(puncta_mask, pts_fg, max_radius=8)

    else:
        labels, puncta_mask, markers, dist, thr, pts_raw_used, pts_local, pts_fg = out
        # keep pts_raw from loader for plotting consistency
        # (pts_raw_used may be identical, but depends on your implementation)

    # --- fast matplotlib plots ---
    plot_seed_debug_xy(
        puncta_mask=puncta_mask,
        pts_raw=pts_raw,
        pts_local=pts_local,
        pts_fg=pts_fg,
        title=f"Seeds vs puncta mask (XY). thr={thr:.3f}, std={num_stddevs}",
    )

    # # optional: z sanity check
    # try:
    #     plot_seed_debug_z(pts_raw, pts_fg, z_max=puncta_mask.shape[0])
    # except NameError:
    #     pass

    # optional: quick console stats
    if pts_raw.shape[0] > 0:
        print("[debug] image shape (Z,Y,X):", normch4_dendrites.shape)
        print("[debug] thr:", thr)
        print("[debug] puncta_mask voxels:", int(puncta_mask.sum()))
        print("[debug] seeds raw/local/fg:", pts_raw.shape[0], pts_local.shape[0], pts_fg.shape[0])



if __name__ == "__main__":
    image_path = "/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/Automated_Puncta_Detection/Image1/SOM022_Image 1_MotionCorrected.tif"
    points_csv_path ='/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/Aligned_afterManualCheck/SOM022_b2_AlignedChecked.csv'  # <-- set this

    main(
        image_path=image_path,
        points_csv_path=points_csv_path,
        min_intensity=10,
        max_intensity=80,
        multiplier=100.0,
        num_stddevs=1.0,
        min_puncta_size=3,
        seed_radius_vox=5,            # try 2 if seeds often fall just outside foreground
        snap_to_local_maxima=True,    # usually helps
        snap_win_zyx=(1, 3, 3),       # tune to your puncta size / z spacing
        ensure_seeds_inside_foreground=False,
        foreground_dilate_vox=5,      # try 1–2 if your threshold mask is slightly too tight
    )

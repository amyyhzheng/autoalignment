import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
import pandas as pd
import napari


# ------------------------------------------------------
# LOAD TIFF CHANNEL 2
# ------------------------------------------------------
def load_ch2(tif_path, channel_index=1):
    arr = tifffile.imread(tif_path)

    if arr.ndim == 3:
        # Already 3D
        return arr
    elif arr.ndim == 4:
        # Assume (C, Z, Y, X)
        return arr[channel_index]
    else:
        raise ValueError(f"Unexpected TIFF shape: {arr.shape}")
def load_xyz_points(path):
    """
    Expects columns: xpos S1, ypos S1, zpos S1
    Returns a numpy array of shape (N, 3) in (z, y, x)
    """
    df = pd.read_csv(path, sep="\t", engine="python")  # or sep="," if CSV
    df = df.dropna()

    x = df["xpos S1"].values
    y = df["ypos S1"].values
    z = df["zpos S1"].values

    points_zyx = np.vstack([z, y, x]).T
    return points_zyx


# ------------------------------------------------------
# BUILD MASK (threshold ch2)
# ------------------------------------------------------
def make_mask(vol):
    th = threshold_otsu(vol)
    mask = vol > th
    return mask


# ------------------------------------------------------
# CREATE MARKERS (one label per point)
# ------------------------------------------------------
def make_markers(points_zyx, shape):
    markers = np.zeros(shape, dtype=np.int32)
    Z, Y, X = shape

    for i, (z, y, x) in enumerate(points_zyx, start=1):
        zi, yi, xi = int(z), int(y), int(x)
        if 0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X:
            markers[zi, yi, xi] = i

    return markers


# ------------------------------------------------------
# MAIN SEGMENTATION FUNCTION
# ------------------------------------------------------
def segment_from_points(tif_path, points_path):
    vol = load_ch2(tif_path)          # use argument
    pts = load_xyz_points(points_path)
    mask = make_mask(vol)
    markers = make_markers(pts, vol.shape)

    dist = distance_transform_edt(mask)
    energy = -dist

    labels = watershed(energy, markers=markers, mask=mask)
    return vol, pts, labels


if __name__ == "__main__":
    tif_path = "/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/Automated_Puncta_Detection/Image1/With_normch4-SOM022_Image 1_MotionCorrected.tif"
    points_path = "/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/PunctaScoring/b2/SynapseMarkers/Aligned_afterManualCheck/CombinedResults.csv"

    vol, pts, labels = segment_from_points(tif_path, points_path)

    # --- visualize in napari ---
    viewer = napari.Viewer()
    viewer.add_image(vol, name="ch2", contrast_limits=[vol.min(), vol.max()])
    viewer.add_points(pts, name="S1 points", size=4)
    viewer.add_labels(labels, name="watershed_masks")
    napari.run()


import numpy as np
import pandas as pd
from skimage.draw import disk
import napari
import tifffile as tif

def points_mask_from_points_layer(points_zyx, shape_zyx, radius_px=20):
    Z, Y, X = 300, 800, 800
    m = np.zeros((Z, Y, X), dtype=bool)

    pts = np.asarray(points_zyx, dtype=float)

    z = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    x = np.round(pts[:, 2]).astype(int)

    in_z = (0 <= z) & (z < Z)
    in_xy = (0 <= y) & (y < Y) & (0 <= x) & (x < X)
    in_all = in_z & in_xy

    print(f"Total points: {len(pts)}")
    print(f"In-bounds z: {int(in_z.sum())} / {len(pts)}")
    print(f"In-bounds xyz (center): {int(in_all.sum())} / {len(pts)}")
    if (~in_z).any():
        bad = np.where(~in_z)[0]
        print("Example z out of bounds (up to 10):")
        for i in bad[:10]:
            print(f"  idx {i}: (z,y,x)=({z[i]},{y[i]},{x[i]})")

    for zi, yi, xi in zip(z[in_z], y[in_z], x[in_z]):
        rr, cc = disk((yi, xi), radius_px, shape=(Y, X))  # clips to XY bounds
        m[zi, rr, cc] = True

    return m



def colocalization_mask(point_mask_zyx: np.ndarray, other_channel_zyx: np.ndarray, thresh=None) -> np.ndarray:
    if point_mask_zyx.shape != other_channel_zyx.shape:
        raise ValueError(f"Shape mismatch: {point_mask_zyx.shape} vs {other_channel_zyx.shape}")
    other_pos = (other_channel_zyx > 0) if thresh is None else (other_channel_zyx > thresh)
    return point_mask_zyx & other_pos


def load_points_layer_from_csv(viewer: napari.Viewer, file_path: str, layer_name: str = None):
    """
    Matches your desired reading pattern:
      - df = pd.read_csv(file_path)
      - points = df[['axis-0','axis-1','axis-2']].to_numpy()
      - features: label, Notes, type (optional)
      - add_points(..., features=..., text=...)
    """
    df = pd.read_csv(file_path)

    # napari points convention: (axis-0, axis-1, axis-2) == (z, y, x) for 3D data
    points = df[["axis-0", "axis-1", "axis-2"]].to_numpy(dtype=float)

    features = {
        "label": df["label"].to_numpy() if "label" in df.columns else np.array([""] * len(df)),
        "Notes": df["Notes"].to_numpy() if "Notes" in df.columns else np.array([""] * len(df)),
        "type": df["type"].to_numpy() if "type" in df.columns else np.array(["Unknown"] * len(df)),
    }

    if layer_name is None:
        layer_name = f"Points Layer {len([l for l in viewer.layers if isinstance(l, napari.layers.Points)]) + 1}"

    points_layer = viewer.add_points(
        points,
        features=features,
        size=7,
        border_width=0.1,
        border_color="white",
        # If you have your own type->color mapping, swap this line out
        face_color="type",
        text={"text": "label", "size": 10, "color": "white", "anchor": "center"},
        name=layer_name,
    )
    return points_layer

if __name__ == "__main__":
    # Edit these to your files
    points_csv = '/Volumes/nedividata/Amy/files_for_amy_fromJoe/SOM023Files_afterNewUnmixingAndScoring/PunctaScoring/branch2/Image0_branch2.csv'   # must have axis-0/1/2 columns
    # other_channel should be a (Z,Y,X) numpy array you already loaded
    other_image = tif.imread('/Volumes/nedividata/Amy/files_for_amy_fromJoe/SOM023Files_afterNewUnmixingAndScoring/NewUnmixing_concatenatedimages_blinded/SOM023_Image0.tif')
    other_channel = other_image[:, 2, :, :]
    print("Other channel shape:", other_channel.shape)

    radius_px =20
    thresh = None  # set a number if you want

    viewer = napari.Viewer(ndisplay=3)
    # If you already have an image:
    # viewer.add_image(other_channel, name="other", rendering="mip")

    pts_layer = load_points_layer_from_csv(viewer, points_csv, layer_name="centroids")


    # pt_mask = points_mask_from_points_layer(pts_layer.data, radius_px=radius_px)
    #     # colo = colocalization_mask(pt_mask, other_channel, thresh=thresh)

    # viewer.add_labels(pt_mask.astype(np.uint8), name=f"disk_r{radius_px}_mask")
    # Build masks ONLY after you have an image shape to target
    if other_channel is not None:
        pt_mask = points_mask_from_points_layer(pts_layer.data, other_channel.shape, radius_px=radius_px)
        # colo = colocalization_mask(pt_mask, other_channel, thresh=thresh)

        viewer.add_labels(pt_mask.astype(np.uint8), name=f"disk_r{radius_px}_mask")
        # viewer.add_labels(colo.astype(np.uint8), name="colocalization_mask")

    napari.run()

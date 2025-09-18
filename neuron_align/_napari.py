from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

try:
    import napari
    import tifffile as tif
except Exception:  # pragma: no cover
    napari = None
    tif = None

from .io_utils import z_objectj_to_imagej


def test_results(actual_csv_path, predicted_rows):
    if napari is None:
        print("napari not installed; skipping visualization")
        return {}

    df_actual = pd.read_csv(actual_csv_path)
    pred_df = pd.DataFrame(predicted_rows, columns=['image','markerID','markerType','x','y','z'])

    def color_for(t: str):
        m = {
            'inhibitoryshaft':'red', 'nothing':'gray',
            'spinewithinhibitorysynapse':'blue', 'spinewithinhsynapse':'blue',
            'nudespine':'lightblue', 'landmark':'white'
        }
        return m.get(str(t).lower(), 'white')

    viewers = {}
    for img_i in range(1, 7):
        viewer = napari.Viewer(ndisplay=3)
        viewers[img_i] = viewer
        # Actual
        S = f'S{img_i}'
        final_col = f'Final S{img_i}'
        xcol, ycol, zcol = f'xpos S{img_i}', f'ypos S{img_i}', f'zpos S{img_i}'
        valid = df_actual[~pd.isna(df_actual[final_col])]
        actual_pts = list(zip(valid[zcol], valid[xcol], valid[ycol]))
        actual_lbls = [f"A{g}" for g in valid['Marker'].tolist()]
        actual_cols = [color_for(t) for t in valid[final_col].tolist()]
        if actual_pts:
            viewer.add_points(actual_pts, size=10, face_color=actual_cols, name='Actual',
                              text={'string': actual_lbls, 'size':10, 'color':'white'})
        # Predicted
        pred_img = pred_df[pred_df['image'] == f'Image{img_i}']
        pred_pts = list(zip(pred_img['z'], pred_img['x'], pred_img['y']))
        pred_lbls = [f"P{g}" for g in pred_img['markerID'].tolist()]
        pred_cols = [color_for(t) for t in pred_img['markerType'].tolist()]
        if pred_pts:
            viewer.add_points(pred_pts, size=10, face_color=pred_cols, name='Predicted',
                              text={'string': pred_lbls, 'size':10, 'color':'white'})
    return viewers

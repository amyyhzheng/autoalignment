import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMenu, QInputDialog
import time
from PyQt5.QtGui import QMouseEvent, QCursor
from PyQt5.QtCore import Qt

'''
layer.utils contains all functions that are dependent on a points layer. This includes the following:
- The increasing points number with successive points
- Point editing logic

Everything is then tied together with the create_point_label_handler which ties
these functions to new layers. [Include all the places this is called]
'''

# def create_point_label_handler(points_layer):
#     """
#     Creates a handler to manage point labeling incrementally.

#     :param points_layer: The points layer to attach the handler to.
#     """
#     @points_layer.events.data.connect
#     def label_points(event):
#         num_points = len(points_layer.data)
#         labels = points_layer.features.get('label', pd.Series(dtype=int))

#         # Check if labels exist correctly
#         new_label = (labels.iloc[-1] + 1) if not labels.empty else 0

#         points_layer.feature_defaults['label'] = new_label if num_points > 1 else 0
#         print(f"Updated Labels: {points_layer.features['label']}")
#     return label_points

def assign_incremental_labels(points_layer):
    """
    Automatically assigns incremental labels to new points in the layer.
    
    :param points_layer: The points layer to attach the handler to.
    """
    @points_layer.events.data.connect
    def update_labels(event):
        num_points = len(points_layer.data)
        labels = points_layer.features.get('label', pd.Series(dtype=int))

        # Auto-increment labels for new points
        new_label = (labels.iloc[-1] + 1) if not labels.empty else 0
        points_layer.feature_defaults['label'] = new_label if num_points > 1 else 0
        print(f"Updated Labels: {points_layer.features['label']}")

def enable_Notes_editing(points_layer, viewer):
    """
    Enables right-click editing of the 'Notes' feature in Napari.
    
    :param points_layer: The points layer where 'Notes' should be editable.
    """
    @points_layer.mouse_drag_callbacks.append
    def on_right_click(layer, event):
        """Detects right-click and opens a context menu for editing Notes."""
        if event.type == 'mouse_press' and event.button == 2:  # Right-click (button 2)
            index = layer.get_value(event.position)
            
            if index is not None:
                menu = QMenu()
                edit_action = menu.addAction("Edit Notes")
                action = menu.exec_(event.native.globalPos())  # Show the menu at cursor position
                
                if action == edit_action:
                    # Open input dialog
                    current_Notes = str(layer.features['Notes'].iloc[index])
                    new_Notes, ok = QInputDialog.getText(
                        None, "Edit Notes", "Enter new Notes value:", text=current_Notes
                    )
                    
                    if ok and new_Notes:
                        # Update Notes feature
                        layer.features.loc[index, 'Notes'] = str(new_Notes)

                        # # Force a refresh of the text labels
                        # layer.text = {'string': '{label} | {Notes}', 'size': 10, 'color': 'white', 'anchor': 'center'}
                        
                        layer.refresh()  # Refresh Napari display

                
                # **Step 2: Clear selection**
                layer.selected_data = set()  # Deselect the point
                # **Step 1: Temporarily switch to select mode**
                time.sleep(0.05)
                layer.mode = 'pan_zoom'
                pos = QCursor.pos()  # Get current cursor position
                global_pos = viewer.window._qt_viewer.canvas.mapFromGlobal(pos)  # Convert global pos to canvas pos
                event = QMouseEvent(QMouseEvent.MouseButtonRelease, QPoint(global_pos.x(), global_pos.y()), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
                viewer.window._qt_viewer.canvas.mouseReleaseEvent(event)
                
                # # **Step 3: Prevent Napari from re-selecting the point**
                # time.sleep(0.05)  # Small delay to allow UI update
                # layer.selected_data = set()
def create_point_label_handler(points_layer, viewer):
    """
    Combines automatic label assignment and editable Notes values.
    
    :param points_layer: The napari points layer to enhance.
    """
    assign_incremental_labels(points_layer)
    enable_Notes_editing(points_layer, viewer)

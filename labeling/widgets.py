import numpy as np
from tifffile import imread
import napari
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QApplication, QFileDialog, QMessageBox
import sys
import pandas as pd
from magicgui import magicgui

from constants import INITIAL_FEATURES, INITIAL_POINTS, EXCITATORY_TYPE_TO_COLOR, EXCITATORY_MAPPING_NAME
from layer_utils import create_point_label_handler
from typetocolor import TypeToColor


class UpdatePointTypeWidget(QWidget):
    def __init__(self, viewer, points_layers, mapping_name):

        '''
        Currently following logic that each viewer 
        has it's own separate color mapping/types -> If 
        color mapping/types choose to be different within a single viewer, 
        Need to update the update_point_type function

        Need to add input for the color types
        '''        
        super().__init__()
        self.viewer = viewer

        self.points_layers = points_layers
        self.current_points_layer = next(iter(points_layers.values()))
        self.viewer.layers.selection.events.active.connect(self.update_active_layer)

        #Define the mapping names/get type to color
        self.mapping_name = mapping_name
        self.type_to_color = TypeToColor.get_mapping(mapping_name)
        self.type_choices = list(self.type_to_color.keys())


        layout = QVBoxLayout()
        self.label = QLabel("Select Point Type:")
        layout.addWidget(self.label)


        self.combo_box = QComboBox()
        self.combo_box.addItems(self.type_choices)
        self.combo_box.currentTextChanged.connect(self.update_point_type)
        layout.addWidget(self.combo_box)

        self.new_type_label = QLabel("Add New Type:")
        layout.addWidget(self.new_type_label)

        self.new_type = QLineEdit()
        layout.addWidget(self.new_type)

        self.new_color_label = QLabel("New Color (#RRGGBB):")
        layout.addWidget(self.new_color_label)

        self.new_color = QLineEdit()
        self.new_color.setText("#FFFFFF")
        layout.addWidget(self.new_color)

        self.add_button = QPushButton("Add Type")
        self.add_button.clicked.connect(self.add_new_type)
        layout.addWidget(self.add_button)

        self.viewer.layers.events.changed.connect(self.update_widget_for_active_layer)

        self.setLayout(layout)

    def update_active_layer(self, event):
        """Callback to update the current active points layer."""
        active_layer = self.viewer.layers.selection.active
        if isinstance(active_layer, napari.layers.Points):
            self.current_points_layer = active_layer
        else:
            self.current_points_layer = None

    def update_point_type(self, point_type):
        """Update the type for selected points in the current points layer."""
        if self.current_points_layer is not None:
            selected_points = list(self.current_points_layer.selected_data)
            if selected_points:
                # Update 'type' feature for selected points
                for idx in selected_points:
                    self.current_points_layer.features['type'][idx] = point_type
                
                # Update all face colors based on types
                all_types = self.current_points_layer.features['type']
                new_colors = TypeToColor.map_types_to_colors(all_types, self.mapping_name)
                
                # Update the face colors for all points
                self.current_points_layer.face_color = new_colors
                self.current_points_layer.face_color_mode = 'direct'
                
                # Force a refresh
                self.current_points_layer.refresh_colors()
            else:
                # Update default for new points
                self.current_points_layer.feature_defaults['type'] = point_type
                color = TypeToColor.map_types_to_colors(point_type, self.mapping_name)[0]
                self.current_points_layer.feature_defaults['face_color'] = color

            
    def add_new_type(self):
        '''
        Updated to update the TypeToColor class 
        '''
        new_type = self.new_type.text()
        new_color = self.new_color.text()

        if not new_type:
            QMessageBox.warning(self, "Invalid Input", "Type name cannot be empty.")
            return

        if not (new_color.startswith('#') and len(new_color) == 7):
            QMessageBox.warning(self, "Invalid Input", "Color must be in the format #RRGGBB.")
            return

        # Check if the type already exists
        if new_type in self.type_choices:
            QMessageBox.warning(self, "Duplicate Type", f"Type '{new_type}' already exists.")
            return
        
        # Add the new type and color to the TypeToColor mapping
        TypeToColor.update_mapping(self.mapping_name, new_type, new_color)

        # Update the type_choices and combo box
        self.type_choices.append(new_type)
        self.combo_box.addItem(new_type)

        # Refresh the colors of the current points layer
        if self.current_points_layer is not None:
            all_types = self.current_points_layer.features['type']
            new_colors = TypeToColor.map_types_to_colors(all_types, self.mapping_name)
            self.current_points_layer.face_color = new_colors
            self.current_points_layer.face_color_mode = 'direct'
            self.current_points_layer.refresh_colors()

        # Clear the input fields
        self.new_type.clear()
        self.new_color.setText("#FFFFFF")

    def update_widget_for_active_layer(self, event):
        '''Update the widget when the active layer changes.'''
        selected_layers = list(self.viewer.layers.selection)  # Get currently selected layers
        if selected_layers:
            active_layer = selected_layers[0]  # Get the first selected layer
            if active_layer in self.points_layers.values():
                # Update current_points_layer to the selected layer
                self.current_points_layer = active_layer
                self.update_combo_box()  # Refresh the combo box for the new layer
            else:
                # If the selected layer is not a points_layer, clear everything
                self.current_points_layer = None
                self.combo_box.clear()
        else:
            # If no layer is selected, clear everything
            self.current_points_layer = None
            self.combo_box.clear()
'''

# Can edit later - Functions to update the widget based on the active layer. 


    def update_combo_box(self):
        """Update the combo box based on the current points layer."""
        if self.current_points_layer:
            self.combo_box.clear()
            self.combo_box.addItems(type_choices)
            self.combo_box.setCurrentText(self.current_points_layer.feature_defaults.get('type', type_choices[0]))
            self.combo_box.currentTextChanged.connect(self.update_point_type)

    def update_point_type(self, point_type):
        """Update the type for selected points in the current points layer."""
        if self.current_points_layer is not None:
            selected_points = list(self.current_points_layer.selected_data)
            if selected_points:
                # Update 'type' feature for selected points
                for idx in selected_points:
                    self.current_points_layer.features['type'][idx] = point_type
                
                # Update all face colors based on types
                all_types = self.current_points_layer.features['type']
                new_colors = map_types_to_colors(all_types)
                
                # Update the face colors for all points
                self.current_points_layer.face_color = new_colors
                self.current_points_layer.face_color_mode = 'direct'
                
                # Force a refresh
                self.current_points_layer.refresh_colors()
            else:
                # Update default for new points
                self.current_points_layer.feature_defaults['type'] = point_type
                color = map_types_to_colors([point_type])[0]
                self.current_points_layer.feature_defaults['face_color'] = color'''

class AddPointsLayerWidget(QWidget):
    """Widget to add new points layers dynamically."""
    def __init__(self, viewer, points_layers, update_widget):
        super().__init__()
        self.viewer = viewer
        self.points_layers = points_layers
        self.update_widget = update_widget

        layout = QVBoxLayout()

        self.add_button = QPushButton("Add Points Layer")
        self.add_button.clicked.connect(self.add_points_layer)
        layout.addWidget(self.add_button)

        self.setLayout(layout)

    def add_points_layer(self):
        """Add a new points layer to the viewer."""
        layer_name = f"Points Layer {len(self.points_layers) + 1}"
        points_layer = self.viewer.add_points(
            INITIAL_POINTS,
            features=INITIAL_FEATURES,
            size=7,
            edge_width=0.1,
            edge_color='white',
            face_color= TypeToColor.map_types_to_colors(INITIAL_FEATURES['type'], self.update_widget.mapping_name)[0],
            text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
            name=layer_name,
        )

        self.points_layers[layer_name] = points_layer

        # Configure point defaults with correct type and initial label
        points_layer.feature_defaults = {
            'label': 0, 
            'Notes': 1, 
            'type': 'Ignore'
        }
        
        # Create handler for auto-incrementing labels
        create_point_label_handler(points_layer, self.viewer)

        # Set initial features for the layer
        points_layer.features = {
            'label': np.array([1]),
            'Notes': np.array([1]),
            'type': np.array(['Ignore'])
        }

        # Update the UpdatePointTypeWidget with the new layer
        self.update_widget.points_layers = self.points_layers
        self.update_widget.update_widget_for_active_layer(None)  # Force update

        print(f"Added new points layer: {layer_name}")

class CenterOnPointWidget(QWidget):
    """Widget to center the viewer on a specific point in the active points layer."""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        layout = QVBoxLayout()
        self.label = QLabel("Enter Point Number:")
        self.point_input = QLineEdit()
        self.center_button = QPushButton("Center on Point")
        self.center_button.clicked.connect(self.center_on_point)

        layout.addWidget(self.label)
        layout.addWidget(self.point_input)
        layout.addWidget(self.center_button)
        self.setLayout(layout)

    def center_on_point(self):
        """Center the viewer on the specified point."""
        try:
            point_number = int(self.point_input.text())
            active_layer = self.viewer.layers.selection.active

            if not isinstance(active_layer, napari.layers.Points):
                QMessageBox.warning(self, "Error", "No active points layer selected.")
                return

            # Retrieve the 'label' feature from the active points layer
            labels = active_layer.features['label']
            point_index = labels[labels == point_number].index[0]

            # Get the coordinates of the specified point
            point_coords = active_layer.data[point_index]

            # Center the viewer's camera on the X and Y axes
            self.viewer.camera.center = (point_coords[1], point_coords[2])  
            self.viewer.camera.zoom = 5  # Adjust zoom level as needed

            # Handle the Z dimension separately
            if len(point_coords) > 2:
                self.viewer.dims.set_point(2, point_coords[0])  # Adjust the Z-dimension position

        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid integer for the point number.")
        except (IndexError, KeyError):
            QMessageBox.warning(self, "Error", "Point number not found or 'label' feature is missing.")

class ZoomLevelWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.label = QLabel(f"Zoom Level: {self.viewer.camera.zoom:.2f}")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.viewer.camera.events.zoom.connect(self.update_zoom_level)

    def update_zoom_level(self, event):
        self.label.setText(f"Zoom Level: {self.viewer.camera.zoom:.2f}")

class AddPointsFromCSVWidget(QWidget):
    """Widget to load points layer data from a CSV file."""
    def __init__(self, viewer, points_layers, update_widget):
        super().__init__()
        self.viewer = viewer
        self.points_layers = points_layers
        self.update_widget = update_widget

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Points Layer from CSV")
        self.load_button.clicked.connect(self.load_points_from_csv)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

    def load_points_from_csv(self):
        """Load points layer from a CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options = options
        )
        if not file_path:
            return  # User canceled the file dialog

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract the coordinates
            points = df[['axis-0', 'axis-1', 'axis-2']].to_numpy()

            # Extract the features
            features = {
                'label': df['label'].to_numpy(),
                'Notes': df['Notes'].to_numpy(),
                'type': df['type'].values if 'type' in df.columns else np.array(['Unknown'] * len(df))
            }

            # Add a new points layer
            layer_name = f"Points Layer {len(self.points_layers) + 1}"
            points_layer = self.viewer.add_points(
                points,
                features=features,
                size=7,
                edge_width=0.1,
                edge_color='white',
                face_color = TypeToColor.map_types_to_colors(features['type'], self.update_widget.mapping_name),
                text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
                name=layer_name,
            )

            # Add the new layer to points_layers dictionary
            self.points_layers[layer_name] = points_layer
            create_point_label_handler(points_layer, self.viewer)
            # Update the UpdatePointTypeWidget with the new layer
            self.update_widget.points_layers = self.points_layers
            self.update_widget.update_widget_for_active_layer(None)  # Force update

            print(f"Loaded points layer from CSV: {layer_name}")
        except Exception as e:
            print(f"Error loading points layer from CSV: {e}")



class AddPointsFromObjectJWidget(QWidget):
    """Widget to load points layer data from a custom CSV file."""
    def __init__(self, viewer, points_layers, update_widget):
        super().__init__()
        self.viewer = viewer
        self.points_layers = points_layers
        self.update_widget = update_widget

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Points Layer from ObjectJ")
        self.load_button.clicked.connect(self.load_points_from_csv)
        layout.addWidget(self.load_button)

        self.setLayout(layout)
import pandas as pd
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog

class AddPointsFromObjectJWidget(QWidget):
    """Widget to load points layer data from a custom CSV file.
    Currently an arbitrary scaling - likely need to change it to 
    Aygul's settings 
    """
    def __init__(self, viewer, points_layers, update_widget):
        super().__init__()
        self.viewer = viewer
        self.points_layers = points_layers
        self.update_widget = update_widget
        self.scale_factors = [1,12, 12] #Z, X, Y Scaling 

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Points Layer from ObjectJ")
        self.load_button.clicked.connect(self.load_points_from_csv)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

    def load_points_from_csv(self):
        """Load points layer from a custom CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if not file_path:
            return  # User canceled the file dialog

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Drop unnamed columns
            df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

            # **Check for format type**
            if {'X', 'Y', 'Z'}.issubset(df.columns):
                self.process_old_format(df)
            else:
                self.process_new_format(df)

        except Exception as e:
            print(f"Error loading points layer from CSV: {e}")

    def process_old_format(self, df):
        """Process CSV files in the old format (explicit X, Y, Z columns)."""
        # Ensure coordinates are numeric
        for col in ['X', 'Y', 'Z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['X', 'Y', 'Z'])

        # Extract coordinates
        points = df[['Z', 'Y', 'X']].to_numpy()

        # Extract types from 'Session # 2' column
        types = df.get('Session # 2', '-1').fillna('-1').astype(str).to_numpy()
        mapped_types = [
            'Landmark' if 'landmark' in t.lower() else 
            next((key for key in EXCITATORY_TYPE_TO_COLOR if key.startswith(t[0] + ':')), '-1: Nothing')
            for t in types
        ]

        features = {
            'label': df.get('Synapse number', 'Unknown').fillna('Unknown').astype(str).to_numpy(),
            'type': mapped_types
        }

        self.add_points_layer(points, features)


    def process_new_format(self, df):
        """Process CSV files in the new format (types embedded in column names)."""
        points_list = []
        types_list = []

        for col in df.columns:
            if col.endswith("_xpos"):
                base_name = col.replace("_xpos", "")
                xpos, ypos, zpos = f"{base_name}_xpos", f"{base_name}_ypos", f"{base_name}_zpos"

                if xpos in df and ypos in df and zpos in df:
                    valid_rows = df[[zpos, ypos, xpos]].dropna()
                    points = valid_rows.to_numpy()
                    points_list.append(points)
                    types_list.extend([base_name] * len(points))

        if points_list:
            all_points = np.vstack(points_list)
            scale_z, scale_y, scale_x = self.scale_factors
            all_points *= np.array([scale_z, scale_y, scale_x])  # Apply scaling
            all_types = np.array(types_list)
        else:
            print("No valid points found in the CSV.")
            return

        features = {
            'label': np.arange(len(all_points)).astype(str),
            'type': all_types
        }

        self.add_points_layer(all_points, features)


    def add_points_layer(self, points, features):
        """Adds a new points layer to the viewer."""
        layer_name = f"Points Layer {len(self.points_layers) + 1}"
        points_layer = self.viewer.add_points(
            points,
            features=features,
            size=7,
            edge_width=0.1,
            edge_color='white',
            face_color=TypeToColor.map_types_to_colors(features['type'], self.update_widget.mapping_name),
            text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
            name=layer_name,
        )

        # Store the new layer
        self.points_layers[layer_name] = points_layer
        self.update_widget.points_layers = self.points_layers
        self.update_widget.update_widget_for_active_layer(None)

        print(f"Loaded points layer from CSV: {layer_name}")

    # def load_points_from_csv(self):
    #     """Load points layer from a custom CSV file."""
    #     options = QFileDialog.Options()
    #     file_path, _ = QFileDialog.getOpenFileName(
    #         self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
    #     )
    #     if not file_path:
    #         return  # User canceled the file dialog

    #     try:
    #         # Read the CSV file
    #         df = pd.read_csv(file_path)
    #         # df.rename(columns={
    #         # 'CorticalSpinePSD_xpos': 'X',
    #         # 'CorticalSpinePSD_ypos': 'Y',
    #         # 'CorticalSpinePSD_zpos': 'Z'
    #         # }, inplace=True)

    #         # Filter necessary columns and handle missing data
    #         df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

    #         # Ensure 'X', 'Y', 'Z' columns are numeric and drop invalid rows
    #         for col in ['X', 'Y', 'Z']:
    #             df[col] = pd.to_numeric(df[col], errors='coerce')
    #         df = df.dropna(subset=['X', 'Y', 'Z'])

    #         # Extract the coordinates
    #         points = df[['Z', 'Y', 'X']].to_numpy()
    #         # Extract features and map to colors
    #         # Extract types and map to colors

    #         types = df['Session # 2'].fillna('-1').astype(str).to_numpy()
    #         mapped_types = []
    #         for t in types:
    #             # Check for 'landmark' and handle it separately
    #             if 'landmark' in t.lower():
    #                 mapped_types.append('Landmark')  # Adjust if necessary
    #             else:
    #                 # Find the matching excitatory type based on the first character
    #                 mapped_type = next(
    #                     (key for key in EXCITATORY_TYPE_TO_COLOR if key.startswith(t[0] + ':')),
    #                     '-1: Nothing'
    #                 )
    #                 mapped_types.append(mapped_type)

    #         # # Map types to colors
    #         # mapped_colors = TypeToColor.map_types_to_colors(mapped_types, EXCITATORY_MAPPING_NAME)


    #         # Extract features (e.g., labels)
    #         features = {
    #             'label': df['Synapse number'].fillna('Unknown').astype(str).to_numpy(),
    #             'type': mapped_types
    #         }

    #         # Add a new points layer
    #         layer_name = f"Points Layer {len(self.points_layers) + 1}"
    #         points_layer = self.viewer.add_points(
    #             points,
    #             features=features,
    #             size=7,
    #             edge_width=0.1,
    #             border_color='white',
    #             face_color= TypeToColor.map_types_to_colors(features['type'], self.update_widget.mapping_name),  # Default color for simplicity
    #             text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
    #             name=layer_name,
    #         )
    #         print(mapped_types)

    #         # Add the new layer to points_layers dictionary
    #         self.points_layers[layer_name] = points_layer

    #         # Update the UpdatePointTypeWidget with the new layer
    #         self.update_widget.points_layers = self.points_layers
    #         self.update_widget.update_widget_for_active_layer(None)  # Force update

    #         print(f"Loaded points layer from CSV: {layer_name}")
    #     except Exception as e:
    #         print(f"Error loading points layer from CSV: {e}")

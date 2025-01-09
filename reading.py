'''COMMENT OUT START HERE IF TURNING OFF CSV READ - also paste csv path '''
df = pd.read_csv(r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\savedpoints\1317F3Smoothedpointswithoutcheckingautodeletedoutsidecellfill.csv")

# Extract the coordinates
points = df[['axis-0', 'axis-1', 'axis-2']].to_numpy()

# Extract the features
features = {
    'label': df['label'].to_numpy(),
    'confidence': df['confidence'].to_numpy(),
    'type': df['type'].values if 'type' in df.columns else np.array(['Unknown'] * len(df))
}

# Set the face color cycle for the 'type' category
face_color_cycle = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'magenta']  # Customize based on your needs

# Add a points layer
points_layer = viewer.add_points(
    points,
    features=features,
    size=7,
    border_width=0.1,
    border_width_is_relative=True,
    border_color='white',
    face_color='type',
    face_color_cycle=face_color_cycle,
    text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'}
)

@magicgui(call_button="Center on Point")
def go_to_point(point_number: int):
    try:
        # Retrieve the 'label' column from the features as a pandas Series
        labels = points_layer.features['label']
        
        # Find the index of the specified label using a boolean condition
        point_index = labels[labels == point_number].index[0]
        

        # Get the coordinates of the specified point
        point_coords = points_layer.data[point_index]
        print(point_coords)
        
        # Set the viewer camera to center on the specified point
        viewer.camera.center = (point_coords[0], point_coords[1], point_coords[2])
        viewer.camera.zoom = 8
        # Move to the specific Z slice if your data is in 3D
        # viewer.dims.set_point(2, point_coords[2])  # Adjust if Z is in a different dimension
        
    except (IndexError, KeyError):
        print("Point number not found in labels or label feature missing.")


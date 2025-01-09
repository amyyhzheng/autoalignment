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

**In Vivo Analysis Code**
- Currently contains custom widgets/visualization tools for labeling/data management
- Contains normalized puncta detection across different thresholds and connected components
- MAP puncta detection
- Spline fitting for tracing visualization
- Auto alignment script to cluster and match puncta across development 

labeling
run all python files individually, then run main.py last. When making local changes to a file, run this file again and then run main.py
- constants.py: Contains Inhibitory Mapping and Excitatory Mapping. Contains initial points i.e. The [0, 0, 0] point for testing
- typetocolor.py: TypeToColor class to manage mappings
- layer_utils: Can add more. Currently only contains the consecutive number handling
- widgets.py: Contains all of the widgets.
- main.py: Run the file here. Also edit image paths here. 

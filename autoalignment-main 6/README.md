**In Vivo Analysis Code**
- Currently contains custom widgets/visualization tools for labeling/data management
- Contains normalized puncta detection across different thresholds and connected components
- MAP puncta detection
- Spline fitting for tracing visualization
- Auto alignment script to cluster and match puncta across development 

labeling
run all python files individually, then run main.py last. When making local changes to a file, run this file again and then run main.py
- constants.py: Contains Default Inhibitory Mapping and Excitatory Mapping. Contains initial points i.e. The [0, 0, 0] point for testing
- typetocolor.py: TypeToColor class to manage mappings. Used to handle all of the color matchings used. 
- layer_utils: Can add more. Handles incrementing the numbers by 1 for each new point. Handles confidence/notes editing features. Linked with each new layer. 
- widgets.py: Contains all of the widgets. 
- main.py: Run the file here. Also edit image paths here. 

**Puncta Detection**
- likely most efficient as of 1/30 to export all of the detected puncta and re upload/drag them into labeling software

**Updating Configurations**
1. Add the color mapping to typetocolor constants
2. import the mapping name and add it to the dropdown UI
3. Should be all good!


**Aygul's Colors**


EXCITATORY_TYPE_TO_COLOR = {
    '-1: Nothing': 'green',
    '0: SpinePSD95-NoSynTD': 'orange',
    '1: SpinePSD95-SynTD': 'red',
    '3: SynNudeSpine': 'blue',
    '4: NoSynNudeSpine': 'magenta', 
    'Landmark': 'cyan'
}

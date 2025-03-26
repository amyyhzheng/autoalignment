import numpy as np

# IMAGE_PATHS = [
# r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\TiffSaveinLab\2024-8-30_session2.lif - 1317#1_dob2-16-24_Ms-Gephyrin-488__Rb...Ch-GFP-568_Gp-Bassoon-647__63x_0.66zstepsize_1.46zoom__Cell1A4.tif"
# ]# 


INHIBITORY_MAPPING_NAME = "Inhibitory Map"

INHIBITORY_TYPE_TO_COLOR = {
    'Ignore': 'black',
    'ShaftGephBassoonNoSynTd': 'orange',
    'ShaftGephBassoonSynTd': 'cyan',
    'SpineGephBassoonNoSynTd': 'white',
    'SpineGephBassoonSynTd': 'blue',
    'GephNoBassoon': 'green',
    'Geph+Bassoon': 'yellow',
    'Ambiguous': 'magenta',
}

EXCITATORY_MAPPING_NAME = "Excitatory 2P"

EXCITATORY_TYPE_TO_COLOR = {
    '-1: Nothing': 'blue',
    '0: SpinePSD95-NoSynTD': 'green',
    '1: SpinePSD95-SynTD': 'yellow',
    '3: NoSynNudeSpine': 'red',
    '4: SynNudeSpine': 'purple', 
    'Landmark': 'cyan'
}

EXCITATORY_CONFOCAL_NAME = "Excitatory MAP"

EXC_CONFOCAL_TYPE_TO_COLOR = {
    'PSD95+Spine only': 'blue', 
    'PSD95+ VGlut2+ Spine': 'orange', 
    'PSD95+ VGlut1+ Spine':'cyan', 
    'Ambiguous': 'magenta'
}

# Initial points and features for all viewers
INITIAL_POINTS = np.array([[0, 0, 0]])
INITIAL_FEATURES = {
    'label': [1],
    'Notes': [1],
    'type': ['Ignore']
}

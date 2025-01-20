import numpy as np

IMAGE_PATHS = [
r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\TiffSaveinLab\2024-8-30_session2.lif - 1317#1_dob2-16-24_Ms-Gephyrin-488__Rb...Ch-GFP-568_Gp-Bassoon-647__63x_0.66zstepsize_1.46zoom__Cell1A4.tif"
]


INHIBITORY_MAPPING_NAME = "Inhibitory Map"

INHIBITORY_TYPE_TO_COLOR = {
    'Ignore': 'black',
    'ShaftGephBassoonNoSynTd': 'yellow',
    'ShaftGephBassoonSynTd': 'cyan',
    'SpineGephBassoonNoSynTd': 'green',
    'SpineGephBassoonSynTd': 'blue',
    'GephNoBassoon': 'red',
    'Ambiguous': 'magenta',
}

EXCITATORY_MAPPING_NAME = "Aygul's Map"

EXCITATORY_TYPE_TO_COLOR = {
}

# Initial points and features for all viewers
INITIAL_POINTS = np.array([[0, 0, 0]])
INITIAL_FEATURES = {
    'label': [1],
    'confidence': [1],
    'type': ['Ignore']
}

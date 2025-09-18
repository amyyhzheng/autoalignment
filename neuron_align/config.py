from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class Settings:
    inhibitory_shaft: str = "InhibitoryShaft"
    inhibitory_spine: str = "spinewithInhsynapse"
    scaling_factor: List[float] = None # [sx, sy, sz] µm per pixel/step
    num_channels: int = 4
    ojj_tif_key: str = "Image"
    snt_branch_fmt: str = "b%s" # e.g., "b1"; alt: "Path (%s)"
    animal_id: str = "SOM022"
    branch_id: str = "2"
    n_timepoints: int = 6


    branch_csvs: Dict[str, Path] = None #{"Timepoint 1": Path(...), ...}
    marker_csvs: List[Path] = None #[tp1.csv, tp2.csv, ...]
    fiducials_csv: Path = None
    export_dir: Path = None

    
    def __post_init__(self):
        if self.scaling_factor is None:
            self.scaling_factor = [0.25, 0.25, 1.0]
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class PipelineConfig:
    """Top-level configuration for the rebounding pipeline.

    Paths are relative to the project root unless absolute.
    """

    # Game-specific inputs
    video_path: str = "data/game.mp4"
    pbp_path: str = "data/game_pbp.csv"

    # Kaggle training data
    kaggle_train_pbp: str = "kaggle/train_pbp.csv"
    kaggle_train_locs: str = "kaggle/train_locs.csv"

    # Manual calibration inputs
    homography_points_path: str = "calibration/homography_points.csv"
    frame_anchors_path: str = "calibration/frame_anchors.csv"
    team_map_path: str = "calibration/team_map.csv"

    # Output locations
    models_dir: str = "models"
    output_csv: str = "outputs/game_shotquality_rebounding.csv"

    # Video parameters
    frame_rate: float = 30.0

    # How many frames before the PBP-aligned timestamp to pick as the shot frame
    shot_frame_offset: int = -3

    # Which hoop is considered the offensive hoop in the canonical coordinate system
    # Usually (4, 25) in ShotQuality / Kaggle coordinates
    hoop_x: float = 4.0
    hoop_y: float = 25.0

    # Restricted area radius in feet and paint extents
    restricted_radius: float = 4.0
    paint_x_min: float = 0.0
    paint_x_max: float = 19.0
    paint_y_min: float = 17.0
    paint_y_max: float = 33.0

    # Whether to reflect the court x/y to match Kaggle orientation
    reflect_x: bool = False
    reflect_y: bool = False

    def resolve_path(self, root: Path, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return root / p

def load_config(path: str | Path) -> PipelineConfig:
    """Load PipelineConfig from a YAML file."""
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    cfg = PipelineConfig(**raw)
    return cfg

def save_default_config(path: str | Path) -> None:
    """Write a default config YAML if you want a starting point."""
    cfg = PipelineConfig()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)

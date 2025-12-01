from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class TeamMap:
    """Mapping from track_id to team label (e.g., 'home'/'away')."""

    track_to_team: Dict[int, str]

    @classmethod
    def from_csv(cls, path: str) -> "TeamMap":
        df = pd.read_csv(path)
        if not {"track_id", "team"}.issubset(df.columns):
            raise ValueError("team_map.csv must have columns: track_id, team")
        mapping = {int(r.track_id): str(r.team) for r in df.itertuples()}
        return cls(track_to_team=mapping)

    def get(self, track_id: int, default: str | None = None) -> str | None:
        return self.track_to_team.get(track_id, default)


def choose_shooter(
    players_df: pd.DataFrame,
    offense_team: str,
    hoop_x: float,
    hoop_y: float,
    target_shot_distance: float | None = None,
) -> int:
    """Pick a shooter track_id from offensive players.

    Heuristic: choose the offensive player whose distance to the hoop
    best matches the target shot distance (if provided), otherwise the
    closest offensive player to the hoop.
    """
    off = players_df[players_df["team"] == offense_team]
    if off.empty:
        raise ValueError("No offensive players found in frame")

    dx = off["court_x"] - hoop_x
    dy = off["court_y"] - hoop_y
    dists = (dx**2 + dy**2) ** 0.5

    if target_shot_distance is not None and target_shot_distance > 0:
        score = (dists - target_shot_distance).abs()
    else:
        score = dists

    best_idx = score.idxmin()
    shooter_track_id = int(off.loc[best_idx, "track_id"])
    return shooter_track_id


def assign_roles_for_frame(
    players_df: pd.DataFrame,
    offense_team: str,
    hoop_x: float,
    hoop_y: float,
    target_shot_distance: float | None = None,
) -> pd.DataFrame:
    """Assign Kaggle-style roles (s, t1-t4, d1-d5) for a single frame.

    players_df columns expected:
      - track_id
      - team ('home'/'away' or similar)
      - court_x
      - court_y
    """
    players_df = players_df.copy()
    shooter_id = choose_shooter(players_df, offense_team, hoop_x, hoop_y, target_shot_distance)

    # Mark offense vs defense
    players_df["side"] = np.where(players_df["team"] == offense_team, "offense", "defense")

    # Assign shooter
    players_df["annotation_code"] = None
    players_df.loc[players_df["track_id"] == shooter_id, "annotation_code"] = "s"

    # Other offense: t1-t4
    off_others = players_df[(players_df["side"] == "offense") & (players_df["track_id"] != shooter_id)]
    # Sort by distance to hoop (closest first)
    off_others = off_others.copy()
    off_others["dist_hoop"] = ((off_others["court_x"] - hoop_x) ** 2 + (off_others["court_y"] - hoop_y) ** 2) ** 0.5
    off_others = off_others.sort_values("dist_hoop")
    for i, (_, row) in enumerate(off_others.iterrows()):
        code = f"t{i+1}"
        players_df.loc[players_df["track_id"] == row["track_id"], "annotation_code"] = code

    # Defense: d1-d5 sorted by distance to hoop
    defs = players_df[players_df["side"] == "defense"].copy()
    defs["dist_hoop"] = ((defs["court_x"] - hoop_x) ** 2 + (defs["court_y"] - hoop_y) ** 2) ** 0.5
    defs = defs.sort_values("dist_hoop")
    for i, (_, row) in enumerate(defs.iterrows()):
        code = f"d{i+1}"
        players_df.loc[players_df["track_id"] == row["track_id"], "annotation_code"] = code

    return players_df

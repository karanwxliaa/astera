from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class CourtGeometry:
    hoop_x: float = 4.0
    hoop_y: float = 25.0
    restricted_radius: float = 4.0
    paint_x_min: float = 0.0
    paint_x_max: float = 19.0
    paint_y_min: float = 17.0
    paint_y_max: float = 33.0


class FeatureEngineer:
    """Shot-time spatial features based on Kaggle ShotQuality competition.

    This expects Kaggle-style locs rows for each play:
      - id
      - court_x
      - court_y
      - annotation_code in {s, t1..t4, d1..d5}
    """

    def __init__(self, geom: CourtGeometry | None = None):
        self.geom = geom or CourtGeometry()

    def _dist(self, x1, y1, x2, y2):
        return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    def _is_in_paint(self, x, y) -> bool:
        g = self.geom
        return (g.paint_x_min <= x <= g.paint_x_max) and (g.paint_y_min <= y <= g.paint_y_max)

    def _is_in_restricted(self, x, y) -> bool:
        g = self.geom
        return self._dist(x, y, g.hoop_x, g.hoop_y) <= g.restricted_radius

    def build_features_for_play(self, locs_df: pd.DataFrame, play_id: str | int | None = None) -> Dict:
        """Compute a dict of features for a single play.

        locs_df should contain only rows for one play id.
        """
        g = self.geom
        feat: Dict[str, float | int | str] = {}
        if play_id is not None:
            feat["id"] = play_id

        # Ensure we have shooter
        shooter = locs_df[locs_df["annotation_code"] == "s"]
        if shooter.empty:
            # Can't build features without shooter
            return feat
        sx = float(shooter["court_x"].iloc[0])
        sy = float(shooter["court_y"].iloc[0])

        feat["shot_x"] = sx
        feat["shot_y"] = sy
        feat["shot_distance_hoop"] = self._dist(sx, sy, g.hoop_x, g.hoop_y)
        feat["shooter_in_paint"] = int(self._is_in_paint(sx, sy))
        feat["shooter_in_restricted_area"] = int(self._is_in_restricted(sx, sy))

        # Offense & defense sets
        offense_codes = {"s", "t1", "t2", "t3", "t4"}
        defense_codes = {"d1", "d2", "d3", "d4", "d5"}
        off = locs_df[locs_df["annotation_code"].isin(offense_codes)].copy()
        defs = locs_df[locs_df["annotation_code"].isin(defense_codes)].copy()

        # Distances shooter -> defenders
        def_dists = []
        for _, row in defs.iterrows():
            d = self._dist(sx, sy, float(row.court_x), float(row.court_y))
            def_dists.append(d)
        def_dists_sorted = sorted(def_dists)
        for i in range(3):
            if i < len(def_dists_sorted):
                feat[f"dist_def_{i+1}"] = def_dists_sorted[i]
            else:
                feat[f"dist_def_{i+1}"] = np.nan
        if def_dists:
            feat["avg_def_dist"] = float(np.mean(def_dists))
            # harmonic mean
            feat["harm_def_dist"] = float(len(def_dists) / np.sum(1.0 / np.array(def_dists)))
        else:
            feat["avg_def_dist"] = np.nan
            feat["harm_def_dist"] = np.nan

        # Distance-based paint / restricted counts
        off_in_paint = 0
        def_in_paint = 0
        off_in_ra = 0
        def_in_ra = 0
        off_inside8 = 0
        def_inside8 = 0
        off_positions = []
        def_positions = []

        for _, row in off.iterrows():
            x = float(row.court_x)
            y = float(row.court_y)
            off_positions.append((x, y))
            if self._is_in_paint(x, y):
                off_in_paint += 1
            if self._is_in_restricted(x, y):
                off_in_ra += 1
            if self._dist(x, y, g.hoop_x, g.hoop_y) <= 8.0:
                off_inside8 += 1

        for _, row in defs.iterrows():
            x = float(row.court_x)
            y = float(row.court_y)
            def_positions.append((x, y))
            if self._is_in_paint(x, y):
                def_in_paint += 1
            if self._is_in_restricted(x, y):
                def_in_ra += 1
            if self._dist(x, y, g.hoop_x, g.hoop_y) <= 8.0:
                def_inside8 += 1

        feat["num_off_in_paint"] = off_in_paint
        feat["num_def_in_paint"] = def_in_paint
        feat["num_off_in_restricted"] = off_in_ra
        feat["num_def_in_restricted"] = def_in_ra
        feat["num_off_inside8"] = off_inside8
        feat["num_def_inside8"] = def_inside8
        feat["off_minus_def_inside8"] = off_inside8 - def_inside8

        # Number of defenders closer to hoop than shooter
        shooter_hoop_dist = feat["shot_distance_hoop"]
        def_closer = 0
        for x, y in def_positions:
            if self._dist(x, y, g.hoop_x, g.hoop_y) < shooter_hoop_dist:
                def_closer += 1
        feat["num_def_closer_than_shooter"] = def_closer

        # Spacing: average distance between offensive players
        spacing_vals = []
        for i in range(len(off_positions)):
            for j in range(i + 1, len(off_positions)):
                x1, y1 = off_positions[i]
                x2, y2 = off_positions[j]
                spacing_vals.append(self._dist(x1, y1, x2, y2))
        if spacing_vals:
            feat["spacing_offense"] = float(np.mean(spacing_vals))
        else:
            feat["spacing_offense"] = np.nan

        # Weak side / strong side (relative to hoop_y)
        off_weak = 0
        off_strong = 0
        def_weak = 0
        def_strong = 0
        for x, y in off_positions:
            if y > g.hoop_y:
                off_weak += 1
            else:
                off_strong += 1
        for x, y in def_positions:
            if y > g.hoop_y:
                def_weak += 1
            else:
                def_strong += 1

        feat["off_weakside_count"] = off_weak
        feat["off_strongside_count"] = off_strong
        feat["def_weakside_count"] = def_weak
        feat["def_strongside_count"] = def_strong

        # Mean distance to hoop for each side
        if off_positions:
            off_dists_hoop = [self._dist(x, y, g.hoop_x, g.hoop_y) for x, y in off_positions]
            feat["mean_off_dist_hoop"] = float(np.mean(off_dists_hoop))
        else:
            feat["mean_off_dist_hoop"] = np.nan

        if def_positions:
            def_dists_hoop = [self._dist(x, y, g.hoop_x, g.hoop_y) for x, y in def_positions]
            feat["mean_def_dist_hoop"] = float(np.mean(def_dists_hoop))
        else:
            feat["mean_def_dist_hoop"] = np.nan

        return feat

    def build_feature_table(self, locs: pd.DataFrame) -> pd.DataFrame:
        """Vectorized feature computation for all plays in locs.

        locs must have at least columns: id, court_x, court_y, annotation_code.
        """
        records: List[Dict] = []
        for play_id, grp in locs.groupby("id"):
            rec = self.build_features_for_play(grp, play_id=play_id)
            if rec:
                records.append(rec)
        return pd.DataFrame(records)

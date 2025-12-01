from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class CourtHomography:
    """Mapping between pixel coordinates and court coordinates (94x50)."""

    H: np.ndarray  # 3x3 homography matrix

    @classmethod
    def from_points_csv(
        cls,
        path: str | Path,
    ) -> "CourtHomography":
        """Load calibration points from CSV and compute homography.

        CSV columns: pixel_x, pixel_y, court_x, court_y
        """
        df = pd.read_csv(path)
        src = df[["pixel_x", "pixel_y"]].to_numpy().astype(float)
        dst = df[["court_x", "court_y"]].to_numpy().astype(float)
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("Failed to compute homography from calibration points")
        return cls(H=H)

    def pixel_to_court(self, xs, ys):
        pts = np.stack([xs, ys, np.ones_like(xs)], axis=0)  # 3 x N
        mapped = self.H @ pts
        mapped /= mapped[2]
        return mapped[0], mapped[1]

    def single_pixel_to_court(self, x: float, y: float) -> Tuple[float, float]:
        xs = np.array([x], dtype=float)
        ys = np.array([y], dtype=float)
        cx, cy = self.pixel_to_court(xs, ys)
        return float(cx[0]), float(cy[0])

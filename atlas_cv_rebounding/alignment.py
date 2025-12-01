from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_clock_str(clock: str) -> float:
    """Parse a MM:SS clock string into seconds remaining in period."""
    if isinstance(clock, (int, float)):
        return float(clock)
    m, s = clock.split(":")
    return int(m) * 60 + int(s)


@dataclass
class FrameAnchors:
    """Linear mapping between PBP (period, clock) and video frame index.

    For the pilot we assume a simple linear relation per period based on
    a few manually labelled anchor points.
    """

    # mapping: (period) -> (slope, intercept) such that
    # frame_idx ≈ slope * clock_seconds + intercept
    period_params: Dict[int, Tuple[float, float]]

    @classmethod
    def from_csv(cls, path: str | Path) -> "FrameAnchors":
        """Load anchors and fit linear models per period.

        CSV columns: period, clock, frame_idx
        where clock is MM:SS as it appears on the broadcast scoreboard.
        """
        df = pd.read_csv(path)
        period_params: Dict[int, Tuple[float, float]] = {}
        for period, grp in df.groupby("period"):
            clocks = grp["clock"].apply(parse_clock_str).to_numpy()
            frames = grp["frame_idx"].to_numpy()
            if len(clocks) < 2:
                raise ValueError(
                    f"Need at least 2 anchors per period, got {len(clocks)} for period {period}"
                )
            # Fit line: frame = a * clock + b
            a, b = np.polyfit(clocks, frames, 1)
            period_params[int(period)] = (float(a), float(b))
        return cls(period_params=period_params)

    def frame_for_event(self, period: int, clock_str: str, frame_offset: int = -3) -> int:
        """Estimate frame index for an event at (period, clock)."""
        if period not in self.period_params:
            raise KeyError(f"No anchor parameters for period {period}")
        a, b = self.period_params[period]
        clock_seconds = parse_clock_str(clock_str)
        frame = a * clock_seconds + b
        return int(round(frame)) + frame_offset

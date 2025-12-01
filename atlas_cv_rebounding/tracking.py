from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]
    score: float
    last_frame: int
    history: List[Tuple[int, Tuple[float, float, float, float]]] = field(default_factory=list)


class SimpleTracker:
    """Very simple IOU-based multi-object tracker.

    This is *not* as strong as DeepSORT/ByteTrack but is easy to run
    and sufficient for a pilot project. It links detections between
    consecutive frames based on IOU overlap.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}

    def step(self, frame_idx: int, detections: List[Tuple[float, float, float, float, float]]):
        """Update tracker with detections for a single frame.

        Returns list of (frame_idx, track_id, x1, y1, x2, y2, score).
        """
        # Mark tracks as aged
        for t in self.tracks.values():
            t.last_frame = t.last_frame

        used_dets = set()
        # First, try to match to existing tracks
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = None
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                box = det[:4]
                iou_val = iou(track.box, box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = i
            if best_det_idx is not None and best_iou >= self.iou_threshold:
                # Update track
                x1, y1, x2, y2, score = detections[best_det_idx]
                track.box = (x1, y1, x2, y2)
                track.score = score
                track.last_frame = frame_idx
                track.history.append((frame_idx, track.box))
                used_dets.add(best_det_idx)
            else:
                # Age track; if too old, delete
                if frame_idx - track.last_frame > self.max_age:
                    del self.tracks[track_id]

        # Any remaining detections start new tracks
        for i, det in enumerate(detections):
            if i in used_dets:
                continue
            x1, y1, x2, y2, score = det
            track = Track(
                track_id=self.next_track_id,
                box=(x1, y1, x2, y2),
                score=score,
                last_frame=frame_idx,
                history=[(frame_idx, (x1, y1, x2, y2))],
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        # Emit current frame tracks
        outputs = []
        for track in self.tracks.values():
            x1, y1, x2, y2 = track.box
            outputs.append((frame_idx, track.track_id, x1, y1, x2, y2, track.score))
        return outputs

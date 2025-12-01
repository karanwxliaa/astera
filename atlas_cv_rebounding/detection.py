from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


class PlayerDetector:
    """YOLO-based player detector.

    This is intentionally minimal: you can swap in a Roboflow-exported
    model or a basketball-specific checkpoint by changing model_path.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.25):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. "
                "Install with `pip install ultralytics` and try again."
            )
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_players(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Return player detections as (x1, y1, x2, y2, score).

        This assumes the model has a 'person' or 'player' class you can filter on.
        For simplicity, this implementation keeps all detections.
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                detections.append((float(x1), float(y1), float(x2), float(y2), float(s)))
        return detections

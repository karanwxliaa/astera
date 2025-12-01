from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .alignment import FrameAnchors
from .config import PipelineConfig, load_config
from .detection import PlayerDetector
from .feature_engineering import CourtGeometry, FeatureEngineer
from .homography import CourtHomography
from .oreb_model import OrebModel
from .role_assignment import TeamMap, assign_roles_for_frame
from .tracking import SimpleTracker
from .video_io import VideoReader


def run_detection_and_tracking(
    video: VideoReader,
    detector: PlayerDetector,
    tracker: SimpleTracker,
    stride: int = 1,
) -> pd.DataFrame:
    """Run detector+tracker across entire video, return tracks DataFrame.

    Columns: frame_idx, track_id, x1, y1, x2, y2, score
    """
    records: List[dict] = []
    for frame_idx, frame in tqdm(
        video.iter_frames(stride=stride),
        total=video.frame_count // stride,
        desc="Detection+Tracking",
    ):
        dets = detector.detect_players(frame)
        tracks = tracker.step(frame_idx, dets)
        for fidx, tid, x1, y1, x2, y2, score in tracks:
            records.append(
                {
                    "frame_idx": fidx,
                    "track_id": tid,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": score,
                }
            )
    return pd.DataFrame.from_records(records)


def center_bottom(box):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = y2  # approximate feet position
    return cx, cy


def reconstruct_locs_for_game(
    pbp: pd.DataFrame,
    tracks: pd.DataFrame,
    team_map: TeamMap,
    anchors: FrameAnchors,
    homography: CourtHomography,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Reconstruct Kaggle-style locs (id, court_x, court_y, annotation_code)."""
    records: List[dict] = []

    for row in pbp.itertuples():
        if getattr(row, "missed_shot", 1) != 1:
            # Only consider missed shots
            continue
        period = int(row.period)
        clock_str = str(row.clock)
        frame_idx = anchors.frame_for_event(period, clock_str, frame_offset=cfg.shot_frame_offset)

        # Get tracks for this frame
        frame_tracks = tracks[tracks["frame_idx"] == frame_idx]
        if frame_tracks.empty:
            continue

        # Map track -> team label
        teams = []
        for tid in frame_tracks["track_id"]:
            teams.append(team_map.get(int(tid), None))
        frame_tracks = frame_tracks.copy()
        frame_tracks["team"] = teams

        # Drop any tracks with unknown team
        frame_tracks = frame_tracks[frame_tracks["team"].notna()]
        if frame_tracks.empty:
            continue

        # Pixel -> court mapping
        pix_xs = []
        pix_ys = []
        for _, r in frame_tracks.iterrows():
            cx, cy = center_bottom((r.x1, r.y1, r.x2, r.y2))
            pix_xs.append(cx)
            pix_ys.append(cy)
        pix_xs = np.array(pix_xs)
        pix_ys = np.array(pix_ys)
        court_xs, court_ys = homography.pixel_to_court(pix_xs, pix_ys)

        frame_tracks["court_x"] = court_xs
        frame_tracks["court_y"] = court_ys

        # Reflect if requested
        if cfg.reflect_x:
            frame_tracks["court_x"] = 94.0 - frame_tracks["court_x"]
        if cfg.reflect_y:
            frame_tracks["court_y"] = 50.0 - frame_tracks["court_y"]

        offense_team = getattr(row, "offense_team")  # expected in pbp CSV
        shot_distance = getattr(row, "shot_distance", None)

        roles_df = assign_roles_for_frame(
            frame_tracks,
            offense_team=offense_team,
            hoop_x=cfg.hoop_x,
            hoop_y=cfg.hoop_y,
            target_shot_distance=shot_distance,
        )
        # Create Kaggle-style locs rows
        play_id = getattr(row, "id", getattr(row, "play_id", None))
        for r in roles_df.itertuples():
            records.append(
                {
                    "id": play_id,
                    "court_x": float(r.court_x),
                    "court_y": float(r.court_y),
                    "annotation_code": r.annotation_code,
                }
            )

    return pd.DataFrame.from_records(records)


def run_full_pipeline(project_root: str | Path, config_path: str | Path):
    """End-to-end run: train model on Kaggle + apply to game."""
    project_root = Path(project_root)

    cfg = load_config(config_path)

    # 1. Train or load rebound model from Kaggle
    kaggle_pbp = pd.read_csv(project_root / cfg.kaggle_train_pbp)
    kaggle_locs = pd.read_csv(project_root / cfg.kaggle_train_locs)

    geom = CourtGeometry(
        hoop_x=cfg.hoop_x,
        hoop_y=cfg.hoop_y,
        restricted_radius=cfg.restricted_radius,
        paint_x_min=cfg.paint_x_min,
        paint_x_max=cfg.paint_x_max,
        paint_y_min=cfg.paint_y_min,
        paint_y_max=cfg.paint_y_max,
    )
    fe = FeatureEngineer(geom)
    kaggle_features = fe.build_feature_table(kaggle_locs)
    kaggle_full = kaggle_features.merge(kaggle_pbp[["id", "is_oreb"]], on="id")
    X_train = kaggle_full.drop(columns=["id", "is_oreb"])
    y_train = kaggle_full["is_oreb"]

    model = OrebModel.create_default()
    val_logloss, val_brier = model.fit(X_train, y_train)
    print(f"Trained OREB model - val logloss={val_logloss:.4f}, brier={val_brier:.4f}")

    model_path = project_root / cfg.models_dir / "oreb_model.pkl"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # 2. Run CV pipeline for the game
    video_path = project_root / cfg.video_path
    pbp_path = project_root / cfg.pbp_path
    anchors_path = project_root / cfg.frame_anchors_path
    homography_points_path = project_root / cfg.homography_points_path
    team_map_path = project_root / cfg.team_map_path

    pbp = pd.read_csv(pbp_path)
    anchors = FrameAnchors.from_csv(anchors_path)
    homography = CourtHomography.from_points_csv(homography_points_path)
    team_map = TeamMap.from_csv(team_map_path)

    video = VideoReader(video_path)
    detector = PlayerDetector()
    tracker = SimpleTracker()

    tracks = run_detection_and_tracking(video, detector, tracker)
    video.close()

    # 3. Reconstruct per-play locs for this game
    game_locs = reconstruct_locs_for_game(
        pbp=pbp,
        tracks=tracks,
        team_map=team_map,
        anchors=anchors,
        homography=homography,
        cfg=cfg,
    )

    # 4. Build features for game
    game_features = fe.build_feature_table(game_locs)

    # 5. Apply rebound model
    game_X = game_features.drop(columns=["id"])
    game_pred = model.predict_proba(game_X)
    game_features["pred_oreb"] = game_pred

    # 6. Merge back with PBP
    # Expect PBP id/play_id column to match feature "id"
    id_col = "id" if "id" in pbp.columns else "play_id"
    merged = pbp.merge(
        game_features,
        left_on=id_col,
        right_on="id",
        how="inner",
    )

    output_path = project_root / cfg.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Wrote final game CSV to {output_path}")

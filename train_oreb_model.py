"""Train a rebound model from Kaggle data only.

This is a convenience wrapper around the OrebModel + FeatureEngineer.
"""

from pathlib import Path

import pandas as pd

from atlas_cv_rebounding.config import load_config
from atlas_cv_rebounding.feature_engineering import CourtGeometry, FeatureEngineer
from atlas_cv_rebounding.oreb_model import OrebModel


def main(config_path: str = "config.yaml"):
    project_root = Path(".")
    cfg = load_config(config_path)

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


if __name__ == "__main__":
    main()

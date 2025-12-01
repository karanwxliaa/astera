from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split


@dataclass
class OrebModel:
    """Wrapper around a GradientBoostingClassifier for P(OREB)."""

    model: GradientBoostingClassifier

    @classmethod
    def create_default(cls) -> "OrebModel":
        gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        )
        return cls(model=gb)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """Fit model and return (val_logloss, val_brier)."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        proba_val = self.model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, proba_val)
        br = brier_score_loss(y_val, proba_val)
        return ll, br

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "OrebModel":
        model = joblib.load(path)
        return cls(model=model)

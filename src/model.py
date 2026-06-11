"""Train a 3-class XGBoost classifier on home-win / draw / away-win.

Run:
    python -m src.model
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from .features import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

LABELS = ["H", "D", "A"]  # home win, draw, away win
LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def train_test_split_temporal(df: pd.DataFrame, split_year: int = 2022):
    train = df[df["date"].dt.year < split_year]
    test = df[df["date"].dt.year >= split_year]
    return train, test


def train_model(feats: pd.DataFrame):
    feats = feats.dropna(subset=FEATURE_COLS + ["outcome"]).copy()
    feats["y"] = feats["outcome"].map(LABEL_TO_IDX)

    train, test = train_test_split_temporal(feats, split_year=2022)
    X_train, y_train = train[FEATURE_COLS], train["y"]
    X_test, y_test = test[FEATURE_COLS], test["y"]

    print(f"Train: {len(X_train):,} rows | Test (>=2022): {len(X_test):,} rows")

    print("\nBaseline: multinomial logistic regression")
    base = LogisticRegression(max_iter=2000, multi_class="multinomial")
    base.fit(X_train, y_train)
    base_proba = base.predict_proba(X_test)
    print(f"  log-loss: {log_loss(y_test, base_proba, labels=[0,1,2]):.4f}")
    print(f"  accuracy: {accuracy_score(y_test, base_proba.argmax(1)):.4f}")

    print("\nXGBoost (3-class)")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = xgb.predict_proba(X_test)
    print(f"  log-loss: {log_loss(y_test, proba, labels=[0,1,2]):.4f}")
    print(f"  accuracy: {accuracy_score(y_test, proba.argmax(1)):.4f}")

    print("\nClass distribution in test set:")
    print(pd.Series(y_test).map({v: k for k, v in LABEL_TO_IDX.items()}).value_counts())

    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    for name, imp in importance.items():
        print(f"  {name:25s} {imp:.4f}")

    MODELS.mkdir(exist_ok=True, parents=True)
    joblib.dump({"model": xgb, "features": FEATURE_COLS, "labels": LABELS},
                MODELS / "model.pkl")
    print(f"\nSaved -> {MODELS / 'model.pkl'}")
    return xgb


def main() -> None:
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    train_model(feats)


if __name__ == "__main__":
    main()

"""
Reproducible training + honest evaluation for AutoPrice.

What this script does
---------------------
1. Loads the cleaned dataset.
2. Makes ONE fixed 80/20 train/test split (random_state=42). No seed hunting.
3. Compares several models with 5-fold cross-validation on the TRAIN part only.
4. Picks the winner by cross-validated MAE (lower is better).
5. Scores the winner ONCE on the untouched test set.
6. Refits the winner on all data and saves:
       model/best_model.pkl   (the pipeline the app uses)
       model/metrics.json     (numbers the app displays)

Run from the project root:
    python train.py

The saved pipeline has the same 7 input columns and the same
ColumnTransformer(OneHotEncoder) -> estimator layout as the original
notebook model, so app.py works with it unchanged.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = next(
    (p for p in (BASE_DIR / "data" / "Cleaned_car.csv", BASE_DIR / "data" / "Cleaned_car.xls") if p.exists()), None
)
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

TARGET = "Price"
FEATURES = ["Car Name", "Year", "Distance", "Owner", "Fuel", "Drive", "Type"]  # order matters for the app
CATEGORICAL = ["Car Name", "Fuel", "Drive", "Type"]
SEED = 42


def build_pipeline(estimator, categories):
    """OneHotEncoder on categoricals, numeric columns passed through (same as the notebook)."""
    encoder = OneHotEncoder(categories=categories, sparse_output=False)
    return make_pipeline(make_column_transformer((encoder, CATEGORICAL), remainder="passthrough"), estimator)


def candidates() -> dict:
    log_target = dict(func=np.log1p, inverse_func=np.expm1)
    return {
        "Linear Regression (original)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression (log price)": TransformedTargetRegressor(regressor=LinearRegression(), **log_target),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED
        ),
        "Gradient Boosting (log price)": TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                learning_rate=0.08, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED
            ),
            **log_target,
        ),
    }


def scores(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main() -> int:
    if DATA_PATH is None:
        print("Dataset not found. Put Cleaned_car.csv in the data/ folder.")
        return 1
    df = pd.read_csv(DATA_PATH, index_col=0).dropna(subset=FEATURES + [TARGET])
    X, y = df[FEATURES], df[TARGET]
    categories = [sorted(X[c].unique()) for c in CATEGORICAL]
    print(f"Loaded {len(df):,} rows from {DATA_PATH.name}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"\n5-fold cross-validation on {len(X_train):,} training rows")
    print(f"{'Model':<34}{'R²':>8}{'MAE (₹)':>12}{'RMSE (₹)':>12}")
    results = {}
    for name, est in candidates().items():
        res = cross_validate(
            build_pipeline(est, categories), X_train, y_train, cv=cv,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
        )
        results[name] = {
            "r2": float(res["test_r2"].mean()),
            "mae": float(-res["test_neg_mean_absolute_error"].mean()),
            "rmse": float(-res["test_neg_root_mean_squared_error"].mean()),
            "r2_std": float(res["test_r2"].std()),
        }
        r = results[name]
        print(f"{name:<34}{r['r2']:>8.3f}{r['mae']:>12,.0f}{r['rmse']:>12,.0f}")

    best_name = min(results, key=lambda n: results[n]["mae"])
    print(f"\nBest by cross-validated MAE: {best_name}")

    # One-time evaluation on the untouched test set.
    best = build_pipeline(candidates()[best_name], categories).fit(X_train, y_train)
    test = scores(y_test, best.predict(X_test))
    baseline = build_pipeline(LinearRegression(), categories).fit(X_train, y_train)
    baseline_test = scores(y_test, baseline.predict(X_test))
    print(f"Test set  ->  R² {test['r2']:.3f} | MAE ₹{test['mae']:,.0f} | RMSE ₹{test['rmse']:,.0f}")
    print(f"Baseline linear regression on the same test set -> R² {baseline_test['r2']:.3f} | MAE ₹{baseline_test['mae']:,.0f}")

    # Production model: refit on ALL data (test metrics above come from the train-only fit).
    final = build_pipeline(candidates()[best_name], categories).fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(final, fh)

    metrics = {
        "model_name": best_name,
        "selected_by": "5-fold CV mean absolute error on the training split",
        "split": {"test_size": 0.2, "random_state": SEED, "n_train": len(X_train), "n_test": len(X_test)},
        "test": test,
        "baseline_linear_test": baseline_test,
        "cross_validation": results,
        "n_rows": int(len(df)),
        "sklearn_version": sklearn.__version__,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH.relative_to(BASE_DIR)} and {METRICS_PATH.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

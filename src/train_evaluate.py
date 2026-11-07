"""Compare regression models (5-fold CV on train + hold-out test) and save the best pipeline.

Metrics: R², MAE, RMSE (in ₹ crore) — the right metrics for a regression problem.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import FEATURES, MODEL_DIR, MODEL_FILE, OUTPUTS, RANDOM_STATE, TARGET, TEST_FILE, TRAIN_FILE

MODELS = {
    "Baseline (mean)": DummyRegressor(),
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                                  random_state=RANDOM_STATE),
}


def make_pipeline(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def main() -> None:
    train, test = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
    X_train, y_train, X_test, y_test = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, model in MODELS.items():
        pipe = make_pipeline(model)
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"model": name, "cv_rmse_mean": cv_rmse.mean(), "cv_rmse_std": cv_rmse.std(),
                     "test_r2": r2_score(y_test, pred), "test_mae": mean_absolute_error(y_test, pred),
                     "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                     "test_mape_pct": 100 * mean_absolute_percentage_error(y_test, pred)})
        fitted[name] = pipe
        r = rows[-1]
        print(f"{name:18s} CV RMSE {r['cv_rmse_mean']:.3f} | test R² {r['test_r2']:.3f} "
              f"MAE {r['test_mae']:.3f} RMSE {r['test_rmse']:.3f} (₹ Cr)")

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean")
    OUTPUTS.mkdir(exist_ok=True)
    results.round(4).to_csv(OUTPUTS / "model_comparison.csv", index=False)
    best = results.iloc[0]["model"]  # selected on CV, never on the test set
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_FILE)
    with open(OUTPUTS / "best_model_metrics.json", "w") as f:
        json.dump(results.iloc[0].to_dict(), f, indent=2, default=float)
    print(f"\nBest model (lowest CV RMSE): {best} -> {MODEL_FILE}")


if __name__ == "__main__":
    main()

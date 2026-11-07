"""Generate README charts from the saved model and data."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import FEATURES, IMAGES, MODEL_FILE, OUTPUTS, RANDOM_STATE, RAW_DATA, TARGET, TEST_FILE

IMAGES.mkdir(exist_ok=True)
raw, test = pd.read_csv(RAW_DATA), pd.read_csv(TEST_FILE)
model = joblib.load(MODEL_FILE)
X_test, y_test = test[FEATURES], test[TARGET]
pred = model.predict(X_test)

# 1. Price map + price by room count
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(raw.Longitude, raw.Latitude, c=raw[TARGET], cmap="viridis", s=4, alpha=0.6)
plt.colorbar(sc, ax=axes[0], label="Price (₹ crore)")
axes[0].set_title("Price by location", fontweight="bold")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
raw.boxplot(column=TARGET, by="Total_Rooms", ax=axes[1], grid=False)
axes[1].set_title("Price by total rooms", fontweight="bold"); axes[1].set_xlabel("Total rooms")
axes[1].set_ylabel("Price (₹ crore)")
fig.suptitle("")
plt.tight_layout()
plt.savefig(IMAGES / "eda_price_drivers.png", dpi=130)
plt.close()

# 2. Predicted vs actual
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, pred, s=6, alpha=0.4)
lims = [0, max(y_test.max(), pred.max()) * 1.02]
ax.plot(lims, lims, "r--", lw=1, label="perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual price (₹ crore)"); ax.set_ylabel("Predicted price (₹ crore)")
ax.set_title("Hold-out test set: predicted vs actual", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMAGES / "predicted_vs_actual.png", dpi=130)
plt.close()

# 3. Permutation importance
imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                             scoring="neg_root_mean_squared_error")
order = np.argsort(imp.importances_mean)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(np.array(FEATURES)[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#55A868")
ax.set_xlabel("Increase in RMSE when feature is shuffled (₹ crore)")
ax.set_title("Permutation feature importance", fontweight="bold")
plt.tight_layout()
plt.savefig(IMAGES / "feature_importance.png", dpi=130)
plt.close()
OUTPUTS.mkdir(exist_ok=True)
pd.DataFrame({"feature": FEATURES, "rmse_increase": imp.importances_mean}).sort_values(
    "rmse_increase", ascending=False).round(4).to_csv(OUTPUTS / "feature_importance.csv", index=False)
print("Charts saved to", IMAGES)

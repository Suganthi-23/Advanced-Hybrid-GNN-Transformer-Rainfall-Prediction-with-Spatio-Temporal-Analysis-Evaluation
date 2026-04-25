import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from model import HybridRainfallModel
from train import RainfallDataset, get_edge_index

# ============================
# CONFIGURATION
# ============================
DATA_PATH = "processed_data"

CLF_MODEL = "final_model.pth"                 # rain occurrence model
REG_MODEL = "final_model_regression.pth"      # rain amount model

SEQ_LEN = 30
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
RAIN_THRESHOLD = 0.2  # mm

# ============================
# EVALUATION
# ============================
def evaluate():
    print("Loading data...")

    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))
    A = np.load(os.path.join(DATA_PATH, "A.npy"))

    num_nodes, total_days, num_features = X_raw.shape

    # ---- Targets ----
    rainfall = X_raw[:, :, 5]                       # rainfall in mm
    rain_binary = (rainfall > RAIN_THRESHOLD).astype(np.float32)

    # ---- Normalize inputs ----
    X_min = X_raw.min(axis=(0,1), keepdims=True)
    X_max = X_raw.max(axis=(0,1), keepdims=True)
    X = (X_raw - X_min) / (X_max - X_min + 1e-6)

    # ---- Time split ----
    split = int(total_days * TRAIN_SPLIT)
    X_test = X[:, split:, :]
    rain_test = rainfall[:, split:]
    rain_bin_test = rain_binary[:, split:]

    test_dataset = RainfallDataset(X_test, rain_bin_test, SEQ_LEN)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    edge_index = get_edge_index(A)

    # ---- Load models (STRICT FIX HERE) ----
    clf = HybridRainfallModel(num_nodes, num_features, SEQ_LEN)
    clf.load_state_dict(torch.load(CLF_MODEL), strict=False)
    clf.eval()

    reg = HybridRainfallModel(num_nodes, num_features, SEQ_LEN)
    reg.load_state_dict(torch.load(REG_MODEL), strict=False)
    reg.eval()

    print("Running two-stage regression evaluation...")

    preds_all = []
    true_all = []

    with torch.no_grad():
        day_idx = 0
        for xb, yb in test_loader:
            prob_rain = clf(xb, edge_index).squeeze(-1)
            rain_amt = reg(xb, edge_index).squeeze(-1)

            # Two-stage combination
            final_pred = prob_rain * rain_amt

            preds_all.append(final_pred.numpy())
            true_all.append(rain_test[:, day_idx+SEQ_LEN : day_idx+SEQ_LEN+final_pred.shape[0]].T)

            day_idx += final_pred.shape[0]

    y_pred = np.concatenate(preds_all, axis=0).flatten()
    y_true = np.concatenate(true_all, axis=0).flatten()

    # ---- Metrics ----
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0]

    print("\n===== FINAL TWO-STAGE METRICS =====")
    print(f"RMSE (mm):        {rmse:.3f}")
    print(f"MAE (mm):         {mae:.3f}")
    print(f"R² Score:         {r2:.3f}")
    print(f"Correlation:      {corr:.3f}")

    # ---- Save metrics ----
    metrics = {
        "RMSE_mm": float(rmse),
        "MAE_mm": float(mae),
        "R2": float(r2),
        "Correlation": float(corr)
    }

    with open("two_stage_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ---- Plots ----
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.4)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.title("Two-Stage Prediction vs Actual")
    plt.tight_layout()
    plt.savefig("two_stage_prediction_vs_actual.png")

    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual (mm)")
    plt.title("Two-Stage Residual Distribution")
    plt.tight_layout()
    plt.savefig("two_stage_residuals.png")

    print("\n✔ Evaluation complete")
    print("✔ Saved:")
    print("  - two_stage_metrics.json")
    print("  - two_stage_prediction_vs_actual.png")
    print("  - two_stage_residuals.png")


if __name__ == "__main__":
    evaluate()

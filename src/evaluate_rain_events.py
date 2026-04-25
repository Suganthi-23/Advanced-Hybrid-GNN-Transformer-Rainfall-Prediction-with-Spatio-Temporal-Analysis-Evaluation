import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from model import HybridRainfallModel
from train import get_edge_index

# ============================
DATA_PATH = "processed_data"
MODEL_PATH = "final_model_regression.pth"

SEQ_LEN = 30
TRAIN_SPLIT = 0.8
RAIN_THRESHOLD = 1.0  # mm

RAIN_IDX = 5

# ============================
def evaluate():
    print("Loading data...")

    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))
    A = np.load(os.path.join(DATA_PATH, "A.npy"))

    # LOG TARGET
    Y_log = np.log1p(X_raw[:, :, RAIN_IDX])

    # Normalize inputs
    Xmin = X_raw.min(axis=(0,1), keepdims=True)
    Xmax = X_raw.max(axis=(0,1), keepdims=True)
    X = (X_raw - Xmin) / (Xmax - Xmin + 1e-6)

    split = int(X.shape[1] * TRAIN_SPLIT)
    X_test = X[:, split:]
    Y_test = Y_log[:, split:]

    inputs, targets = [], []

    for i in range(X_test.shape[1] - SEQ_LEN):
        inputs.append(X_test[:, i:i+SEQ_LEN, :])
        targets.append(Y_test[:, i+SEQ_LEN])

    inputs = torch.FloatTensor(np.array(inputs))
    targets = np.array(targets)

    edge_index = get_edge_index(A)

    model = HybridRainfallModel(
        num_nodes=X.shape[0],
        num_features=X.shape[2],
        seq_len=SEQ_LEN,
        hidden_dim=64
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("Running regression predictions...")

    with torch.no_grad():
        preds_log = model(inputs, edge_index).squeeze(-1).numpy()

    # 🔴 INVERSE LOG
    y_true = np.expm1(targets)
    y_pred = np.expm1(preds_log)

    # Rain-event masking
    mask = y_true >= RAIN_THRESHOLD
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    print(f"Rain-event samples used: {len(y_true)}")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    print("\n===== FINAL RAIN-EVENT METRICS =====")
    print(f"RMSE (mm):        {rmse:.3f}")
    print(f"MAE (mm):         {mae:.3f}")
    print(f"R² Score:         {r2:.3f}")
    print(f"Correlation:      {corr:.3f}")

    # Plots
    plt.figure(figsize=(7,7))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.title("Rain-Event Prediction vs Actual")
    plt.savefig("rain_event_prediction_vs_actual.png")

    plt.figure(figsize=(7,5))
    plt.hist(y_true - y_pred, bins=50)
    plt.title("Residual Distribution (Rain Events)")
    plt.xlabel("Error (mm)")
    plt.ylabel("Count")
    plt.savefig("rain_event_residuals.png")

    metrics = {
        "RMSE_mm": float(rmse),
        "MAE_mm": float(mae),
        "R2_score": float(r2),
        "Correlation": float(corr)
    }

    with open("rain_event_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n✔ Evaluation complete")
    print("✔ Metrics & plots saved")

# ============================
if __name__ == "__main__":
    evaluate()

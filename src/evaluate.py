import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import HybridRainfallModel
from train import RainfallDataset, get_edge_index

# ======================
# CONFIG
# ======================
DATA_PATH = "processed_data"
MODEL_PATH = "final_model_regression.pth"
SEQ_LEN = 30
BATCH_SIZE = 32
RAIN_IDX = 5   # rainfall column index
HIDDEN_DIM = 128

# ======================
# EVALUATION
# ======================
def evaluate():
    print("Loading data...")

    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))
    A = np.load(os.path.join(DATA_PATH, "A.npy"))

    # Target rainfall (mm)
    Y_raw = X_raw[:, :, RAIN_IDX]

    # Normalize inputs (same as training)
    Xmin = X_raw.min(axis=(0, 1), keepdims=True)
    Xmax = X_raw.max(axis=(0, 1), keepdims=True)
    X = (X_raw - Xmin) / (Xmax - Xmin + 1e-6)

    # Test split (last 20%)
    split = int(X.shape[1] * 0.8)
    X_test = X[:, split:, :]
    Y_test = Y_raw[:, split:]

    test_ds = RainfallDataset(X_test, Y_test, SEQ_LEN)

    edge_index = get_edge_index(A)

    device = torch.device("cpu")
    model = HybridRainfallModel(
        num_nodes=X.shape[0],
        num_features=X.shape[2],
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds = []
    trues = []

    print("Running regression evaluation...")

    with torch.no_grad():
        for xb, yb in test_ds:
            xb = xb.unsqueeze(0).to(device)
            yb = yb.numpy()

            pred = model(xb, edge_index).squeeze(0).cpu().numpy()

            preds.append(pred.flatten())
            trues.append(yb.flatten())

    preds = np.concatenate(preds).astype(np.float64)
    trues = np.concatenate(trues).astype(np.float64)

    # ======================
    # CLEAN NaNs / Infs
    # ======================
    mask = np.isfinite(preds) & np.isfinite(trues)
    preds = preds[mask]
    trues = trues[mask]

    # ======================
    # METRICS
    # ======================
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    corr = pearsonr(trues, preds)[0] if len(trues) > 1 else np.nan

    print("\n===== FINAL METRICS =====")
    print(f"RMSE (mm):        {rmse:.3f}")
    print(f"MAE (mm):         {mae:.3f}")
    print(f"R² Score:        {r2:.3f}")
    print(f"Correlation:      {corr:.3f}")

    # ======================
    # SAVE METRICS
    # ======================
    metrics = {
        "RMSE_mm": float(rmse),
        "MAE_mm": float(mae),
        "R2_score": float(r2),
        "Correlation": float(corr)
    }

    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open("evaluation_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # ======================
    # PLOTS
    # ======================

    # 1️⃣ Prediction vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, alpha=0.3)
    plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], "r--")
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png")

    # 2️⃣ Residuals
    residuals = preds - trues
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Residual Error Distribution")
    plt.tight_layout()
    plt.savefig("residuals.png")

    # 3️⃣ Time-series (sample)
    plt.figure(figsize=(12, 4))
    plt.plot(trues[:300], label="Actual")
    plt.plot(preds[:300], label="Predicted")
    plt.legend()
    plt.title("Time-Series Forecast (Sample)")
    plt.tight_layout()
    plt.savefig("timeseries_forecast.png")

    # 4️⃣ Metrics bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(["RMSE", "MAE", "R²", "Corr"], [rmse, mae, r2, corr])
    plt.title("Regression Metrics Summary")
    plt.tight_layout()
    plt.savefig("metrics_barplot.png")

    print("\n✔ Evaluation complete")
    print("✔ Saved plots:")
    print("  - prediction_vs_actual.png")
    print("  - residuals.png")
    print("  - timeseries_forecast.png")
    print("  - metrics_barplot.png")


if __name__ == "__main__":
    evaluate()

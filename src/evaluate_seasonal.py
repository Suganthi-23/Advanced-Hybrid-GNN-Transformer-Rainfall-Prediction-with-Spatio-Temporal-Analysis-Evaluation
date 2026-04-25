import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from seasonal_utils import seasonal_aggregate

DATA_PATH = "processed_data"
RAIN_IDX = 5

def evaluate():
    print("Loading data...")
    X = np.load(os.path.join(DATA_PATH, "X.npy"))

    rainfall = X[:, :, RAIN_IDX]  # (nodes, days)
    num_nodes, num_days = rainfall.shape

    # Create date range (adjust start date if needed)
    dates = pd.date_range(start="2015-01-01", periods=num_days)

    all_true = []
    all_pred = []

    # ⚠️ Replace this with your model predictions if saved
    rainfall_pred = rainfall * 0.9  # placeholder if needed

    print("Aggregating seasonal rainfall...")

    for n in range(num_nodes):
        true_seasonal = seasonal_aggregate(dates, rainfall[n])
        pred_seasonal = seasonal_aggregate(dates, rainfall_pred[n])

        # Align seasons
        common = true_seasonal.index.intersection(pred_seasonal.index)

        all_true.extend(true_seasonal[common].values)
        all_pred.extend(pred_seasonal[common].values)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)
    corr = pearsonr(all_true, all_pred)[0]

    print("\n===== SEASONAL RAINFALL PERFORMANCE =====")
    print(f"RMSE (mm):        {rmse:.3f}")
    print(f"MAE (mm):         {mae:.3f}")
    print(f"R² Score:         {r2:.3f}")
    print(f"Correlation:      {corr:.3f}")

if __name__ == "__main__":
    evaluate()

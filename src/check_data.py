import numpy as np
import os

# ============================
# DIAGNOSIS SCRIPT
# ============================
DATA_PATH = "processed_data"

def check_data():
    print("Running Forensic Data Check...")
    if not os.path.exists(os.path.join(DATA_PATH, "X.npy")):
        print("❌ Error: Processed data not found. Run preprocess_graph.py first.")
        return

    X = np.load(os.path.join(DATA_PATH, "X.npy"))
    # X Shape: (Nodes, Days, Features)
    # Feature 5 is Rainfall (Index 5)
    
    rainfall = X[:, :, 5]
    total_days = rainfall.shape[1]
    
    # Replicate the Exact Split used in Train.py
    split_idx = int(total_days * 0.8)
    
    train_data = rainfall[:, :split_idx].flatten()
    test_data  = rainfall[:, split_idx:].flatten()
    
    print("\n" + "="*40)
    print("DATA DISTRIBUTION REPORT")
    print("="*40)
    print(f"Total History: {total_days} days")
    print(f"Training (Past): {split_idx} days")
    print(f"Testing (Future): {total_days - split_idx} days")
    
    # Check for "Rain" (> 0.2mm)
    train_rain_count = np.sum(train_data > 0.2)
    test_rain_count  = np.sum(test_data > 0.2)
    
    train_pct = (train_rain_count / len(train_data)) * 100
    test_pct  = (test_rain_count / len(test_data)) * 100
    
    print("\n--- Imbalance Check ---")
    print(f"Train Set Rain: {train_rain_count} events ({train_pct:.2f}%)")
    print(f"Test Set Rain:  {test_rain_count} events  ({test_pct:.2f}%)")
    
    # Check for pure zeros (Data Loss)
    test_max = np.max(test_data)
    
    print("\n--- Verdict ---")
    if test_max == 0.0:
        print("⚠ CRITICAL FAILURE: The Test Set contains ONLY ZEROS.")
        print("Reason: The NASA download for the last 20% likely failed or is incomplete.")
        print("Solution: We must re-download data or change the Date Range.")
    elif test_pct < 1.0:
        print("⚠ VALID BUT IMBALANCED: The Test Set is real, just very dry.")
        print("Solution: DO NOT SHUFFLE. Use 'Threshold Optimization' (My Method).")
        print("This preserves the Time-Series integrity required for your PhD.")
    else:
        print("✔ Data is healthy.")

if __name__ == "__main__":
    check_data()
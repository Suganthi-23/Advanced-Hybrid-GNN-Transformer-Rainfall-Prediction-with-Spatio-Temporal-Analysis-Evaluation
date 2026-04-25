import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import HybridRainfallModel

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "processed_data"
MODEL_PATH = "final_model_regression.pth"

SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 120          # Reasonable upper bound
LR = 0.001
HIDDEN_DIM = 128
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

RAIN_IDX = 5          # PRECTOTCORR index

# =========================
# DATASET
# =========================
class RainfallDataset(Dataset):
    def __init__(self, X, Y, seq_len):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.days = X.shape[1]

    def __len__(self):
        return self.days - self.seq_len

    def __getitem__(self, idx):
        x = self.X[:, idx:idx + self.seq_len, :]
        y = self.Y[:, idx + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# =========================
# GRAPH UTILS
# =========================
def get_edge_index(A):
    rows, cols = np.where(A > 0)
    edge_array = np.array([rows, cols])
    return torch.tensor(edge_array, dtype=torch.long)

# =========================
# TRAINING
# =========================
def train():
    print("Loading data...")

    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))   # (nodes, days, features)
    A = np.load(os.path.join(DATA_PATH, "A.npy"))

    rainfall = X_raw[:, :, RAIN_IDX]

    # -------------------------
    # NORMALIZATION (FEATURES)
    # -------------------------
    X_min = X_raw.min(axis=(0, 1), keepdims=True)
    X_max = X_raw.max(axis=(0, 1), keepdims=True)
    X = (X_raw - X_min) / (X_max - X_min + 1e-6)

    # -------------------------
    # TIME-BASED SPLIT
    # -------------------------
    total_days = X.shape[1]
    train_end = int(total_days * TRAIN_SPLIT)
    val_end = int(total_days * (TRAIN_SPLIT + VAL_SPLIT))

    X_train = X[:, :train_end]
    Y_train = rainfall[:, :train_end]

    X_val = X[:, train_end:val_end]
    Y_val = rainfall[:, train_end:val_end]

    print(f"Train days: {X_train.shape[1]}")
    print(f"Val days:   {X_val.shape[1]}")

    train_ds = RainfallDataset(X_train, Y_train, SEQ_LEN)
    val_ds = RainfallDataset(X_val, Y_val, SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # MODEL
    # -------------------------
    model = HybridRainfallModel(
        num_nodes=X.shape[0],
        num_features=X.shape[2],
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    edge_index = get_edge_index(A)

    best_val_loss = float("inf")
    history = []

    print("\nStarting Training...\n")

    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        train_loss = 0.0

        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb, edge_index).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb, edge_index).squeeze(-1)
                loss = criterion(preds, yb)
                val_loss += loss.item()

        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"])
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    # -------------------------
    # SAVE TRAINING HISTORY
    # -------------------------
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n===================================")
    print("✔ Training Complete")
    print(f"✔ Best model saved: {MODEL_PATH}")
    print("✔ Training history saved: training_history.json")
    print("===================================\n")

# =========================
if __name__ == "__main__":
    train()
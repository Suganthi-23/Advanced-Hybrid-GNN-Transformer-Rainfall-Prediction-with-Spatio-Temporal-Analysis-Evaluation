import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from model import HybridRainfallModel

# ============================
# CONFIG
# ============================
SEQ_LEN = 30
BATCH = 32
EPOCHS = 80
LR = 0.001
HIDDEN = 64
TRAIN_SPLIT = 0.8

DATA_PATH = "processed_data"
MODEL_PATH = "final_model_regression.pth"

RAIN_IDX = 5  # PRECTOTCORR

# ============================
class RainfallDataset(Dataset):
    def __init__(self, X, Y, seq):
        self.X = X
        self.Y = Y
        self.seq = seq
        self.days = X.shape[1]

    def __len__(self):
        return self.days - self.seq

    def __getitem__(self, idx):
        x = self.X[:, idx:idx+self.seq, :]
        y = self.Y[:, idx+self.seq]
        return torch.FloatTensor(x), torch.FloatTensor(y)

def get_edge_index(A):
    rows, cols = np.where(A > 0)
    return torch.tensor(np.array([rows, cols]), dtype=torch.long)

# ============================
def train():
    print("Loading data...")

    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))
    A = np.load(os.path.join(DATA_PATH, "A.npy"))

    # 🔴 LOG-TRANSFORM RAINFALL TARGET
    Y_raw = np.log1p(X_raw[:, :, RAIN_IDX])

    # Normalize input features ONLY
    Xmin = X_raw.min(axis=(0,1), keepdims=True)
    Xmax = X_raw.max(axis=(0,1), keepdims=True)
    X = (X_raw - Xmin) / (Xmax - Xmin + 1e-6)

    split = int(X.shape[1] * TRAIN_SPLIT)
    Xtr, Ytr = X[:, :split], Y_raw[:, :split]

    train_ds = RainfallDataset(Xtr, Ytr, SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    edge_index = get_edge_index(A)

    model = HybridRainfallModel(
        num_nodes=X.shape[0],
        num_features=X.shape[2],
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best = float("inf")
    history = []

    print("\nStarting Training...\n")

    for e in range(EPOCHS):
        model.train()
        total = 0

        for xb, yb in train_dl:
            pred = model(xb, edge_index).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        avg = total / len(train_dl)
        history.append(avg)

        print(f"Epoch {e+1}/{EPOCHS} | Loss = {avg:.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), MODEL_PATH)

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n✔ Training complete")
    print("✔ Best model saved to:", MODEL_PATH)

# ============================
if __name__ == "__main__":
    train()

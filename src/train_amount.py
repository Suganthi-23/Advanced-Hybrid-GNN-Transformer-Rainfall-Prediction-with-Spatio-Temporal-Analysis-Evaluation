import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from model import HybridRainfallModel

# ---------------- CONFIG ----------------
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001
HIDDEN = 64
TRAIN_SPLIT = 0.8

DATA_PATH = "processed_data"
MODEL_PATH = "final_model_amount.pth"
RAIN_IDX = 5
RAIN_THRESHOLD = 0.2
# ----------------------------------------

class RainAmountDataset(Dataset):
    def __init__(self, X, Y, seq):
        self.samples = []
        for t in range(seq, X.shape[1]):
            y = Y[:, t]
            if np.any(y > RAIN_THRESHOLD):
                self.samples.append(t)

        self.X = X
        self.Y = Y
        self.seq = seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t = self.samples[idx]
        x = self.X[:, t-self.seq:t, :]
        y = self.Y[:, t]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def get_edge_index(A):
    r, c = np.where(A > 0)
    return torch.tensor([r, c], dtype=torch.long)


def train():
    print("Loading data...")
    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))
    A = np.load(os.path.join(DATA_PATH, "A.npy"))

    Y = X_raw[:, :, RAIN_IDX]

    # Normalize inputs
    Xmin = X_raw.min(axis=(0,1), keepdims=True)
    Xmax = X_raw.max(axis=(0,1), keepdims=True)
    X = (X_raw - Xmin) / (Xmax - Xmin + 1e-6)

    split = int(X.shape[1] * TRAIN_SPLIT)
    Xtr, Ytr = X[:, :split], Y[:, :split]

    ds = RainAmountDataset(Xtr, Ytr, SEQ_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridRainfallModel(
        num_nodes=X.shape[0],
        num_features=X.shape[2],
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN
    )

    edge_index = get_edge_index(A)
    optimz = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best = 1e9
    print("\nTraining rain-amount model...\n")

    for e in range(EPOCHS):
        model.train()
        total = 0

        for xb, yb in dl:
            pred = model(xb, edge_index).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimz.zero_grad()
            loss.backward()
            optimz.step()
            total += loss.item()

        avg = total / len(dl)
        print(f"Epoch {e+1}/{EPOCHS} | Loss = {avg:.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), MODEL_PATH)

    print("\n✔ Rain amount model trained and saved.")


if __name__ == "__main__":
    train()
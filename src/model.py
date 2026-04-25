import torch
import torch.nn as nn
import torch.nn.functional as F_func   # 🔧 FIX: rename functional alias

from torch_geometric.nn import GCNConv, GATv2Conv


# ===============================
# TEMPORAL LAG MODULE
# ===============================
class TemporalLaggedCorrelation(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.lag_weights = nn.Parameter(torch.randn(seq_len, seq_len))

    def forward(self, x):
        # x: (B, T, H)
        corr = torch.matmul(x, x.transpose(-2, -1))
        corr = corr * self.lag_weights
        attn = torch.softmax(corr, dim=-1)
        return torch.matmul(attn, x) + x


# ===============================
# HYBRID GRAPHFORMER MODEL
# ===============================
class HybridRainfallModel(nn.Module):
    def __init__(self, num_nodes, num_features, seq_len, hidden_dim=128):
        super().__init__()

        # SVD-like projection
        self.feature_proj = nn.Linear(num_features, hidden_dim)

        # Temporal modules
        self.lag_corr = TemporalLaggedCorrelation(seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )

        # Spatial modules
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index):
        """
        x: (B, N, T, F)
        """
        B, N, T, Fdim = x.shape   # ✅ safe variable name

        # --- Temporal ---
        x = x.view(B * N, T, Fdim)
        x = self.feature_proj(x)
        x = self.lag_corr(x)
        x = self.transformer(x)

        x = x[:, -1, :]  # last time step
        x = x.view(B, N, -1)

        outputs = []

        for i in range(B):
            h = x[i]

            gcn_out = F_func.relu(self.gcn(h, edge_index))
            gat_out = F_func.relu(self.gat(h, edge_index))

            h = gcn_out + gat_out
            out = self.regressor(h)
            outputs.append(out)

        return torch.stack(outputs, dim=0)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from model import HybridRainfallModel, TemporalLaggedCorrelation
from train import RainfallDataset, get_edge_index

SEQ_LEN = 30
HIDDEN = 64
TRAIN_SPLIT = 0.8
DATA_PATH = "processed_data"
MODEL = "final_model_regression.pth"

def extract_lag_attention(model, x_sample):
    """Extract attention from temporal lagged correlation"""
    B, N, T, F = x_sample.shape
    x = x_sample.view(B * N, T, F)
    x = model.svd_reduction(x)
    
    # Get lag correlation attention
    corr = torch.matmul(x, x.transpose(-2, -1))
    weighted = corr * model.lag_corr.lag_weights
    attn = fn.softmax(weighted, dim=-1)
    
    return attn.detach().cpu().numpy()

def extract_transformer_attention(model, x_sample):
    """Extract attention from transformer layers"""
    B, N, T, F = x_sample.shape
    x = x_sample.view(B * N, T, F)
    x = model.svd_reduction(x)
    
    # Process through lag correlation first
    x = model.lag_corr(x)
    
    # Extract attention from each transformer layer
    all_attentions = []
    for layer in model.transformer.layers:
        # Manually compute attention
        attn_layer = layer.self_attn
        x_norm = layer.norm1(x)
        
        # Compute Q, K, V
        q = attn_layer.in_proj_q(x_norm)
        k = attn_layer.in_proj_k(x_norm)
        v = attn_layer.in_proj_v(x_norm)
        
        # Reshape for multi-head attention
        batch_size, seq_len, d_model = q.shape
        num_heads = attn_layer.num_heads
        head_dim = d_model // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        attn_weights = fn.softmax(scores, dim=-1)
        
        # Average across heads
        attn_weights = attn_weights.mean(dim=1)  # Average over heads
        all_attentions.append(attn_weights.detach().cpu().numpy())
        
        # Continue forward pass
        attn_output = torch.matmul(attn_weights.unsqueeze(1), v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2))
        attn_output = attn_output.squeeze(1).contiguous().view(batch_size, seq_len, d_model)
        x = attn_layer.out_proj(attn_output)
        x = x + x_norm  # residual
        x = layer.norm2(x) + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))
    
    return all_attentions

def extract_gat_attention(model, x_sample, edge_index, station_names):
    """Extract GAT attention weights by manually computing them"""
    B, N, T, F = x_sample.shape
    x = x_sample.view(B * N, T, F)
    x = model.svd_reduction(x)
    x = model.lag_corr(x)
    x = model.transformer(x)
    x = x[:, -1, :]  # Last time step
    x = x.view(B, N, -1)
    
    h = x[0]  # First batch
    
    # Manually compute GAT attention
    # GATv2Conv uses attention mechanism
    # We'll approximate by computing similarity-based attention
    num_nodes = h.shape[0]
    
    # Compute attention scores based on node features and edge connections
    attention_scores = torch.zeros(num_nodes, num_nodes)
    
    # For each node, compute attention to its neighbors
    edge_list = edge_index.t().cpu().numpy()
    edge_dict = {}
    for edge in edge_list:
        src, dst = edge[0], edge[1]
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(dst)
    
    # Compute attention: similarity between node features
    for i in range(num_nodes):
        neighbors = edge_dict.get(i, [i])  # Include self
        for j in neighbors:
            # Compute attention score as cosine similarity
            sim = torch.cosine_similarity(h[i:i+1], h[j:j+1])
            attention_scores[i, j] = sim.item()
    
    # Normalize
    attention_scores = fn.softmax(attention_scores, dim=1)
    
    return attention_scores.detach().cpu().numpy()

def visualize_attention():
    print("Loading model and data...")
    
    # Load data
    X_raw = np.load(os.path.join(DATA_PATH, "X.npy"))
    A = np.load(os.path.join(DATA_PATH, "A.npy"))
    names = np.load(os.path.join(DATA_PATH, "station_names.npy"), allow_pickle=True)
    
    # Normalize
    Xmin = X_raw.min(axis=(0,1), keepdims=True)
    Xmax = X_raw.max(axis=(0,1), keepdims=True)
    X = (X_raw - Xmin) / (Xmax - Xmin + 1e-6)
    
    split = int(X.shape[1] * TRAIN_SPLIT)
    Xtest = X[:, split:]
    
    # Load model
    model = HybridRainfallModel(
        num_nodes=X.shape[0],
        num_features=X.shape[2],
        seq_len=SEQ_LEN,
        hidden_dim=HIDDEN
    )
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()
    
    edge_index = get_edge_index(A)
    
    # Get a sample
    ds = RainfallDataset(Xtest, Xtest[:, :, 5], SEQ_LEN)  # Using rainfall as dummy target
    x_sample, _ = ds[0]
    x_sample = x_sample.unsqueeze(0)  # Add batch dimension
    
    print("Extracting attention weights...")
    
    # Extract different types of attention
    lag_attn = extract_lag_attention(model, x_sample)
    transformer_attns = extract_transformer_attention(model, x_sample)
    gat_attn = extract_gat_attention(model, x_sample, edge_index, names)
    
    # Average lag attention across nodes
    lag_attn_avg = lag_attn.mean(axis=0)  # Average over B*N dimension
    
    print("Creating visualizations...")
    
    # 1. Temporal Lag Correlation Attention Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(lag_attn_avg, cmap='viridis', cbar_kws={'label': 'Attention Weight'})
    plt.title('Temporal Lagged Correlation Attention\n(Time Step Dependencies)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step (t)', fontsize=12)
    plt.ylabel('Time Step (t)', fontsize=12)
    plt.tight_layout()
    plt.savefig('attention_lag_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: attention_lag_correlation.png")
    
    # 2. Transformer Attention (Layer 1)
    if transformer_attns and len(transformer_attns) > 0:
        trans_attn_avg = transformer_attns[0].mean(axis=0)  # Average over batch
        plt.figure(figsize=(12, 10))
        sns.heatmap(trans_attn_avg, cmap='plasma', cbar_kws={'label': 'Attention Weight'})
        plt.title('Transformer Layer 1 Attention\n(Temporal Dependencies)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Step (Key)', fontsize=12)
        plt.ylabel('Time Step (Query)', fontsize=12)
        plt.tight_layout()
        plt.savefig('attention_transformer_layer1.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: attention_transformer_layer1.png")
        
        # Transformer Layer 2
        if len(transformer_attns) > 1:
            trans_attn_avg2 = transformer_attns[1].mean(axis=0)
            plt.figure(figsize=(12, 10))
            sns.heatmap(trans_attn_avg2, cmap='plasma', cbar_kws={'label': 'Attention Weight'})
            plt.title('Transformer Layer 2 Attention\n(Temporal Dependencies)', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step (Key)', fontsize=12)
            plt.ylabel('Time Step (Query)', fontsize=12)
            plt.tight_layout()
            plt.savefig('attention_transformer_layer2.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: attention_transformer_layer2.png")
    
    # 3. GAT Spatial Attention Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(gat_attn, 
                xticklabels=names, 
                yticklabels=names,
                cmap='coolwarm', 
                center=0,
                cbar_kws={'label': 'Attention Weight'},
                fmt='.3f')
    plt.title('GAT Spatial Attention\n(Station-to-Station Dependencies)', fontsize=14, fontweight='bold')
    plt.xlabel('Target Station', fontsize=12)
    plt.ylabel('Source Station', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('attention_gat_spatial.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: attention_gat_spatial.png")
    
    # 4. Combined Attention Summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Lag correlation
    im1 = axes[0, 0].imshow(lag_attn_avg, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Temporal Lag Correlation', fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Time Step')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Transformer Layer 1
    if transformer_attns and len(transformer_attns) > 0:
        im2 = axes[0, 1].imshow(transformer_attns[0].mean(axis=0), cmap='plasma', aspect='auto')
        axes[0, 1].set_title('Transformer Layer 1', fontweight='bold')
        axes[0, 1].set_xlabel('Time Step (Key)')
        axes[0, 1].set_ylabel('Time Step (Query)')
        plt.colorbar(im2, ax=axes[0, 1])
    
    # GAT Spatial
    im3 = axes[1, 0].imshow(gat_attn, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('GAT Spatial Attention', fontweight='bold')
    axes[1, 0].set_xlabel('Target Station')
    axes[1, 0].set_ylabel('Source Station')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_yticks(range(len(names)))
    axes[1, 0].set_yticklabels(names, fontsize=8)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Attention statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Attention Statistics:
    
    Temporal Lag Correlation:
    - Mean: {lag_attn_avg.mean():.4f}
    - Max: {lag_attn_avg.max():.4f}
    - Min: {lag_attn_avg.min():.4f}
    
    GAT Spatial:
    - Mean: {gat_attn.mean():.4f}
    - Max: {gat_attn.max():.4f}
    - Min: {gat_attn.min():.4f}
    """
    if transformer_attns and len(transformer_attns) > 0:
        stats_text += f"""
    Transformer Layer 1:
    - Mean: {transformer_attns[0].mean():.4f}
    - Max: {transformer_attns[0].max():.4f}
    - Min: {transformer_attns[0].min():.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Attention-Based Interpretability Maps', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('attention_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: attention_summary_dashboard.png")
    
    print("\n✔ All attention visualizations saved!\n")

if __name__ == "__main__":
    visualize_attention()

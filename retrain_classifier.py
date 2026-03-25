#!/usr/bin/env python3
"""Retrain the attack classifier MLP on embeddings from the CURRENT GNN encoder.

This fixes the mismatch where attack_classifier.pt was trained on a different
graph construction pipeline than what app/graph_builder.py produces at inference.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from collections import Counter
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split

# Force unbuffered output
def log(msg):
    print(msg, flush=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import (
    CHECKPOINT_STAGE1, SCALER_PATH, FEATURE_COLS_PATH,
    CLASSIFIER_PATH, META_COLS, LABEL_COL, NUM_CLASSES, ID_TO_ATTACK
)
from app.models import GATEncoderWrapper
from app.graph_builder import GraphBuilder


class AttackClassifier(nn.Module):
    """Must match app/inference_engine.py exactly."""
    def __init__(self, input_dim=128, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def load_gnn(device):
    """Load the current GNN encoder."""
    log(f"[1/4] Loading GNN from {CHECKPOINT_STAGE1}...")
    ckpt = torch.load(CHECKPOINT_STAGE1, map_location='cpu', weights_only=False)
    model = GATEncoderWrapper.from_checkpoint(ckpt, device=str(device))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"  -> GNN loaded (epoch {ckpt.get('epoch', '?')}, device={device})")
    return model


def build_graphs_from_csv(csv_path, feature_cols, scaler, max_rows=None, skip_rows=0):
    """Build graphs from a CSV file using the same pipeline as app/."""
    t0 = time.time()
    log(f"  Reading CSV (skip={skip_rows}, rows={max_rows})...")
    skiprows = range(1, skip_rows + 1) if skip_rows > 0 else None
    try:
        df = pd.read_csv(csv_path, encoding='latin-1', nrows=max_rows,
                         skiprows=skiprows, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1', encoding_errors='replace',
                         nrows=max_rows, skiprows=skiprows, low_memory=False)

    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    log(f"  Read {len(df)} rows in {time.time()-t0:.1f}s. Building graphs...")

    t1 = time.time()
    builder = GraphBuilder(feature_cols=feature_cols, scaler=scaler)
    graphs = builder.add_flows(df)
    log(f"  -> {len(graphs)} graphs built in {time.time()-t1:.1f}s")
    return graphs


@torch.no_grad()
def extract_embeddings(gnn, graphs, device):
    """Extract 128-dim graph embeddings using the current GNN."""
    log(f"[3/4] Extracting embeddings from {len(graphs)} graphs...")
    embeddings = []
    labels = []
    for i, (pyg_data, nx_graph, metadata) in enumerate(graphs):
        batch = Batch.from_data_list([pyg_data]).to(device)
        node_emb = gnn.encode(batch.x, batch.edge_index)
        graph_emb = global_mean_pool(node_emb, batch.batch)
        embeddings.append(graph_emb.cpu())

        label = pyg_data.y_attack.item() if hasattr(pyg_data, 'y_attack') else 0
        labels.append(label)

        if (i + 1) % 100 == 0:
            log(f"  Embedded {i+1}/{len(graphs)} graphs...")

    if not embeddings:
        return None, None
    return torch.cat(embeddings, dim=0), torch.tensor(labels, dtype=torch.long)


def train_classifier(embeddings, labels, device, epochs=100, lr=1e-3):
    """Train the MLP classifier on extracted embeddings."""
    log(f"[4/4] Training classifier (epochs={epochs}, lr={lr})...")

    # Compute normalization stats
    emb_mean = embeddings.mean(dim=0)
    emb_std = embeddings.std(dim=0).clamp(min=1e-6)
    norm_embs = (embeddings - emb_mean) / emb_std

    # Split
    indices = np.arange(len(labels))
    label_counts = Counter(labels.numpy().tolist())
    min_count = min(label_counts.values())
    use_stratify = min_count >= 3

    if use_stratify:
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels.numpy(), random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=labels[train_idx].numpy(), random_state=42)
    else:
        log(f"  WARNING: min class count={min_count}, using random split")
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    X_train = norm_embs[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    X_val = norm_embs[val_idx].to(device)
    y_val = labels[val_idx].to(device)
    X_test = norm_embs[test_idx].to(device)
    y_test = labels[test_idx].to(device)

    log(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    log(f"  Train labels: {Counter(y_train.cpu().numpy().tolist())}")
    log(f"  Test labels:  {Counter(y_test.cpu().numpy().tolist())}")

    input_dim = embeddings.shape[1]
    model = AttackClassifier(input_dim=input_dim, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Class weights for imbalanced data
    class_counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    class_weights = 1.0 / (class_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    log(f"  Class weights: {class_weights.tolist()}")

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            train_acc = (logits.argmax(1) == y_train).float().mean().item()
            log(f"  Epoch {epoch+1:3d} | loss={loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Final test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test).float().mean().item()

    log(f"\n  Best val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")
    for cls_id in range(NUM_CLASSES):
        mask = y_test == cls_id
        if mask.sum() > 0:
            cls_acc = (test_preds[mask] == y_test[mask]).float().mean().item()
            log(f"    {ID_TO_ATTACK.get(cls_id, '?'):12s} ({cls_id}): {cls_acc:.3f} ({mask.sum().item()} samples)")

    return model, best_val_acc, test_acc, emb_mean, emb_std


def main():
    t_start = time.time()
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log(f"Device: {device}")

    gnn = load_gnn(device)

    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    feature_cols = joblib.load(FEATURE_COLS_PATH) if os.path.exists(FEATURE_COLS_PATH) else None
    log(f"Scaler: {'loaded' if scaler else 'MISSING'}")
    log(f"Feature cols: {len(feature_cols) if feature_cols else 'MISSING'}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Target densest attack regions based on row-range analysis
    csv_configs = [
        # DoS: Hulk at [74860-331032], slowloris [6558-69273], Slowhttp [69275-74859]
        ("datasets/csv/Wednesday-workingHours.pcap_ISCX.csv", 6500, 70000),     # DoS slowloris+Slowhttp
        ("datasets/csv/Wednesday-workingHours.pcap_ISCX.csv", 74000, 100000),   # DoS Hulk dense region
        ("datasets/csv/Wednesday-workingHours.pcap_ISCX.csv", 200000, 100000),  # More DoS Hulk
        ("datasets/csv/Wednesday-workingHours.pcap_ISCX.csv", 330000, 50000),   # DoS GoldenEye
        # DDoS: [18883-197542]
        ("datasets/csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 18000, 100000),  # DDoS dense
        ("datasets/csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 120000, 80000),  # More DDoS
        # PortScan: [1463-279088]
        ("datasets/csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 1000, 100000),  # PortScan
        ("datasets/csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 150000, 100000), # More PortScan
        # WebAttack: BruteForce [12637-62373], XSS [72134-86077]
        ("datasets/csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 12000, 80000), # WebAttack region
        # BruteForce: FTP [11347-161987], SSH [161989-445817]
        ("datasets/csv/Tuesday-WorkingHours.pcap_ISCX.csv", 11000, 80000),      # FTP-Patator
        ("datasets/csv/Tuesday-WorkingHours.pcap_ISCX.csv", 160000, 80000),     # SSH-Patator
        # Bot: [24072-191022] (sparse, only 1966 rows in 167k)
        ("datasets/csv/Friday-WorkingHours-Morning.pcap_ISCX.csv", 24000, 100000), # Bot region
        # Benign baseline
        ("datasets/csv/Monday-WorkingHours.pcap_ISCX.csv", 0, 30000),           # Pure Benign
        ("datasets/csv/Wednesday-workingHours.pcap_ISCX.csv", 0, 6000),         # Benign before attacks
    ]

    log(f"\n[2/4] Building graphs from {len(csv_configs)} CSV sources...")
    all_graphs = []
    for i, (csv_rel, skip, nrows) in enumerate(csv_configs):
        csv_path = os.path.join(base_dir, csv_rel)
        if not os.path.exists(csv_path):
            log(f"  SKIP: {csv_rel} not found")
            continue
        log(f"\n  [{i+1}/{len(csv_configs)}] {os.path.basename(csv_rel)}")
        graphs = build_graphs_from_csv(csv_path, feature_cols, scaler,
                                        max_rows=nrows, skip_rows=skip)
        all_graphs.extend(graphs)

    log(f"\nTotal graphs: {len(all_graphs)}")

    if len(all_graphs) < 20:
        log("ERROR: Too few graphs. Check CSV paths.")
        sys.exit(1)

    # Extract embeddings
    embeddings, labels = extract_embeddings(gnn, all_graphs, device)
    log(f"Embeddings shape: {embeddings.shape}")
    log(f"Label distribution: {dict(Counter(labels.numpy().tolist()))}")
    log(f"Label names: {', '.join(f'{ID_TO_ATTACK[k]}={v}' for k, v in sorted(Counter(labels.numpy().tolist()).items()))}")

    # Train
    log("\n" + "=" * 60)
    model, val_acc, test_acc, emb_mean, emb_std = train_classifier(
        embeddings, labels, device, epochs=150, lr=1e-3
    )

    # Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': embeddings.shape[1],
        'num_classes': NUM_CLASSES,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'emb_mean': emb_mean.cpu(),
        'emb_std': emb_std.cpu(),
    }
    torch.save(checkpoint, CLASSIFIER_PATH)
    elapsed = time.time() - t_start
    log(f"\nDONE in {elapsed:.0f}s. Saved to {CLASSIFIER_PATH}")
    log(f"  val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
    log(f"  emb_mean/emb_std included for inference normalization")


if __name__ == '__main__':
    main()

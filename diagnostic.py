#!/usr/bin/env python3
"""End-to-end diagnostic: why simulated attack pcaps are classified as Benign."""

import sys
import os
import glob
import pickle
import joblib
import warnings
import subprocess
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 30)

PROJECT = '/Users/adityas/sem8/Project2-NetworkFlow'
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from app.pcap_processor import process_pcap, _docker_available
from app.config import ATTACK_LABEL_MAP, ID_TO_ATTACK

# ============================================================
# STEP 1: Process simulated pcap
# ============================================================
print("=" * 80)
print("STEP 1: Process simulated pcap through pcap_processor")
print("=" * 80)

pcap_files = glob.glob(os.path.join(PROJECT, 'wireshark', 'dos_hulk_sim*'))
if not pcap_files:
    print("ERROR: No dos_hulk_sim pcap found in wireshark/")
    sys.exit(1)

pcap_path = pcap_files[0]
print(f"Processing: {pcap_path}")
df = process_pcap(pcap_path)

print(f"\nNumber of rows (flows): {len(df)}")
print(f"\nColumn names ({len(df.columns)} total):")
print(list(df.columns))

key_cols = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow IAT Mean',
            'Flow Bytes/s', 'SYN Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'Source IP', 'Destination IP', 'Destination Port', 'Label']

available_keys = [c for c in key_cols if c in df.columns]
missing_keys = [c for c in key_cols if c not in df.columns]
if missing_keys:
    print(f"\nMISSING key columns: {missing_keys}")

print(f"\nFirst 5 rows of key features:")
print(df[available_keys].head(5).to_string())

# Separate attacker vs benign IPs
from app.config import META_COLS, LABEL_COL
feature_cols_all = [c for c in df.columns if c not in META_COLS + [LABEL_COL, 'Flow ID']]
numeric_cols = [c for c in feature_cols_all if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

attacker_mask = df['Source IP'].str.startswith('172.16.0.')
benign_mask = ~attacker_mask

print(f"\nAttacker IPs (172.16.0.x): {attacker_mask.sum()} flows")
print(f"Other IPs: {benign_mask.sum()} flows")
print(f"Unique Source IPs: {df['Source IP'].unique()}")
print(f"Unique Dest IPs: {df['Destination IP'].unique()}")

if attacker_mask.sum() > 0:
    print(f"\n--- Attacker flows summary stats (mean) ---")
    att_stats = df.loc[attacker_mask, numeric_cols].describe().loc[['mean', '50%', 'std']]
    print(att_stats.to_string())

if benign_mask.sum() > 0:
    print(f"\n--- Benign/other flows summary stats (mean) ---")
    ben_stats = df.loc[benign_mask, numeric_cols].describe().loc[['mean', '50%', 'std']]
    print(ben_stats.to_string())

# ============================================================
# STEP 2: Compare with CIC-IDS2017
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Compare with CIC-IDS2017 DoS Hulk")
print("=" * 80)

cic_csv = os.path.join(PROJECT, 'datasets/csv/Wednesday-workingHours.pcap_ISCX.csv')
if os.path.exists(cic_csv):
    cic_df = pd.read_csv(cic_csv)
    cic_df.columns = cic_df.columns.str.strip()

    hulk_mask = cic_df['Label'].str.strip() == 'DoS Hulk'
    cic_hulk = cic_df[hulk_mask]
    print(f"CIC-IDS2017 DoS Hulk rows: {len(cic_hulk)}")

    # Get matching numeric cols
    cic_numeric = [c for c in numeric_cols if c in cic_hulk.columns]
    cic_hulk_clean = cic_hulk[cic_numeric].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n--- CIC-IDS2017 DoS Hulk summary stats ---")
    cic_stats = cic_hulk_clean.describe().loc[['mean', '50%', 'std']]
    print(cic_stats.to_string())
else:
    print(f"ERROR: CIC-IDS2017 CSV not found at {cic_csv}")
    cic_hulk = None
    cic_numeric = []

# ============================================================
# STEP 3: Feature-by-feature comparison (top 20)
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: Feature-by-feature comparison (top 20 by difference)")
print("=" * 80)

if cic_hulk is not None and attacker_mask.sum() > 0:
    sim_means = df.loc[attacker_mask, numeric_cols].mean()
    cic_means = cic_hulk_clean[cic_numeric].mean()

    # Align on common features
    common = sorted(set(sim_means.index) & set(cic_means.index))

    diffs = []
    for feat in common:
        s = sim_means[feat]
        c = cic_means[feat]
        abs_diff = abs(s - c)
        # Relative ratio
        if c != 0:
            ratio = s / c
        elif s != 0:
            ratio = float('inf')
        else:
            ratio = 1.0
        wildly_off = abs(ratio - 1.0) > 2.0 if ratio != float('inf') else True
        diffs.append((feat, s, c, abs_diff, ratio, wildly_off))

    diffs.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'Feature':<35} {'Sim Mean':>15} {'CIC Mean':>15} {'Ratio':>10} {'Off?':>6}")
    print("-" * 85)
    for feat, s, c, ad, ratio, off in diffs[:20]:
        ratio_str = f"{ratio:.3f}" if ratio != float('inf') else "INF"
        off_str = "YES" if off else "no"
        print(f"{feat:<35} {s:>15.2f} {c:>15.2f} {ratio_str:>10} {off_str:>6}")

    # Also show key discriminating features
    print("\n--- Key discriminating features specifically ---")
    key_feats = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                 'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow IAT Mean',
                 'Flow Bytes/s', 'SYN Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                 'Init_Win_bytes_forward', 'Init_Win_bytes_backward']
    for feat in key_feats:
        if feat in sim_means.index and feat in cic_means.index:
            s = sim_means[feat]
            c = cic_means[feat]
            ratio = s / c if c != 0 else float('inf')
            ratio_str = f"{ratio:.3f}" if ratio != float('inf') else "INF"
            print(f"  {feat:<35} sim={s:>15.2f}  cic={c:>15.2f}  ratio={ratio_str}")

# ============================================================
# STEP 4: Check which CICFlowMeter was used
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: CICFlowMeter backend check")
print("=" * 80)

docker_ok = _docker_available()
print(f"Docker available: {docker_ok}")
try:
    r = subprocess.run(['docker', 'info'], capture_output=True, timeout=5)
    print(f"Docker daemon running: {r.returncode == 0}")
except Exception as e:
    print(f"Docker check error: {e}")

if docker_ok:
    print("=> Java CICFlowMeter (Docker) was used")
else:
    print("=> Python cicflowmeter fallback was used")

# ============================================================
# STEP 5: Build graph and check structure
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: Build graph and check structure")
print("=" * 80)

# Load feature cols and scaler from checkpoints (pickle required by project)
with open(os.path.join(PROJECT, 'checkpoints/stage1/feature_cols.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)
scaler = joblib.load(os.path.join(PROJECT, 'checkpoints/stage1/scaler.pkl'))

print(f"Feature cols ({len(feature_cols)}): {feature_cols[:10]}...")
print(f"Scaler type: {type(scaler).__name__}")
print(f"Scaler mean (first 10): {scaler.mean_[:10]}")
print(f"Scaler scale (first 10): {scaler.scale_[:10]}")

# Check if all feature_cols are present in df
missing_feat = [c for c in feature_cols if c not in df.columns]
extra_feat = [c for c in df.columns if c not in feature_cols and c not in META_COLS + [LABEL_COL, 'Flow ID']]
print(f"\nFeature cols missing from df: {missing_feat}")
print(f"Extra cols in df not in feature_cols: {extra_feat}")

from app.graph_builder import GraphBuilder

gb = GraphBuilder(feature_cols=feature_cols, scaler=scaler)
results = gb.add_flows(df)

print(f"\nGraphs produced: {len(results)}")

for i, (pyg_data, G, metadata) in enumerate(results):
    print(f"\n--- Graph {i} ---")
    print(f"  Metadata: {metadata}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    node_list = list(G.nodes())
    print(f"  Node IPs (first 20): {node_list[:20]}")

    # Attacker node features
    attacker_nodes = [n for n in node_list if str(n).startswith('172.16.0.')]
    victim_nodes = [n for n in node_list if str(n) == '192.168.10.50']

    for an in attacker_nodes:
        feat = G.nodes[an]['features']
        lbl = G.nodes[an].get('label', 'N/A')
        print(f"  Attacker {an} (label={lbl}): features[:10] = {feat[:10]}")

    for vn in victim_nodes:
        feat = G.nodes[vn]['features']
        lbl = G.nodes[vn].get('label', 'N/A')
        print(f"  Victim {vn} (label={lbl}): features[:10] = {feat[:10]}")

    # If no standard attacker/victim, show all nodes
    if not attacker_nodes and not victim_nodes:
        print("  WARNING: No 172.16.0.x or 192.168.10.50 nodes found!")
        for n in node_list[:10]:
            feat = G.nodes[n]['features']
            lbl = G.nodes[n].get('label', 'N/A')
            print(f"  Node {n} (label={lbl}): features[:10] = {feat[:10]}")

    print(f"  PyG data: x.shape={pyg_data.x.shape}, edge_index.shape={pyg_data.edge_index.shape}")
    print(f"  Graph label (y_attack): {pyg_data.y_attack.item()} => {ID_TO_ATTACK.get(pyg_data.y_attack.item(), '?')}")
    print(f"  Node attack labels: {pyg_data.node_attack_labels.tolist()}")
    print(f"  Node attack label distribution: {dict(zip(*np.unique(pyg_data.node_attack_labels.numpy(), return_counts=True)))}")

# ============================================================
# STEP 6: Run inference
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: Run inference")
print("=" * 80)

from app.inference_engine import InferenceEngine

engine = InferenceEngine()
engine.load_models()

for i, (pyg_data, G, metadata) in enumerate(results):
    print(f"\n--- Inference for Graph {i} ---")

    # Attack prediction
    pred_class, probs = engine.get_attack_prediction(pyg_data)
    print(f"  Predicted class: {pred_class} => {ID_TO_ATTACK.get(pred_class, '?')}")
    print(f"  Class probabilities:")
    for cls_id in range(len(probs)):
        print(f"    {cls_id} ({ID_TO_ATTACK.get(cls_id, '?'):>12}): {probs[cls_id]:.6f}")

    # GNN embedding
    emb = engine.get_graph_embedding(pyg_data)
    print(f"  GNN embedding (128-dim), first 10: {emb[:10]}")
    print(f"  Embedding stats: mean={emb.mean():.4f}, std={emb.std():.4f}, min={emb.min():.4f}, max={emb.max():.4f}")

    # Normalized embedding (what classifier sees)
    if engine._emb_mean is not None:
        import torch
        from torch_geometric.data import Batch
        from torch_geometric.nn import global_mean_pool
        batch = Batch.from_data_list([pyg_data]).to(engine.device)
        with torch.no_grad():
            node_emb = engine.gnn_model.encode(batch.x, batch.edge_index)
            graph_emb = global_mean_pool(node_emb, batch.batch)
            norm_emb = (graph_emb - engine._emb_mean) / engine._emb_std
        print(f"  Normalized embedding first 10: {norm_emb.cpu().numpy()[0][:10]}")
        print(f"  Norm emb stats: mean={norm_emb.cpu().numpy()[0].mean():.4f}, std={norm_emb.cpu().numpy()[0].std():.4f}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

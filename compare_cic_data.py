#!/usr/bin/env python3
"""Compare CIC-IDS2017 CSV data flow through graph builder.

Uses pickle to load project checkpoints (feature_cols.pkl, scaler.pkl)
as explicitly requested by the user for this analysis task.
"""

import sys
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/adityas/sem8/Project2-NetworkFlow')
from app.graph_builder import GraphBuilder
from app.config import META_COLS, LABEL_COL, ATTACK_LABEL_MAP, ID_TO_ATTACK

# ============================================================
# STEP 1: Load CIC-IDS2017 Wednesday CSV
# ============================================================
print("=" * 70)
print("STEP 1: Loading CIC-IDS2017 Wednesday CSV")
print("=" * 70)

df = pd.read_csv('/Users/adityas/sem8/Project2-NetworkFlow/datasets/csv/Wednesday-workingHours.pcap_ISCX.csv')
df.columns = df.columns.str.strip()
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns[:10])} ... ({len(df.columns)} total)")
print(f"\nLabel distribution:")
print(df[LABEL_COL].value_counts())

# ============================================================
# STEP 2: Find a 30-second window with DoS Hulk traffic
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Finding 30-second window with DoS Hulk traffic")
print("=" * 70)

# Parse timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
df = df.dropna(subset=['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

# Filter to DoS Hulk flows
hulk_df = df[df[LABEL_COL] == 'DoS Hulk']
print(f"Total DoS Hulk flows: {len(hulk_df)}")
print(f"DoS Hulk time range: {hulk_df['Timestamp'].min()} to {hulk_df['Timestamp'].max()}")

# Pick a window starting from the first DoS Hulk flow
hulk_start = hulk_df['Timestamp'].iloc[0]
window_start = hulk_start
window_end = window_start + pd.Timedelta(seconds=30)

window_df = df[(df['Timestamp'] >= window_start) & (df['Timestamp'] < window_end)].copy()

print(f"\nSelected window: {window_start} to {window_end}")
print(f"Number of total flows: {len(window_df)}")
benign_count = (window_df[LABEL_COL] == 'BENIGN').sum()
attack_count = (window_df[LABEL_COL] != 'BENIGN').sum()
print(f"Benign flows: {benign_count}")
print(f"Attack flows: {attack_count}")
print(f"Unique Source IPs: {window_df['Source IP'].nunique()}")
print(f"Unique Destination IPs: {window_df['Destination IP'].nunique()}")
print(f"\nLabel distribution in window:")
print(window_df[LABEL_COL].value_counts())

print(f"\nSource IPs: {window_df['Source IP'].unique()}")
print(f"Destination IPs: {window_df['Destination IP'].unique()}")

# ============================================================
# STEP 3: Build graph from that window
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Building graph from window")
print("=" * 70)

# Load trusted project checkpoints (pickle)
with open('/Users/adityas/sem8/Project2-NetworkFlow/checkpoints/stage1/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
scaler = joblib.load('/Users/adityas/sem8/Project2-NetworkFlow/checkpoints/stage1/scaler.pkl')

print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]} ...")

# Replace inf values before feeding to graph builder
for col in feature_cols:
    if col in window_df.columns:
        window_df[col] = window_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

gb = GraphBuilder(feature_cols=feature_cols, scaler=scaler, window_size=30, stride=10)
results = gb.add_flows(window_df)

print(f"Number of graphs built: {len(results)}")

if results:
    pyg_data, nx_graph, metadata = results[0]
    print(f"\nGraph metadata: {metadata}")
    print(f"Number of nodes: {nx_graph.number_of_nodes()}")
    print(f"Number of edges: {nx_graph.number_of_edges()}")
    print(f"Node list: {list(nx_graph.nodes())}")

    # Node feature vectors (AFTER scaler normalization)
    attacker_ip = '172.16.0.1'
    victim_ip = '192.168.10.50'

    for ip in [attacker_ip, victim_ip]:
        if ip in nx_graph.nodes():
            feat = nx_graph.nodes[ip]['features']
            print(f"\nNode feature vector for {ip} (first 20 values, AFTER scaler):")
            print(f"  {feat[:20]}")
            print(f"  Shape: {feat.shape}")
        else:
            print(f"\n  WARNING: {ip} not found in graph. Available nodes: {list(nx_graph.nodes())}")

    # Graph-level label
    graph_label = pyg_data.y_attack.item()
    print(f"\nGraph-level label: {graph_label} ({ID_TO_ATTACK.get(graph_label, 'Unknown')})")

    # ============================================================
    # STEP 4: Run inference
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Running inference on CIC-IDS2017 graph")
    print("=" * 70)

    from app.inference_engine import InferenceEngine
    engine = InferenceEngine()
    engine.load_models()

    pred_class, probs = engine.get_attack_prediction(pyg_data)
    print(f"\nPredicted class: {pred_class} ({ID_TO_ATTACK.get(pred_class, 'Unknown')})")
    print(f"\nAll 7 class probabilities:")
    for i in range(len(probs)):
        print(f"  Class {i} ({ID_TO_ATTACK.get(i, 'Unknown'):>12s}): {probs[i]:.6f}")

    # ============================================================
    # STEP 5: Compare raw feature values
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Raw feature values for attacker node (172.16.0.1)")
    print("=" * 70)

    # Get raw (pre-scaler) features for attacker node
    # Re-build graph without scaler to get raw features
    gb_raw = GraphBuilder(feature_cols=feature_cols, scaler=None, window_size=30, stride=10)
    results_raw = gb_raw.add_flows(window_df)

    if results_raw:
        _, nx_raw, _ = results_raw[0]

        features_of_interest = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
            'Flow IAT Mean', 'Flow IAT Std',
            'SYN Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'Flow Bytes/s', 'Flow Packets/s',
        ]

        for ip in [attacker_ip, victim_ip]:
            if ip in nx_raw.nodes():
                feat_raw = nx_raw.nodes[ip]['features']
                print(f"\nRAW features for {ip} (mean of outgoing edge features):")
                for fname in features_of_interest:
                    if fname in feature_cols:
                        idx = feature_cols.index(fname)
                        print(f"  {fname:>35s}: {feat_raw[idx]:>15.4f}")
                    else:
                        # Try case-insensitive match
                        matches = [fc for fc in feature_cols if fc.lower() == fname.lower()]
                        if matches:
                            idx = feature_cols.index(matches[0])
                            print(f"  {matches[0]:>35s}: {feat_raw[idx]:>15.4f}")
                        else:
                            print(f"  {fname:>35s}: NOT FOUND in feature_cols")
            else:
                print(f"\n  WARNING: {ip} not in raw graph")

        # Also print all raw edge features for a sample edge from attacker
        print(f"\n--- Sample individual flow features from attacker ({attacker_ip}) ---")
        out_edges = list(nx_raw.out_edges(attacker_ip))
        if out_edges:
            sample_edge = out_edges[0]
            edge_feat = nx_raw.edges[sample_edge]['features']
            print(f"Edge: {sample_edge[0]} -> {sample_edge[1]}")
            for fname in features_of_interest:
                if fname in feature_cols:
                    idx = feature_cols.index(fname)
                    print(f"  {fname:>35s}: {edge_feat[idx]:>15.4f}")
                else:
                    matches = [fc for fc in feature_cols if fc.lower() == fname.lower()]
                    if matches:
                        idx = feature_cols.index(matches[0])
                        print(f"  {matches[0]:>35s}: {edge_feat[idx]:>15.4f}")
        print(f"\nTotal outgoing edges from {attacker_ip}: {len(out_edges)}")
else:
    print("ERROR: No graphs were built! Checking why...")
    print(f"Window has {len(window_df)} flows")
    print(f"Unique IPs: {window_df['Source IP'].nunique()} src, {window_df['Destination IP'].nunique()} dst")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

#!/usr/bin/env python3
"""Comprehensive feature comparison between simulated attack pcaps and CIC-IDS2017 dataset."""

import sys
import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/adityas/sem8/Project2-NetworkFlow')
from app.pcap_processor import process_pcap

BASE = '/Users/adityas/sem8/Project2-NetworkFlow'
CSV_DIR = os.path.join(BASE, 'datasets/csv')

# Load feature columns (this pickle is from our own training pipeline, safe to load)
with open(os.path.join(BASE, 'checkpoints/stage1/feature_cols.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

print(f"Number of feature columns: {len(feature_cols)}")
print(f"Feature columns: {feature_cols[:10]}...")
print()

# ============================================================
# STEP 1: Process each simulated pcap
# ============================================================
pcap_configs = {
    'DoS Hulk':      glob.glob(os.path.join(BASE, 'wireshark/dos_hulk_sim*.pcap'))[0],
    'DoS GoldenEye': glob.glob(os.path.join(BASE, 'wireshark/dos_goldeneye_sim*.pcap'))[0],
    'DoS slowloris': glob.glob(os.path.join(BASE, 'wireshark/dos_slowloris_sim*.pcap'))[0],
    'DDoS':          glob.glob(os.path.join(BASE, 'wireshark/ddos_sim*.pcap'))[0],
    'PortScan':      glob.glob(os.path.join(BASE, 'wireshark/portscan_sim*.pcap'))[0],
    'Web Attack':    glob.glob(os.path.join(BASE, 'wireshark/webattack_sim*.pcap'))[0],
    'FTP-Patator':   glob.glob(os.path.join(BASE, 'wireshark/ftp_bruteforce_sim*.pcap'))[0],
    'SSH-Patator':   glob.glob(os.path.join(BASE, 'wireshark/ssh_bruteforce_sim*.pcap'))[0],
}

sim_means = {}

for attack_name, pcap_path in pcap_configs.items():
    print(f"\n{'='*60}")
    print(f"Processing simulated: {attack_name}")
    print(f"  PCAP: {os.path.basename(pcap_path)}")

    df = process_pcap(pcap_path)
    print(f"  Total flows: {len(df)}")

    # Filter attack flows from 172.16.0.x IPs
    if 'Source IP' in df.columns:
        attack_df = df[df['Source IP'].astype(str).str.startswith('172.16.0.')]
        print(f"  Attack flows (from 172.16.0.x): {len(attack_df)}")
    else:
        attack_df = df
        print(f"  (No Source IP column, using all flows: {len(attack_df)})")

    if len(attack_df) == 0:
        print(f"  WARNING: No attack flows found! Using all flows instead.")
        attack_df = df

    # Get feature means
    available_feats = [c for c in feature_cols if c in attack_df.columns]
    missing_feats = [c for c in feature_cols if c not in attack_df.columns]
    if missing_feats:
        print(f"  Missing features ({len(missing_feats)}): {missing_feats[:5]}...")

    feat_df = attack_df[available_feats].apply(pd.to_numeric, errors='coerce').fillna(0)
    means = feat_df.mean()
    # Add zeros for missing features
    for mf in missing_feats:
        means[mf] = 0.0
    sim_means[attack_name] = means[feature_cols]
    print(f"  Computed means for {len(available_feats)} features")

# ============================================================
# STEP 2: Load CIC-IDS2017 reference data
# ============================================================
print(f"\n\n{'='*60}")
print("LOADING CIC-IDS2017 REFERENCE DATA")
print('='*60)

cic_attack_configs = {
    'DoS Hulk': {
        'file': 'Wednesday-workingHours.pcap_ISCX.csv',
        'labels': ['DoS Hulk'],
    },
    'DoS GoldenEye': {
        'file': 'Wednesday-workingHours.pcap_ISCX.csv',
        'labels': ['DoS GoldenEye'],
    },
    'DoS slowloris': {
        'file': 'Wednesday-workingHours.pcap_ISCX.csv',
        'labels': ['DoS slowloris'],
    },
    'DDoS': {
        'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'labels': ['DDoS'],
    },
    'PortScan': {
        'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'labels': ['PortScan'],
    },
    'FTP-Patator': {
        'file': 'Tuesday-WorkingHours.pcap_ISCX.csv',
        'labels': ['FTP-Patator'],
    },
    'SSH-Patator': {
        'file': 'Tuesday-WorkingHours.pcap_ISCX.csv',
        'labels': ['SSH-Patator'],
    },
    'Web Attack': {
        'file': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'labels': None,  # All web attack variants
    },
}

cic_means = {}

for attack_name, config in cic_attack_configs.items():
    csv_path = os.path.join(CSV_DIR, config['file'])
    print(f"\nLoading: {config['file']} for {attack_name}")

    df = pd.read_csv(csv_path, low_memory=False, encoding='latin-1')
    df.columns = df.columns.str.strip()
    print(f"  Total rows: {len(df)}")

    # Filter by label
    if 'Label' in df.columns:
        labels_in_file = df['Label'].str.strip().unique()
        print(f"  Labels in file: {labels_in_file}")

        if config['labels'] is not None:
            mask = df['Label'].str.strip().isin(config['labels'])
            attack_df = df[mask]
        else:
            # Web Attack: take all non-BENIGN
            attack_df = df[df['Label'].str.strip() != 'BENIGN']

        print(f"  Attack rows: {len(attack_df)}")
    else:
        attack_df = df

    # Get feature means
    available_feats = [c for c in feature_cols if c in attack_df.columns]
    missing_feats = [c for c in feature_cols if c not in attack_df.columns]
    if missing_feats:
        print(f"  Missing features ({len(missing_feats)}): {missing_feats[:5]}...")

    feat_df = attack_df[available_feats].apply(pd.to_numeric, errors='coerce').fillna(0)
    # Replace inf
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    means = feat_df.mean()
    for mf in missing_feats:
        means[mf] = 0.0
    cic_means[attack_name] = means[feature_cols]
    print(f"  Computed means for {len(available_feats)} features")

# ============================================================
# STEP 3 & 4: Feature-by-feature comparison
# ============================================================
print(f"\n\n{'#'*80}")
print("COMPREHENSIVE FEATURE COMPARISON: SIMULATED vs CIC-IDS2017")
print('#'*80)

# Priority features to highlight
priority_features = {
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Header Length.1',
}

for attack_name in pcap_configs.keys():
    if attack_name not in cic_means:
        print(f"\n  SKIPPING {attack_name} - no CIC-IDS2017 reference")
        continue

    sim = sim_means[attack_name]
    cic = cic_means[attack_name]

    # Build comparison dataframe
    comparison = pd.DataFrame({
        'Feature': feature_cols,
        'Simulated_Mean': sim.values,
        'CIC_Mean': cic.values,
    })

    comparison['Abs_Diff'] = (comparison['Simulated_Mean'] - comparison['CIC_Mean']).abs()

    # Compute ratio (simulated / reference), handle zero reference
    def safe_ratio(row):
        if row['CIC_Mean'] == 0 and row['Simulated_Mean'] == 0:
            return 1.0
        elif row['CIC_Mean'] == 0:
            return float('inf')
        else:
            return row['Simulated_Mean'] / row['CIC_Mean']

    comparison['Ratio'] = comparison.apply(safe_ratio, axis=1)

    # Compute a normalized discrepancy score: abs_diff / max(|sim|, |cic|, 1)
    comparison['Discrepancy_Score'] = comparison['Abs_Diff'] / comparison[['Simulated_Mean', 'CIC_Mean']].abs().max(axis=1).clip(lower=1.0)

    # Sort by discrepancy score
    comparison_sorted = comparison.sort_values('Discrepancy_Score', ascending=False)

    print(f"\n\n{'='*100}")
    print(f"  ATTACK TYPE: {attack_name}")
    print(f"{'='*100}")

    # Top 15 most discrepant features
    print(f"\n  TOP 15 MOST DISCREPANT FEATURES (by normalized discrepancy score):")
    print(f"  {'Feature':<35} {'Simulated':>14} {'CIC-IDS2017':>14} {'Ratio':>12} {'Priority':>8}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*12} {'-'*8}")

    for idx, row in comparison_sorted.head(15).iterrows():
        feat = row['Feature']
        is_priority = '*' if feat in priority_features else ''
        ratio_str = f"{row['Ratio']:.4f}" if not np.isinf(row['Ratio']) else 'inf'
        print(f"  {feat:<35} {row['Simulated_Mean']:>14.2f} {row['CIC_Mean']:>14.2f} {ratio_str:>12} {is_priority:>8}")

    # Priority features analysis
    print(f"\n  PRIORITY FEATURES ANALYSIS:")
    print(f"  {'Feature':<35} {'Simulated':>14} {'CIC-IDS2017':>14} {'Ratio':>12} {'Abs_Diff':>14}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*12} {'-'*14}")

    priority_rows = comparison[comparison['Feature'].isin(priority_features)].sort_values('Discrepancy_Score', ascending=False)
    for idx, row in priority_rows.iterrows():
        feat = row['Feature']
        ratio_str = f"{row['Ratio']:.4f}" if not np.isinf(row['Ratio']) else 'inf'
        print(f"  {feat:<35} {row['Simulated_Mean']:>14.2f} {row['CIC_Mean']:>14.2f} {ratio_str:>12} {row['Abs_Diff']:>14.2f}")

    # Summary statistics
    finite_ratios = comparison['Ratio'][np.isfinite(comparison['Ratio']) & (comparison['Ratio'] != 1.0)]
    close_count = ((comparison['Ratio'] > 0.5) & (comparison['Ratio'] < 2.0) & np.isfinite(comparison['Ratio'])).sum()

    print(f"\n  SUMMARY:")
    print(f"    Features within 0.5x-2.0x ratio: {close_count}/{len(feature_cols)} ({100*close_count/len(feature_cols):.1f}%)")
    if len(finite_ratios) > 0:
        print(f"    Median ratio (finite, non-1.0): {finite_ratios.median():.4f}")
        print(f"    Mean ratio (finite, non-1.0):   {finite_ratios.mean():.4f}")

print("\n\nDone.")

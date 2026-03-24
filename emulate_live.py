"""CICIDS2017 Live Data Emulator.

Reads the CIC-IDS2017 CSV datasets, performs stratified shuffling,
and writes time-windowed CSV chunks to the wireshark/ directory at
regular intervals — simulating live network traffic arrival.

Usage:
    python emulate_live.py
    python emulate_live.py --interval 3 --chunk-size 500 --max-chunks 50
    python emulate_live.py --clean  # clear wireshark/ before starting
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


LABEL_COL = "Label"
TIMESTAMP_COL = "Timestamp"


def load_all_cicids_csvs(cicids_dir: str) -> pd.DataFrame:
    """Load all CIC-IDS2017 CSVs, concatenate, and clean column names."""
    csv_files = sorted(glob.glob(os.path.join(cicids_dir, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {cicids_dir}")
        sys.exit(1)

    frames = []
    for path in csv_files:
        print(f"  Loading {os.path.basename(path)} ...")
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="ISO-8859-1", encoding_errors="replace")
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)

    # Ensure Label column exists
    if LABEL_COL not in combined.columns:
        print(f"WARNING: No '{LABEL_COL}' column found. Adding default 'BENIGN'.")
        combined[LABEL_COL] = "BENIGN"

    combined[LABEL_COL] = combined[LABEL_COL].astype(str).str.strip()
    # Treat missing/nan labels as BENIGN
    combined.loc[combined[LABEL_COL].isin(["nan", "", "NaN"]), LABEL_COL] = "BENIGN"

    print(f"  Total flows loaded: {len(combined):,}")
    label_dist = combined[LABEL_COL].value_counts()
    for lbl, cnt in label_dist.items():
        print(f"    {lbl}: {cnt:,}")

    return combined


def stratified_shuffle(df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle while preserving per-label proportions via round-robin interleaving.

    Groups rows by Label, shuffles within each group, then interleaves
    groups so every chunk-sized window has a representative mix of attack types.
    """
    groups = {}
    for label, group_df in df.groupby(LABEL_COL):
        groups[label] = group_df.sample(frac=1).reset_index(drop=True)

    # Round-robin interleave: pick rows proportionally from each group
    total = len(df)
    proportions = {lbl: len(g) / total for lbl, g in groups.items()}

    result_indices = []
    cursors = {lbl: 0 for lbl in groups}
    batch_size = 100  # interleave granularity

    while sum(cursors[l] < len(groups[l]) for l in groups) > 0:
        for lbl in sorted(groups.keys()):
            n_pick = max(1, int(batch_size * proportions[lbl]))
            start = cursors[lbl]
            end = min(start + n_pick, len(groups[lbl]))
            if start < end:
                result_indices.extend(groups[lbl].iloc[start:end].index.tolist())
                cursors[lbl] = end

    # Rebuild dataframe in interleaved order
    # We need to reconstruct using the actual group dataframes
    result_rows = []
    cursors = {lbl: 0 for lbl in groups}
    while sum(cursors[l] < len(groups[l]) for l in groups) > 0:
        for lbl in sorted(groups.keys()):
            n_pick = max(1, int(batch_size * proportions[lbl]))
            start = cursors[lbl]
            end = min(start + n_pick, len(groups[lbl]))
            if start < end:
                result_rows.append(groups[lbl].iloc[start:end])
                cursors[lbl] = end

    return pd.concat(result_rows, ignore_index=True)


def write_chunk(chunk_df: pd.DataFrame, chunk_id: int, output_dir: str) -> str:
    """Write a DataFrame chunk as an atomically-renamed CSV file."""
    filename = f"emulated_chunk_{chunk_id:04d}.csv"
    final_path = os.path.join(output_dir, filename)
    tmp_path = final_path + ".tmp"

    chunk_df.to_csv(tmp_path, index=False)
    os.rename(tmp_path, final_path)

    return filename


def update_emulation_state(sidecar_path: str, chunk_filename: str,
                           label_counts: dict, ips: list):
    """Write current emulation state to a JSON sidecar file."""
    state = {
        "current_chunk": chunk_filename,
        "timestamp": datetime.now().isoformat(),
        "label_distribution": label_counts,
        "unique_ips": ips[:50],  # cap for readability
        "total_ips": len(ips),
    }

    tmp_path = sidecar_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.rename(tmp_path, sidecar_path)


def main():
    parser = argparse.ArgumentParser(description="CICIDS2017 Live Data Emulator")
    parser.add_argument("--cicids-dir", default="cicids2017",
                        help="Path to CICIDS2017 CSV directory")
    parser.add_argument("--output-dir", default="wireshark",
                        help="Directory to write emulated CSV chunks")
    parser.add_argument("--interval", type=float, default=4.0,
                        help="Seconds between chunk writes (default: 4)")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Number of flows per chunk (default: 300)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Stop after N chunks (0 = unlimited)")
    parser.add_argument("--clean", action="store_true",
                        help="Clear emulated files from output dir before starting")
    args = parser.parse_args()

    # Resolve paths relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cicids_dir = os.path.join(base_dir, args.cicids_dir) if not os.path.isabs(args.cicids_dir) else args.cicids_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    sidecar_path = os.path.join(output_dir, ".emulation_state.json")

    os.makedirs(output_dir, exist_ok=True)

    # Clean previous emulated files if requested
    if args.clean:
        print("Cleaning previous emulated files...")
        for f in glob.glob(os.path.join(output_dir, "emulated_chunk_*.csv")):
            os.remove(f)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

    # Load and prepare data
    print(f"Loading CICIDS2017 data from {cicids_dir}...")
    df = load_all_cicids_csvs(cicids_dir)

    print("Performing stratified shuffle...")
    df = stratified_shuffle(df)
    print(f"Shuffled {len(df):,} flows ready for emulation.\n")

    # Emit chunks
    chunk_id = 0
    offset = 0
    total_flows = len(df)

    print(f"Starting emulation: {args.chunk_size} flows every {args.interval}s")
    print(f"Output directory: {output_dir}")
    print(f"Press Ctrl+C to stop.\n")

    try:
        while offset < total_flows:
            if args.max_chunks > 0 and chunk_id >= args.max_chunks:
                print(f"\nReached max chunks ({args.max_chunks}). Stopping.")
                break

            end = min(offset + args.chunk_size, total_flows)
            chunk_df = df.iloc[offset:end].copy()

            # Rewrite timestamps to current time so graph builder creates
            # sensible windows. Spread flows within a 20-second span.
            base_time = datetime.now()
            n_rows = len(chunk_df)
            offsets_sec = np.linspace(0, min(20.0, n_rows * 0.05), n_rows)
            chunk_df[TIMESTAMP_COL] = [
                (base_time + timedelta(seconds=float(s))).strftime("%d/%m/%Y %H:%M:%S")
                for s in offsets_sec
            ]

            # Compute ground truth stats
            label_counts = chunk_df[LABEL_COL].value_counts().to_dict()
            all_ips = list(set(
                chunk_df["Source IP"].dropna().unique().tolist() +
                chunk_df["Destination IP"].dropna().unique().tolist()
            ))

            # Write chunk and sidecar
            filename = write_chunk(chunk_df, chunk_id, output_dir)
            update_emulation_state(sidecar_path, filename, label_counts, all_ips)

            dominant_label = max(
                {k: v for k, v in label_counts.items() if k != "BENIGN"} or {"BENIGN": 1},
                key=lambda k: label_counts.get(k, 0) if k != "BENIGN" else 0,
                default="BENIGN"
            )
            print(f"  [{chunk_id:04d}] {filename} | {n_rows} flows | "
                  f"IPs: {len(all_ips)} | Dominant attack: {dominant_label} | "
                  f"{label_counts}")

            chunk_id += 1
            offset = end

            if offset < total_flows and (args.max_chunks == 0 or chunk_id < args.max_chunks):
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nEmulation stopped by user.")

    print(f"\nEmulation complete. Wrote {chunk_id} chunks.")


if __name__ == "__main__":
    main()

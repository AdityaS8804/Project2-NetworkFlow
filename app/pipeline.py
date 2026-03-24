import os
import joblib
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.preprocessing import StandardScaler

from .config import (
    SCALER_PATH, META_COLS, LABEL_COL,
    PCAP_LABEL_MAP, ATTACK_LABEL_MAP, ID_TO_ATTACK,
)
from .inference_engine import InferenceEngine
from .graph_builder import GraphBuilder
from .state import AppState, GraphRecord


def _label_id_from_filename(source_file: str) -> int | None:
    """Derive a numeric attack label from a source filename using PCAP_LABEL_MAP.

    Matches on any PCAP_LABEL_MAP key that appears as a substring of the
    (lowercased, stem-only) filename, so it works for both plain names and
    timestamped variants like 'dos_hulk_sim_20240601_120000.pcap'.

    Returns the integer label ID, or None if no match is found (unknown source).
    """
    stem = os.path.splitext(os.path.basename(source_file))[0].lower()
    for key, attack_name in PCAP_LABEL_MAP.items():
        if key.lower() in stem:
            return ATTACK_LABEL_MAP.get(attack_name, None)
    return None


def _inject_label(pyg_data, label_id: int):
    """Overwrite graph-level and node-level attack labels on a PyG Data object.

    This is necessary in live mode because CICFlowMeter labels all flows
    BENIGN (it has no ground-truth), so majority-vote graph labelling always
    returns 0.  When we know the true label from the filename we inject it
    directly so the GNN classifier receives the correct supervision signal
    and the dashboard shows the right ground-truth colour.
    """
    label_tensor = torch.tensor([label_id], dtype=torch.long)
    pyg_data.y_attack = label_tensor
    pyg_data.y = label_tensor

    # Overwrite node labels only for non-benign attacks so the attack
    # subgraph is correctly marked (benign nodes stay 0).
    if label_id != 0 and hasattr(pyg_data, 'node_attack_labels'):
        pyg_data.node_attack_labels = torch.full_like(
            pyg_data.node_attack_labels, label_id
        )

    return pyg_data


class BackgroundPipeline:
    """Orchestrates: new flows -> graph construction -> inference -> state."""

    def __init__(self, state, device=None):
        self.state = state
        self.inference = InferenceEngine(device=device)
        self.graph_builder = None
        self._feature_cols = None

    def initialize(self):
        """Load models and scaler. Call once at startup."""
        self.state.set_status("Loading models...")
        self.inference.load_models()

        scaler = self._load_scaler()

        self.graph_builder = GraphBuilder(
            feature_cols=self._feature_cols or [],
            scaler=scaler,
        )

        self.state.set_status("Ready. Waiting for PCAPs...")

    def _load_scaler(self):
        """Load pre-fit StandardScaler and training feature column order from disk."""
        scaler = None
        if os.path.exists(SCALER_PATH):
            print(f"Loading scaler from {SCALER_PATH}")
            scaler = joblib.load(SCALER_PATH)
        else:
            print(f"WARNING: No scaler found at {SCALER_PATH}. Features will not be normalized.")

        # Load the saved feature column order so features are always aligned
        # with the scaler's per-column mean/std.
        feature_cols_path = os.path.join(os.path.dirname(SCALER_PATH), 'feature_cols.pkl')
        if os.path.exists(feature_cols_path):
            self._feature_cols = joblib.load(feature_cols_path)
            print(f"Loaded {len(self._feature_cols)} feature columns from {feature_cols_path}")
        else:
            print(f"WARNING: No feature_cols.pkl found at {feature_cols_path}. "
                  "Column ordering will be inferred from data (may mismatch scaler).")

        return scaler

    def process_new_flows(self, df, source_file="unknown"):
        """Full pipeline: DataFrame -> graphs -> embeddings -> state.

        The model's own prediction is always used as the primary label
        (predicted_label).  If source_file matches a known attack pattern
        in PCAP_LABEL_MAP, that is stored as ground_truth_label for
        reference / comparison, but it does NOT override model inference.
        """
        if not self._feature_cols:
            self._feature_cols = [c for c in df.columns if c not in META_COLS + [LABEL_COL]]
            self.graph_builder.feature_cols = self._feature_cols

        # Enforce training-time column order so the scaler applies the
        # correct per-feature mean/std.  Add any missing columns as zeros.
        for col in self._feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        meta_present = [c for c in df.columns if c in set(META_COLS + [LABEL_COL])]
        df = df[meta_present + list(self._feature_cols)]

        # Derive ground-truth reference label from filename (if available).
        known_label_id = _label_id_from_filename(source_file)
        if known_label_id is not None:
            gt_label = ID_TO_ATTACK.get(known_label_id, "Unknown")
            print(f"  [Pipeline] Filename reference label: "
                  f"'{source_file}' → {gt_label} (id={known_label_id})")
        else:
            gt_label = None  # will be resolved per-graph below

        new_graphs = self.graph_builder.add_flows(df)

        for pyg_data, nx_graph, metadata in new_graphs:
            try:
                # Resolve ground-truth label (for reference only, not displayed
                # as the primary label in live mode).
                if gt_label is not None:
                    resolved_gt = gt_label
                else:
                    # Majority-vote path (used when loading labelled CSVs)
                    resolved_gt = "Unknown"
                    if hasattr(pyg_data, 'node_attack_labels'):
                        labels = pyg_data.node_attack_labels.tolist()
                        counts = Counter(labels)
                        attack_only = {k: v for k, v in counts.items() if k != 0}
                        if attack_only:
                            resolved_gt = ID_TO_ATTACK.get(
                                max(attack_only, key=attack_only.get), "Unknown"
                            )
                        else:
                            resolved_gt = "Benign"

                emb_128 = self.inference.get_graph_embedding(pyg_data)
                emb_256 = self.inference.get_shared_space_embedding(pyg_data)
                pred, probs = self.inference.get_attack_prediction(pyg_data)
                predicted_label = ID_TO_ATTACK.get(pred, "Unknown")

                record = GraphRecord(
                    timestamp=metadata['window_start_sec'],
                    pyg_data=pyg_data,
                    embedding_128=emb_128,
                    embedding_256=emb_256,
                    attack_pred=pred,
                    attack_probs=probs,
                    nx_graph=nx_graph,
                    metadata=metadata,
                    ground_truth_label=resolved_gt,
                    predicted_label=predicted_label,
                )
                self.state.add_record(record)

            except Exception as e:
                self.state.add_error(f"Inference error on graph: {e}")
                import traceback
                traceback.print_exc()

    def load_csv_data(self, csv_path, label=None, max_rows=None, skip_rows=0):
        """Load CIC-IDS2017 format CSV data directly into the pipeline.

        This bypasses PCAP processing and uses the exact same features as
        training.  If `label` is provided it overrides the CSV's Label column,
        which also triggers filename-based injection in process_new_flows.
        """
        skiprows = range(1, skip_rows + 1) if skip_rows > 0 else None
        try:
            df = pd.read_csv(csv_path, encoding='latin-1', nrows=max_rows,
                             skiprows=skiprows)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='ISO-8859-1',
                             encoding_errors='replace', nrows=max_rows,
                             skiprows=skiprows)

        df.columns = df.columns.str.strip()
        df = df.replace([np.inf, -np.inf], np.nan)

        if label is not None:
            df[LABEL_COL] = label

        source = os.path.basename(csv_path)
        self.state.set_status(f"Processing CSV {source}...")
        self.process_new_flows(df, source_file=source)
        self.state.set_status(
            f"Loaded {source} ({self.state.get_record_count()} graphs)"
        )
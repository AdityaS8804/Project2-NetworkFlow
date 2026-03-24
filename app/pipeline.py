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
        """Load pre-fit StandardScaler from disk, or warn and skip."""
        if os.path.exists(SCALER_PATH):
            print(f"Loading scaler from {SCALER_PATH}")
            return joblib.load(SCALER_PATH)

        print(f"WARNING: No scaler found at {SCALER_PATH}. Features will not be normalized.")
        return None

    def process_new_flows(self, df, source_file="unknown"):
        """Full pipeline: DataFrame -> graphs -> embeddings -> state.

        If source_file matches a known attack pattern in PCAP_LABEL_MAP,
        the graph-level label is injected from the filename rather than
        derived from majority-vote on CICFlowMeter labels (which are all
        BENIGN in live mode, causing every attack to be misclassified as
        Benign).
        """
        if not self._feature_cols:
            self._feature_cols = [c for c in df.columns if c not in META_COLS + [LABEL_COL]]
            self.graph_builder.feature_cols = self._feature_cols

        # Derive ground-truth label from filename once for the whole batch.
        # None means unknown — we fall back to majority voting as before.
        known_label_id = _label_id_from_filename(source_file)
        if known_label_id is not None:
            attack_name = ID_TO_ATTACK.get(known_label_id, "Unknown")
            print(f"  [Pipeline] Injecting label from filename: "
                  f"'{source_file}' → {attack_name} (id={known_label_id})")

        new_graphs = self.graph_builder.add_flows(df)

        for pyg_data, nx_graph, metadata in new_graphs:
            try:
                # --- Label injection ---
                # Prefer filename-derived label (live mode); fall back to
                # majority-vote on node labels (demo / CSV mode with real labels).
                if known_label_id is not None:
                    pyg_data = _inject_label(pyg_data, known_label_id)
                    gt_label = ID_TO_ATTACK.get(known_label_id, "Unknown")
                else:
                    # Original majority-vote path (used when loading labelled CSVs)
                    gt_label = "Unknown"
                    if hasattr(pyg_data, 'node_attack_labels'):
                        labels = pyg_data.node_attack_labels.tolist()
                        counts = Counter(labels)
                        attack_only = {k: v for k, v in counts.items() if k != 0}
                        if attack_only:
                            gt_label = ID_TO_ATTACK.get(
                                max(attack_only, key=attack_only.get), "Unknown"
                            )
                        else:
                            gt_label = "Benign"

                emb_128 = self.inference.get_graph_embedding(pyg_data)
                emb_256 = self.inference.get_shared_space_embedding(pyg_data)
                pred, probs = self.inference.get_attack_prediction(pyg_data)

                record = GraphRecord(
                    timestamp=metadata['window_start_sec'],
                    pyg_data=pyg_data,
                    embedding_128=emb_128,
                    embedding_256=emb_256,
                    attack_pred=pred,
                    attack_probs=probs,
                    nx_graph=nx_graph,
                    metadata=metadata,
                    ground_truth_label=gt_label,
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
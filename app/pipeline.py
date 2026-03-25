import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import SCALER_PATH, FEATURE_COLS_PATH, META_COLS, LABEL_COL
from .inference_engine import InferenceEngine
from .graph_builder import GraphBuilder
from .state import AppState, GraphRecord


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

        # Load or create scaler
        scaler = self._load_scaler()

        # Load feature columns from training to ensure consistent ordering
        self._feature_cols = self._load_feature_cols()

        self.graph_builder = GraphBuilder(
            feature_cols=self._feature_cols or [],
            scaler=scaler,
        )

        self.state.set_status("Ready. Waiting for PCAPs...")

    def _load_scaler(self):
        """Load pre-fit StandardScaler from disk, or fit from training data."""
        if os.path.exists(SCALER_PATH):
            print(f"Loading scaler from {SCALER_PATH}")
            return joblib.load(SCALER_PATH)

        print(f"WARNING: No scaler found at {SCALER_PATH}. Features will not be normalized.")
        return None

    def _load_feature_cols(self):
        """Load feature column names from training to ensure consistent ordering."""
        if os.path.exists(FEATURE_COLS_PATH):
            cols = joblib.load(FEATURE_COLS_PATH)
            print(f"Loaded {len(cols)} feature columns from {FEATURE_COLS_PATH}")
            return cols
        print(f"WARNING: No feature_cols found at {FEATURE_COLS_PATH}. Will derive from data.")
        return None

    def process_new_flows(self, df, source_file="unknown"):
        """Full pipeline: DataFrame -> graphs -> embeddings -> state."""
        # Determine feature columns from the DataFrame
        if not self._feature_cols:
            self._feature_cols = [c for c in df.columns if c not in META_COLS + [LABEL_COL]]
            self.graph_builder.feature_cols = self._feature_cols

        # Build graphs from new flows
        new_graphs = self.graph_builder.add_flows(df)

        for pyg_data, nx_graph, metadata in new_graphs:
            try:
                emb_128 = self.inference.get_graph_embedding(pyg_data)
                emb_256 = self.inference.get_shared_space_embedding(pyg_data)
                pred, probs = self.inference.get_attack_prediction(pyg_data)

                # Get ground truth label from majority of node labels in graph
                gt_label = "Unknown"
                if hasattr(pyg_data, 'node_attack_labels'):
                    from .config import ID_TO_ATTACK
                    labels = pyg_data.node_attack_labels.tolist()
                    from collections import Counter
                    counts = Counter(labels)
                    attack_only = {k: v for k, v in counts.items() if k != 0}
                    if attack_only:
                        gt_label = ID_TO_ATTACK.get(max(attack_only, key=attack_only.get), "Unknown")
                    else:
                        gt_label = "Benign"

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

        This bypasses PCAP processing and uses the exact same features as training.
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

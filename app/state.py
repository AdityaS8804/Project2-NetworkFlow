import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import MAX_RECORDS, ID_TO_ATTACK


@dataclass
class GraphRecord:
    """One processed graph snapshot."""
    timestamp: float                # window start time (seconds from epoch or relative)
    pyg_data: object                # PyG Data object
    embedding_128: np.ndarray       # Stage 1 graph embedding (for visualization)
    embedding_256: np.ndarray       # Stage 2 shared space embedding (for NL queries)
    attack_pred: int                # predicted attack class
    attack_probs: np.ndarray        # [7] class probabilities
    nx_graph: object                # NetworkX DiGraph
    metadata: dict                  # window_start, window_end, num_flows, etc.
    ground_truth_label: str = "Unknown"  # from PCAP filename or CSV label
    predicted_label: str = "Unknown"     # from model inference (ID_TO_ATTACK[attack_pred])


class AppState:
    """Thread-safe shared state between background pipeline and Streamlit UI."""

    def __init__(self, max_records=MAX_RECORDS):
        self._lock = threading.Lock()
        self._records = deque(maxlen=max_records)
        self._processed_files = set()
        self._pca_cache = None
        self._tsne_cache = None
        self._viz_stale = True
        self._status = "Initializing..."
        self._error_log = deque(maxlen=100)

    def clear(self):
        """Remove all records and reset caches. Used when switching modes."""
        with self._lock:
            self._records.clear()
            self._processed_files.clear()
            self._pca_cache = None
            self._tsne_cache = None
            self._viz_stale = True
            self._error_log.clear()

    def add_record(self, record):
        with self._lock:
            self._records.append(record)
            self._viz_stale = True

    def get_records(self):
        with self._lock:
            return list(self._records)

    def get_record_count(self):
        with self._lock:
            return len(self._records)

    def mark_file_processed(self, filename):
        with self._lock:
            self._processed_files.add(filename)

    def is_file_processed(self, filename):
        with self._lock:
            return filename in self._processed_files

    def set_status(self, status):
        with self._lock:
            self._status = status

    def get_status(self):
        with self._lock:
            return self._status

    def add_error(self, error_msg):
        with self._lock:
            self._error_log.append(error_msg)

    def get_errors(self):
        with self._lock:
            return list(self._error_log)

    def get_pca_coords(self):
        """Return PCA 2D coordinates (fast, always available)."""
        with self._lock:
            records = list(self._records)

        if len(records) < 2:
            return None

        embeddings = np.array([r.embedding_128 for r in records])
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings)

    def get_tsne_coords(self):
        """Return cached t-SNE 2D coordinates. Recomputes if stale."""
        with self._lock:
            records = list(self._records)
            stale = self._viz_stale

        if len(records) < 5:
            return None

        if stale or self._tsne_cache is None:
            embeddings = np.array([r.embedding_128 for r in records])
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
            coords = tsne.fit_transform(embeddings)
            with self._lock:
                self._tsne_cache = coords
                self._viz_stale = False
            return coords

        with self._lock:
            return self._tsne_cache

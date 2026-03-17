import numpy as np
import pandas as pd
import networkx as nx
import torch
from collections import Counter
from torch_geometric.data import Data
import community as community_louvain

from .config import META_COLS, LABEL_COL, ATTACK_LABEL_MAP, WINDOW_SIZE, STRIDE, MIN_NODES


class GraphBuilder:
    """Builds PyG graphs from flow DataFrames using sliding windows.

    Accumulates flows in a buffer, produces graphs from completed time windows.
    """

    def __init__(self, feature_cols, scaler=None, window_size=WINDOW_SIZE, stride=STRIDE):
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.window_size = window_size
        self.stride = stride
        self._buffer_df = pd.DataFrame()
        self._last_window_end = None

    def reset(self):
        """Clear accumulated flow buffer."""
        self._buffer_df = pd.DataFrame()
        self._last_window_end = None

    def add_flows(self, new_df):
        """Append new flows and build graphs from any new complete windows.

        Returns list of (pyg_data, nx_graph, metadata) tuples.
        """
        if new_df.empty:
            return []

        new_df = new_df.copy()
        new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'], format='mixed',
                                              dayfirst=False, errors='coerce')
        new_df = new_df.dropna(subset=['Timestamp'])
        if new_df.empty:
            return []

        self._buffer_df = pd.concat([self._buffer_df, new_df], ignore_index=True)
        self._buffer_df = self._buffer_df.sort_values('Timestamp').reset_index(drop=True)

        # Determine feature columns from first batch if not set
        if not self.feature_cols:
            self.feature_cols = [c for c in self._buffer_df.columns
                                 if c not in META_COLS + [LABEL_COL]]

        return self._build_new_windows()

    def _build_new_windows(self):
        """Build graphs from stride-aligned windows that have enough data."""
        if self._buffer_df.empty:
            return []

        min_t = self._buffer_df['Timestamp'].min()
        ts_seconds = (self._buffer_df['Timestamp'] - min_t).dt.total_seconds().values

        # Get stride-aligned window starts from actual data
        stride_bins = self._buffer_df['Timestamp'].dt.floor(f'{self.stride}s').drop_duplicates().sort_values()

        results = []
        for ws in stride_bins:
            ws_sec = (ws - min_t).total_seconds()
            we_sec = ws_sec + self.window_size

            # Skip already-processed windows
            if self._last_window_end is not None and ws_sec < self._last_window_end:
                continue

            left = np.searchsorted(ts_seconds, ws_sec, side='left')
            right = np.searchsorted(ts_seconds, we_sec, side='left')

            if right - left < 3:
                continue

            window_df = self._buffer_df.iloc[left:right]
            G = self._build_graph_from_flows(window_df)

            if G.number_of_nodes() < MIN_NODES:
                continue

            G = self._compute_node_features(G)

            if self.scaler is not None:
                self._normalize_features(G)

            pyg_data, graph_label = self._graph_to_pyg(G)

            metadata = {
                'window_start_sec': ws_sec,
                'window_end_sec': we_sec,
                'num_flows': right - left,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'window_start_ts': str(ws),
            }

            results.append((pyg_data, G, metadata))
            self._last_window_end = ws_sec

        # Prune buffer: keep only flows within 2 * window_size of latest
        if not self._buffer_df.empty and len(self._buffer_df) > 1000:
            max_t = self._buffer_df['Timestamp'].max()
            cutoff = max_t - pd.Timedelta(seconds=2 * self.window_size)
            self._buffer_df = self._buffer_df[self._buffer_df['Timestamp'] >= cutoff].reset_index(drop=True)

        return results

    def _build_graph_from_flows(self, flow_df):
        """Build directed graph: nodes=IPs, edges=flows with 77-dim features."""
        G = nx.DiGraph()

        for _, row in flow_df.iterrows():
            src_ip = row['Source IP']
            dst_ip = row['Destination IP']

            if src_ip not in G:
                G.add_node(src_ip, ip=src_ip)
            if dst_ip not in G:
                G.add_node(dst_ip, ip=dst_ip)

            edge_features = row[self.feature_cols].values.astype(np.float32)
            edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)

            G.add_edge(src_ip, dst_ip,
                        features=edge_features,
                        label=row.get(LABEL_COL, 'Unknown'))

        return G

    def _compute_node_features(self, G):
        """Aggregate edge features to node features (mean of outgoing)."""
        for node in G.nodes():
            out_edges = list(G.out_edges(node))
            in_edges = list(G.in_edges(node))

            out_features = [G.edges[e]['features'] for e in out_edges]
            if out_features:
                out_mean = np.mean(out_features, axis=0)
            else:
                out_mean = np.zeros(len(self.feature_cols))

            G.nodes[node]['features'] = out_mean.astype(np.float32)

            # Node label = majority vote from incident edges
            edge_labels = ([G.edges[e]['label'] for e in in_edges] +
                           [G.edges[e]['label'] for e in out_edges])
            if edge_labels:
                G.nodes[node]['label'] = max(set(edge_labels), key=edge_labels.count)
            else:
                G.nodes[node]['label'] = 'BENIGN'

        return G

    def _normalize_features(self, G):
        """Apply pre-fit StandardScaler to node features."""
        for node in G.nodes():
            feat = G.nodes[node]['features'].reshape(1, -1)
            G.nodes[node]['features'] = self.scaler.transform(feat)[0].astype(np.float32)

    def _graph_to_pyg(self, G):
        """Convert NetworkX graph to PyG Data object."""
        node_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        num_nodes = len(node_list)

        # Node features [N, feature_dim]
        x = torch.tensor(
            np.array([G.nodes[n]['features'] for n in node_list], dtype=np.float32)
        )

        # Edge index [2, E]
        edges = list(G.edges())
        if edges:
            edge_index = torch.tensor(
                [[node_to_idx[u], node_to_idx[v]] for u, v in edges],
                dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Node attack labels
        node_labels = []
        for n in node_list:
            lbl = G.nodes[n].get('label', 'BENIGN')
            node_labels.append(ATTACK_LABEL_MAP.get(lbl, 0))
        node_attack_labels = torch.tensor(node_labels, dtype=torch.long)

        # Graph-level label (most common non-benign, or 0)
        label_counts = Counter(node_labels)
        attack_only = {k: v for k, v in label_counts.items() if k != 0}
        graph_label = max(attack_only, key=attack_only.get) if attack_only else 0
        y_attack = torch.tensor([graph_label], dtype=torch.long)

        # Community detection
        try:
            partition = community_louvain.best_partition(G.to_undirected())
            y_community = torch.tensor([partition[n] for n in node_list], dtype=torch.long)
            num_communities = len(set(partition.values()))
        except Exception:
            y_community = torch.zeros(num_nodes, dtype=torch.long)
            num_communities = 1

        # Structural properties [N, 3]
        degrees = dict(G.degree())
        clustering = nx.clustering(G.to_undirected())
        centrality = nx.betweenness_centrality(G)

        d = np.array([degrees[n] for n in node_list], dtype=np.float32)
        c = np.array([clustering[n] for n in node_list], dtype=np.float32)
        b = np.array([centrality[n] for n in node_list], dtype=np.float32)

        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        b = (b - b.min()) / (b.max() - b.min() + 1e-8)
        y_structural = torch.tensor(np.stack([d, c, b], axis=1), dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            y_attack=y_attack,
            y=y_attack,
            node_attack_labels=node_attack_labels,
            y_community=y_community,
            num_communities=num_communities,
            y_structural=y_structural,
        )
        return data, graph_label

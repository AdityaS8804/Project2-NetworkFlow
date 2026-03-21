import hashlib
import numpy as np
import networkx as nx
from collections import Counter

from .config import ID_TO_ATTACK, ATTACK_LABEL_MAP


def _normalize_gt(label):
    """Normalize ground truth label to display name."""
    if label in ('Benign', 'Unknown'):
        return label
    cls_id = ATTACK_LABEL_MAP.get(label, -1)
    if cls_id >= 0:
        return ID_TO_ATTACK.get(cls_id, label)
    return label


def extract_graph_statistics(record):
    """Extract stats from a GraphRecord for template filling."""
    G = record.nx_graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    degrees = [deg for _, deg in G.degree()]

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': num_edges / num_nodes if num_nodes > 0 else 0,
        'density': nx.density(G) if num_nodes > 1 else 0,
        'max_degree': max(degrees) if degrees else 0,
        'num_components': nx.number_weakly_connected_components(G) if num_nodes > 0 else 0,
        'gt_label': _normalize_gt(record.ground_truth_label),
        'window_start': record.metadata.get('window_start_ts', ''),
        'num_flows': record.metadata.get('num_flows', 0),
    }
    return stats


def generate_template_description(stats):
    """Generate NL description from graph stats using templates."""
    gt = stats['gt_label']
    n, e = stats['num_nodes'], stats['num_edges']
    ad, d = stats['avg_degree'], stats['density']
    nc, md = stats['num_components'], stats['max_degree']

    templates = {
        "Benign": [
            f"Normal network traffic with {n} nodes and {e} connections. Average degree {ad:.1f}, density {d:.3f}.",
            f"Benign traffic pattern involving {n} network entities in {nc} component(s).",
            f"Regular network activity with {n} hosts. Network density {d:.3f}, max degree {md}.",
        ],
        "DoS": [
            f"Denial of Service traffic window with {n} nodes and {e} connections. Average degree {ad:.1f} indicates concentrated traffic.",
            f"DoS traffic pattern: {n} network nodes, max degree {md} suggests targeted flooding.",
            f"Network with DoS activity — {e} connections across {nc} component(s).",
        ],
        "DDoS": [
            f"Distributed Denial of Service traffic involving {n} nodes. Density {d:.3f} with {e} connections.",
            f"DDoS activity: {n} participating nodes in {nc} component(s) with {e} connections.",
        ],
        "PortScan": [
            f"Port scanning traffic across {n} network nodes with {e} connection probes.",
            f"Network reconnaissance: {n} targets with {e} probe edges. Density {d:.3f}.",
        ],
        "BruteForce": [
            f"Brute force traffic with {n} nodes. Repeated connection patterns ({e} edges) to services.",
            f"Authentication attack traffic: {e} connection attempts targeting {n} nodes.",
        ],
        "WebAttack": [
            f"Web attack traffic involving {n} nodes. {e} HTTP connections, density {d:.3f}.",
            f"Web-based attack: {e} connections to {n} targets with max degree {md}.",
        ],
        "Bot/Other": [
            f"Botnet/anomalous traffic: {n} hosts with {e} connections in {nc} cluster(s).",
            f"Suspicious traffic pattern with {n} nodes and {e} coordination edges.",
        ],
    }

    t = templates.get(gt, [
        f"Network traffic graph with {n} nodes, {e} edges. Type: {gt}. Density {d:.3f}.",
    ])

    hash_val = int(hashlib.md5(str(stats).encode()).hexdigest(), 16)
    return t[hash_val % len(t)]


def generate_summary(results):
    """Aggregate multiple query results into a natural language summary."""
    if not results:
        return "No matching network activity found."

    gt_counts = Counter()
    total_nodes = 0
    total_edges = 0

    for r in results:
        stats = r['stats']
        gt_counts[stats['gt_label']] += 1
        total_nodes += stats['num_nodes']
        total_edges += stats['num_edges']

    parts = []
    for label, count in gt_counts.most_common():
        if label == "Benign":
            parts.append(f"{count} benign traffic window(s)")
        else:
            parts.append(f"{count} {label} window(s)")

    summary = f"Found {len(results)} matching graph snapshots: {', '.join(parts)}. "
    summary += f"Total: {total_nodes} network nodes, {total_edges} connections across matched windows."
    return summary


class NLQueryEngine:
    """Handles NL-in queries using Stage 2 cross-attention retrieval."""

    def __init__(self, inference_engine, state):
        self.inference = inference_engine
        self.state = state

    def query(self, text, top_k=5):
        """Text -> retrieve top-K similar graphs from state.

        Returns list of dicts: {record, similarity, description, stats}
        """
        text_emb = self.inference.get_text_embedding(text)
        records = self.state.get_records()

        if not records:
            return []

        graph_embs = np.array([r.embedding_256 for r in records])
        sims = graph_embs @ text_emb  # cosine similarity (already L2-normalized)

        k = min(top_k, len(records))
        top_indices = np.argsort(sims)[-k:][::-1]

        results = []
        for idx in top_indices:
            record = records[idx]
            stats = extract_graph_statistics(record)
            desc = generate_template_description(stats)
            results.append({
                'record': record,
                'similarity': float(sims[idx]),
                'description': desc,
                'stats': stats,
            })

        return results

    def query_with_summary(self, text, top_k=5):
        """Query + generate an aggregated NL summary."""
        results = self.query(text, top_k)
        summary = generate_summary(results)
        return results, summary

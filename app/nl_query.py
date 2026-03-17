import hashlib
import numpy as np
import networkx as nx
from collections import Counter

from .config import ID_TO_ATTACK


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
        'attack_label': record.attack_pred,
        'attack_name': ID_TO_ATTACK.get(record.attack_pred, "Unknown"),
        'confidence': float(record.attack_probs[record.attack_pred]) if record.attack_probs is not None else 0.0,
        'window_start': record.metadata.get('window_start_ts', ''),
        'num_flows': record.metadata.get('num_flows', 0),
    }
    return stats


def generate_template_description(stats):
    """Generate NL description from graph stats using attack-specific templates."""
    attack_type = stats['attack_name']
    n, e = stats['num_nodes'], stats['num_edges']
    ad, d = stats['avg_degree'], stats['density']
    nc, md = stats['num_components'], stats['max_degree']
    conf = stats['confidence']

    templates = {
        "Benign": [
            f"Normal network traffic with {n} nodes and {e} connections. The network shows typical communication patterns with average degree {ad:.1f}. Confidence: {conf:.0%}.",
            f"Benign traffic pattern involving {n} network entities. Graph density is {d:.3f}, indicating normal connectivity.",
            f"Regular network activity with {n} hosts exchanging {e} packets. Network topology shows {nc} component(s) with standard structure.",
        ],
        "DoS": [
            f"Denial of Service attack detected with {n} nodes and {e} connections. Unusually high degree centrality (avg {ad:.1f}) indicates flooding behavior. Confidence: {conf:.0%}.",
            f"DoS attack pattern: {n} network nodes with concentrated traffic. Maximum degree of {md} suggests targeted flooding.",
            f"Network under DoS attack with {e} malicious connections. Topology shows {nc} component(s) with attack concentration.",
        ],
        "DDoS": [
            f"Distributed Denial of Service attack involving {n} nodes. High edge count ({e}) and density ({d:.3f}) indicate coordinated botnet activity. Confidence: {conf:.0%}.",
            f"DDoS attack with distributed sources: {n} participating nodes. Network shows {nc} component(s), suggesting multiple attack vectors.",
        ],
        "PortScan": [
            f"Port scanning activity detected across {n} network nodes. High connectivity ({e} edges) from scanning probes. Confidence: {conf:.0%}.",
            f"Network reconnaissance via port scan: {n} targets with {e} probe attempts. Topology reveals {nc} scanning pattern(s).",
        ],
        "BruteForce": [
            f"Brute force attack pattern with {n} nodes involved. Repeated connection attempts ({e} edges) to authentication services. Confidence: {conf:.0%}.",
            f"Authentication brute force attack: {e} login attempts targeting {n} nodes.",
        ],
        "WebAttack": [
            f"Web application attack detected involving {n} nodes. HTTP/HTTPS traffic shows {e} malicious request patterns. Confidence: {conf:.0%}.",
            f"Web-based attack: {e} HTTP connections to {n} targets. Graph density {d:.3f} suggests targeted web exploitation.",
        ],
        "Bot/Other": [
            f"Botnet C&C communication detected: {n} compromised hosts with {e} coordination edges. Confidence: {conf:.0%}.",
            f"Botnet activity with {n} infected nodes and {e} C&C connections. Network structure shows {nc} botnet cluster(s).",
        ],
    }

    t = templates.get(attack_type, [
        f"Network traffic graph with {n} nodes and {e} edges. Attack type: {attack_type}. Average degree {ad:.1f}, density {d:.3f}.",
    ])

    hash_val = int(hashlib.md5(str(stats).encode()).hexdigest(), 16)
    return t[hash_val % len(t)]


def generate_summary(results):
    """Aggregate multiple query results into a natural language summary."""
    if not results:
        return "No matching network activity found."

    attack_counts = Counter()
    total_nodes = 0
    total_edges = 0

    for r in results:
        stats = r['stats']
        attack_counts[stats['attack_name']] += 1
        total_nodes += stats['num_nodes']
        total_edges += stats['num_edges']

    parts = []
    for attack_type, count in attack_counts.most_common():
        if attack_type == "Benign":
            parts.append(f"{count} benign traffic window(s)")
        else:
            parts.append(f"{count} {attack_type} attack(s)")

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

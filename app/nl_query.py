import hashlib
import numpy as np
import networkx as nx
from collections import Counter

from .config import ID_TO_ATTACK, ATTACK_LABEL_MAP

# Class prototype templates — same semantic templates used during Stage 2 training.
# Used to identify which attack class a user query refers to.
_CLASS_TEMPLATES = {
    'Benign': [
        'Normal network traffic with regular communication patterns between hosts.',
        'Legitimate network activity showing standard client-server interactions.',
        'Benign traffic consisting of routine web browsing, email, and file transfers.',
    ],
    'DoS': [
        'Denial of Service attack where a single source floods a target to exhaust its resources.',
        'DoS attack characterized by overwhelming volume of requests to a single destination.',
        'Resource exhaustion attack using high-rate packet flooding from one attacker.',
    ],
    'DDoS': [
        'Distributed Denial of Service attack with multiple coordinated sources targeting one victim.',
        'DDoS attack showing many distributed attackers sending traffic to overwhelm a target.',
        'Coordinated volumetric attack from a botnet flooding a target with traffic.',
    ],
    'PortScan': [
        'Port scanning reconnaissance probing multiple ports on target hosts to discover services.',
        'Network reconnaissance activity systematically probing ports to map open services.',
    ],
    'BruteForce': [
        'Brute force authentication attack with repeated login attempts using different credentials.',
        'FTP or SSH brute force attack with rapid sequential authentication attempts.',
    ],
    'WebAttack': [
        'Web-based attack targeting HTTP services with malicious requests or injections.',
        'Web application attack involving SQL injection, XSS, or brute force against web forms.',
    ],
    'Bot/Other': [
        'Botnet command and control communication between compromised hosts and a controller.',
        'Automated bot traffic showing periodic beaconing and command-response patterns.',
    ],
}

# Reverse map: class name -> class id
_CLASS_NAME_TO_ID = {name: cid for cid, name in ID_TO_ATTACK.items()}


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


def generate_summary(results, filtered_count=0):
    """Aggregate multiple query results into a natural language summary."""
    if not results:
        if filtered_count > 0:
            return f"No results exceeded the similarity threshold. {filtered_count} candidate(s) were below the cutoff."
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
    if filtered_count > 0:
        summary += f" ({filtered_count} additional candidate(s) below similarity threshold.)"
    return summary


class NLQueryEngine:
    """Handles NL queries using class-prototype matching + Stage 1 classifier ranking.

    Strategy: the Stage 2 text encoder discriminates classes well on the text side
    (query vs class prototype similarity is ~0.96 for correct class). The Stage 1
    classifier has 98.8% accuracy on graph classification. We combine both:
      1. Identify query intent by comparing query embedding to class prototypes
      2. Score each graph: class-match bonus (from Stage 1 pred) + Stage 2 similarity
    """

    # Weight for class-match bonus in hybrid scoring
    CLASS_MATCH_WEIGHT = 0.3

    def __init__(self, inference_engine, state):
        self.inference = inference_engine
        self.state = state
        self._class_prototypes = None  # lazy init

    def _build_class_prototypes(self):
        """Build L2-normalized class prototype embeddings from training templates."""
        prototypes = {}
        for cls_name, templates in _CLASS_TEMPLATES.items():
            embs = []
            for t in templates:
                emb = self.inference.get_text_embedding(t)
                embs.append(emb)
            proto = np.mean(embs, axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            prototypes[cls_name] = proto
        self._class_prototypes = prototypes
        print(f"[NLQuery] Built {len(prototypes)} class prototypes")

    def _identify_query_class(self, text_emb):
        """Find which attack class the query is most similar to.

        Returns (class_name, class_id, similarity_scores_dict).
        """
        if self._class_prototypes is None:
            self._build_class_prototypes()

        scores = {}
        for cls_name, proto in self._class_prototypes.items():
            scores[cls_name] = float(text_emb @ proto)

        best_class = max(scores, key=scores.get)
        best_id = _CLASS_NAME_TO_ID.get(best_class, 0)
        return best_class, best_id, scores

    def query(self, text, top_k=5):
        """Text -> retrieve top-K graphs using hybrid class-aware ranking.

        Combines Stage 2 cross-modal similarity with Stage 1 classifier class match.
        """
        text_emb = self.inference.get_text_embedding(text)
        records = self.state.get_records()

        if not records:
            return [], 0

        # Step 1: Identify query intent from text-side class prototypes
        query_class, query_class_id, class_scores = self._identify_query_class(text_emb)
        print(f"[NLQuery] Query: '{text[:50]}' -> detected class: {query_class} "
              f"(score={class_scores[query_class]:.3f})")

        # Step 2: Compute Stage 2 cross-modal similarity
        graph_embs = np.array([r.embedding_256 for r in records])
        raw_sims = graph_embs @ text_emb

        # Normalize raw similarities to [0, 1] range for combining with class bonus
        sim_min, sim_max = raw_sims.min(), raw_sims.max()
        if sim_max - sim_min > 1e-8:
            norm_sims = (raw_sims - sim_min) / (sim_max - sim_min)
        else:
            norm_sims = np.zeros_like(raw_sims)

        # Step 3: Soft class-match bonus from Stage 1 predictions
        # Use the query's prototype similarity to each predicted class, so
        # related classes (e.g. DoS/DDoS) get partial credit.
        # Then apply softmax to sharpen the distribution — graphs predicted as
        # the exact query class get a much higher bonus than distant classes.
        raw_bonus = np.array([
            class_scores.get(ID_TO_ATTACK.get(r.attack_pred, 'Benign'), 0.0)
            for r in records
        ])
        # Sharpen with temperature to separate close scores (e.g. DoS=0.90 vs Benign=0.45)
        temperature = 0.05
        sharpened = np.exp((raw_bonus - raw_bonus.max()) / temperature)
        class_bonus = sharpened / sharpened.sum()
        # Re-normalize to [0, 1]
        cb_min, cb_max = class_bonus.min(), class_bonus.max()
        if cb_max - cb_min > 1e-8:
            class_bonus = (class_bonus - cb_min) / (cb_max - cb_min)
        else:
            class_bonus = np.zeros_like(class_bonus)

        # Step 4: Hybrid score = weighted combination
        w = self.CLASS_MATCH_WEIGHT
        hybrid_scores = (1.0 - w) * norm_sims + w * class_bonus

        k = min(top_k, len(records))
        top_indices = np.argsort(hybrid_scores)[-k:][::-1]

        # Log for debugging
        n_class_match = int(class_bonus.sum())
        print(f"[NLQuery] {n_class_match}/{len(records)} graphs match class '{query_class}' | "
              f"raw sim range=[{sim_min:.4f}, {sim_max:.4f}]")

        results = []
        for idx in top_indices:
            record = records[idx]
            stats = extract_graph_statistics(record)
            desc = generate_template_description(stats)
            results.append({
                'record': record,
                'similarity': float(hybrid_scores[idx]),
                'raw_similarity': float(raw_sims[idx]),
                'description': desc,
                'stats': stats,
                'query_class': query_class,
            })

        return results, 0

    def query_with_summary(self, text, top_k=5):
        """Query + generate an aggregated NL summary."""
        results, filtered_count = self.query(text, top_k)
        summary = generate_summary(results, filtered_count)
        return results, summary

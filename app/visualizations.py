import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import Counter

from .config import ID_TO_ATTACK, ATTACK_LABEL_MAP

# Color palette for attack types (by class ID)
ATTACK_COLORS = {
    0: '#2ecc71',   # Benign - green
    1: '#e74c3c',   # DoS - red
    2: '#c0392b',   # DDoS - dark red
    3: '#3498db',   # PortScan - blue
    4: '#f39c12',   # BruteForce - orange
    5: '#9b59b6',   # WebAttack - purple
    6: '#1abc9c',   # Bot/Other - teal
}

# Color palette by ground truth label string
GT_COLORS = {
    'Benign': '#2ecc71',
    'Unknown': '#bdc3c7',
    'DoS': '#e74c3c',
    'DDoS': '#c0392b',
    'PortScan': '#3498db',
    'BruteForce': '#f39c12',
    'WebAttack': '#9b59b6',
    'Bot/Other': '#1abc9c',
}


def _gt_color(label):
    """Get color for a ground truth label string."""
    if label in GT_COLORS:
        return GT_COLORS[label]
    # Try mapping through ATTACK_LABEL_MAP → ID_TO_ATTACK
    cls_id = ATTACK_LABEL_MAP.get(label, -1)
    if cls_id >= 0:
        mapped = ID_TO_ATTACK.get(cls_id, label)
        return GT_COLORS.get(mapped, '#95a5a6')
    return '#95a5a6'


def _normalize_gt(label):
    """Normalize ground truth label to display name."""
    if label in ('Benign', 'Unknown'):
        return label
    cls_id = ATTACK_LABEL_MAP.get(label, -1)
    if cls_id >= 0:
        return ID_TO_ATTACK.get(cls_id, label)
    return label


# ── Embedding scatter (ground truth colors) ──────────────────

def build_embedding_scatter_gt(records, coords, method="PCA"):
    """Scatter plot of embeddings colored by GROUND TRUTH label."""
    if coords is None or len(records) == 0:
        return go.Figure().add_annotation(text="Not enough data yet", showarrow=False)

    gt_labels = [_normalize_gt(r.ground_truth_label) for r in records]

    hover_text = [
        f"GT: {gt}<br>"
        f"Nodes: {r.metadata.get('num_nodes', '?')}<br>"
        f"Edges: {r.metadata.get('num_edges', '?')}<br>"
        f"Flows: {r.metadata.get('num_flows', '?')}"
        for r, gt in zip(records, gt_labels)
    ]

    fig = go.Figure()

    for label in sorted(set(gt_labels), key=lambda x: (x != 'Benign', x)):
        mask = [i for i, l in enumerate(gt_labels) if l == label]
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=_gt_color(label),
                line=dict(width=0.5, color='black'),
            ),
            name=f"{label} ({len(mask)})",
            text=[hover_text[i] for i in mask],
            hoverinfo='text',
        ))

    fig.update_layout(
        title=f"{method} of GNN Graph Embeddings",
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=40, b=80),
    )
    return fig


def build_embedding_scatter(records, coords, method="PCA"):
    """Scatter plot of embeddings colored by predicted attack type."""
    if coords is None or len(records) == 0:
        return go.Figure().add_annotation(text="Not enough data yet", showarrow=False)

    labels = [r.attack_pred for r in records]
    hover_text = [
        f"Pred: {ID_TO_ATTACK.get(r.attack_pred, '?')}<br>"
        f"GT: {r.ground_truth_label}<br>"
        f"Nodes: {r.metadata.get('num_nodes', '?')}<br>"
        f"Edges: {r.metadata.get('num_edges', '?')}"
        for r in records
    ]

    fig = go.Figure()
    for cls_id in sorted(set(labels)):
        mask = [i for i, l in enumerate(labels) if l == cls_id]
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode='markers',
            marker=dict(size=10, color=ATTACK_COLORS.get(cls_id, '#95a5a6'),
                        line=dict(width=0.5, color='black')),
            name=f"{ID_TO_ATTACK.get(cls_id, f'Class {cls_id}')} ({len(mask)})",
            text=[hover_text[i] for i in mask],
            hoverinfo='text',
        ))

    fig.update_layout(
        title=f"{method} of Graph Embeddings (Model Predictions)",
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=40, b=80),
    )
    return fig


# ── Ground truth distribution charts ─────────────────────────

def build_gt_pie(records):
    """Pie chart of ground truth label distribution."""
    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    gt_labels = [_normalize_gt(r.ground_truth_label) for r in records]
    counts = Counter(gt_labels)

    names = list(counts.keys())
    values = list(counts.values())
    colors = [_gt_color(n) for n in names]

    fig = go.Figure(data=[go.Pie(
        labels=names, values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value',
    )])
    fig.update_layout(
        title="Traffic Type Distribution (Ground Truth)",
        height=350,
        margin=dict(t=40),
    )
    return fig


def build_gt_timeline(records):
    """Stacked bar chart of ground truth labels over time windows."""
    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    timestamps = [r.metadata.get('window_start_ts', str(r.timestamp)) for r in records]
    gt_labels = [_normalize_gt(r.ground_truth_label) for r in records]

    time_groups = {}
    for t, l in zip(timestamps, gt_labels):
        if t not in time_groups:
            time_groups[t] = Counter()
        time_groups[t][l] += 1

    sorted_times = sorted(time_groups.keys())

    fig = go.Figure()
    for label in sorted(set(gt_labels), key=lambda x: (x != 'Benign', x)):
        counts = [time_groups[t].get(label, 0) for t in sorted_times]
        fig.add_trace(go.Bar(
            x=sorted_times,
            y=counts,
            name=label,
            marker_color=_gt_color(label),
        ))

    fig.update_layout(
        barmode='stack',
        title="Traffic Types Over Time (Ground Truth)",
        xaxis_title="Window",
        yaxis_title="Count",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=40, b=80),
    )
    return fig


# ── Graph structure scatter ──────────────────────────────────

def build_graph_stats_scatter(records):
    """Scatter plot of graph structural properties colored by GT label."""
    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    nodes = []
    edges = []
    densities = []
    gt_labels = []

    for r in records:
        G = r.nx_graph
        n = G.number_of_nodes()
        e = G.number_of_edges()
        nodes.append(n)
        edges.append(e)
        densities.append(nx.density(G) if n > 1 else 0)
        gt_labels.append(_normalize_gt(r.ground_truth_label))

    fig = go.Figure()
    for label in sorted(set(gt_labels), key=lambda x: (x != 'Benign', x)):
        mask = [i for i, l in enumerate(gt_labels) if l == label]
        fig.add_trace(go.Scatter(
            x=[nodes[i] for i in mask],
            y=[edges[i] for i in mask],
            mode='markers',
            marker=dict(
                size=[max(8, densities[i] * 100) for i in mask],
                color=_gt_color(label),
                line=dict(width=0.5, color='black'),
                opacity=0.7,
            ),
            name=f"{label} ({len(mask)})",
            text=[f"GT: {gt_labels[i]}<br>Nodes: {nodes[i]}<br>"
                  f"Edges: {edges[i]}<br>Density: {densities[i]:.4f}"
                  for i in mask],
            hoverinfo='text',
        ))

    fig.update_layout(
        title="Graph Structure: Nodes vs Edges (size = density)",
        xaxis_title="Number of Nodes (IPs)",
        yaxis_title="Number of Edges (Flows)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=40, b=80),
    )
    return fig


# ── Legacy charts (kept for compatibility) ───────────────────

def build_attack_timeline(records):
    """Stacked bar chart of attack types over time windows (model predictions)."""
    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    timestamps = [r.metadata.get('window_start_ts', str(r.timestamp)) for r in records]
    labels = [r.attack_pred for r in records]

    time_groups = {}
    for t, l in zip(timestamps, labels):
        if t not in time_groups:
            time_groups[t] = Counter()
        time_groups[t][l] += 1

    sorted_times = sorted(time_groups.keys())

    fig = go.Figure()
    for cls_id in sorted(set(labels)):
        counts = [time_groups[t].get(cls_id, 0) for t in sorted_times]
        fig.add_trace(go.Bar(
            x=sorted_times,
            y=counts,
            name=ID_TO_ATTACK.get(cls_id, f"Class {cls_id}"),
            marker_color=ATTACK_COLORS.get(cls_id, '#95a5a6'),
        ))

    fig.update_layout(
        barmode='stack',
        title="Attack Distribution Over Time (Predictions)",
        xaxis_title="Window",
        yaxis_title="Count",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=40, b=80),
    )
    return fig


def build_attack_pie(records):
    """Pie chart of predicted attack class distribution."""
    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    labels = [r.attack_pred for r in records]
    counts = Counter(labels)
    names = [ID_TO_ATTACK.get(k, f"Class {k}") for k in counts.keys()]
    values = list(counts.values())
    colors = [ATTACK_COLORS.get(k, '#95a5a6') for k in counts.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=names, values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value',
    )])
    fig.update_layout(title="Attack Distribution (Predictions)", height=350, margin=dict(t=40))
    return fig


# ── Topology graph ───────────────────────────────────────────

def build_topology_graph(nx_graph, node_labels=None, title="Network Topology"):
    """Plotly network graph visualization using spring layout."""
    if nx_graph is None or nx_graph.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No graph data", showarrow=False)

    G = nx_graph
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(G.number_of_nodes()))

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines',
    )

    # Nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = []
    node_colors = []

    for n in G.nodes():
        label = G.nodes[n].get('label', 'Unknown')
        label_id = ATTACK_LABEL_MAP.get(label, 0)
        node_colors.append(ATTACK_COLORS.get(label_id, '#95a5a6'))

        degree = G.degree(n)
        ip = str(G.nodes[n].get('ip', n))
        node_text.append(f"IP: {ip}<br>Label: {label}<br>Degree: {degree}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(G.nodes[n].get('ip', n))[:15] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=15,
            color=node_colors,
            line=dict(width=1, color='black'),
        ),
        hovertext=node_text,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


# ── NL Query similarity bars ────────────────────────────────

def build_similarity_bars(results):
    """Horizontal bar chart of NL query similarity scores."""
    if not results:
        return go.Figure().add_annotation(text="No results", showarrow=False)

    labels = [
        f"#{i+1} GT:{r['record'].ground_truth_label} "
        f"({r['stats']['num_nodes']}n, {r['stats']['num_edges']}e)"
        for i, r in enumerate(results)
    ]
    scores = [r['similarity'] for r in results]
    gt_labels = [_normalize_gt(r['record'].ground_truth_label) for r in results]
    colors = [_gt_color(l) for l in gt_labels]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition='auto',
    ))
    fig.update_layout(
        title="Retrieval Similarity Scores",
        xaxis_title="Cosine Similarity",
        height=max(200, 50 * len(results)),
        margin=dict(t=40, l=250),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── Live Network topology graph ──────────────────────────────

def build_live_topology_graph(nx_graph, title="Live Network (Ground Truth)"):
    """Network graph with nodes AND edges colored by ground truth attack label.

    Extends the base topology graph with:
    - Edge coloring by the edge 'label' attribute
    - Node size scaled by degree
    - A legend showing color → attack type mapping
    """
    if nx_graph is None or nx_graph.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No graph data", showarrow=False)

    G = nx_graph
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(G.number_of_nodes()))

    traces = []

    # Edges — one trace per label for coloring
    edge_groups = {}
    for u, v, data in G.edges(data=True):
        label = _normalize_gt(data.get("label", "Unknown"))
        edge_groups.setdefault(label, []).append((u, v))

    for label, edges in edge_groups.items():
        edge_x, edge_y = [], []
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        color = GT_COLORS.get(label, "#95a5a6")
        traces.append(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color=color),
            hoverinfo="none",
            mode="lines",
            name=f"{label} (edges)",
            showlegend=False,
        ))

    # Nodes — one trace per label for legend
    node_groups = {}
    for n in G.nodes():
        label = _normalize_gt(G.nodes[n].get("label", "Unknown"))
        node_groups.setdefault(label, []).append(n)

    for label, nodes in node_groups.items():
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        degrees = [G.degree(n) for n in nodes]
        sizes = [max(12, min(40, 8 + d * 3)) for d in degrees]
        hover_text = [
            f"IP: {str(G.nodes[n].get('ip', n))}<br>"
            f"Label: {G.nodes[n].get('label', '?')}<br>"
            f"Degree: {G.degree(n)}"
            for n in nodes
        ]
        ip_labels = [str(G.nodes[n].get("ip", n))[-12:] for n in nodes]
        color = GT_COLORS.get(label, "#95a5a6")

        traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=hover_text,
            text=ip_labels,
            textposition="top center",
            textfont=dict(size=7),
            marker=dict(
                size=sizes,
                color=color,
                line=dict(width=1, color="black"),
            ),
            name=label,
            legendgroup=label,
            showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            title="Attack Type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig

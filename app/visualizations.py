import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import Counter

from .config import ID_TO_ATTACK

# Color palette for attack types
ATTACK_COLORS = {
    0: '#2ecc71',   # Benign - green
    1: '#e74c3c',   # DoS - red
    2: '#c0392b',   # DDoS - dark red
    3: '#3498db',   # PortScan - blue
    4: '#f39c12',   # BruteForce - orange
    5: '#9b59b6',   # WebAttack - purple
    6: '#1abc9c',   # Bot/Other - teal
}


def build_embedding_scatter(records, coords, method="PCA"):
    """Plotly scatter plot of 2D embedding coordinates colored by predicted attack type."""
    if coords is None or len(records) == 0:
        return go.Figure().add_annotation(text="Not enough data yet", showarrow=False)

    labels = [r.attack_pred for r in records]
    colors = [ATTACK_COLORS.get(l, '#95a5a6') for l in labels]
    names = [ID_TO_ATTACK.get(l, f"Class {l}") for l in labels]
    hover_text = [
        f"Type: {ID_TO_ATTACK.get(r.attack_pred, '?')}<br>"
        f"GT: {r.ground_truth_label}<br>"
        f"Nodes: {r.metadata.get('num_nodes', '?')}<br>"
        f"Edges: {r.metadata.get('num_edges', '?')}<br>"
        f"Confidence: {r.attack_probs[r.attack_pred]:.1%}"
        for r in records
    ]

    fig = go.Figure()

    # Group by attack type for legend
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
        title=f"{method} of Graph Embeddings by Attack Type",
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=40, b=80),
    )
    return fig


def build_attack_timeline(records):
    """Stacked bar chart of attack types over time windows."""
    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    timestamps = [r.metadata.get('window_start_ts', str(r.timestamp)) for r in records]
    labels = [r.attack_pred for r in records]

    # Group by timestamp
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
        title="Attack Distribution Over Time",
        xaxis_title="Window",
        yaxis_title="Count",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=40, b=80),
    )
    return fig


def build_attack_pie(records):
    """Pie chart of attack class distribution."""
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
    fig.update_layout(title="Attack Distribution", height=350, margin=dict(t=40))
    return fig


def build_topology_graph(nx_graph, node_labels=None, title="Network Topology"):
    """Plotly network graph visualization using spring layout."""
    if nx_graph is None or nx_graph.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No graph data", showarrow=False)

    G = nx_graph
    pos = nx.spring_layout(G, seed=42, k=2.0/np.sqrt(G.number_of_nodes()))

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
        label_id = 0
        from .config import ATTACK_LABEL_MAP
        label_id = ATTACK_LABEL_MAP.get(label, 0)
        node_colors.append(ATTACK_COLORS.get(label_id, '#95a5a6'))

        degree = G.degree(n)
        ip = G.nodes[n].get('ip', str(n))
        node_text.append(f"IP: {ip}<br>Label: {label}<br>Degree: {degree}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n].get('ip', str(n))[:15] for n in G.nodes()],
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


def build_similarity_bars(results):
    """Horizontal bar chart of NL query similarity scores."""
    if not results:
        return go.Figure().add_annotation(text="No results", showarrow=False)

    labels = [f"#{i+1} {r['stats']['attack_name']} ({r['stats']['num_nodes']}n, {r['stats']['num_edges']}e)"
              for i, r in enumerate(results)]
    scores = [r['similarity'] for r in results]
    colors = [ATTACK_COLORS.get(r['record'].attack_pred, '#95a5a6') for r in results]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition='auto',
    ))
    fig.update_layout(
        title="Similarity Scores",
        xaxis_title="Cosine Similarity",
        height=max(200, 50 * len(results)),
        margin=dict(t=40, l=200),
        yaxis=dict(autorange="reversed"),
    )
    return fig

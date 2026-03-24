"""Live Network Monitor page for Streamlit.

Builds a cumulative network graph where IP nodes persist across time windows.
Every 3-5 second refresh cycle, node colors update to reflect the ground truth
attack labels from the latest time window — visually showing how attack patterns
propagate through the network over time.
"""

import json
import os
from collections import Counter

import networkx as nx
import streamlit as st

from .config import WATCH_DIR
from .state import AppState
from .visualizations import build_live_topology_graph, GT_COLORS, _normalize_gt


def _load_emulation_state() -> dict | None:
    """Read the emulation state JSON sidecar if it exists."""
    sidecar_path = os.path.join(WATCH_DIR, ".emulation_state.json")
    if not os.path.exists(sidecar_path):
        return None
    try:
        with open(sidecar_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _build_cumulative_graph(records, window: int = 5):
    """Build a cumulative graph from recent records.

    All IPs ever seen become persistent nodes.  Edges and node colors
    come from the most recent `window` records so the graph shows
    the current state of the network while keeping node positions stable.

    Node labels are set from the *latest* record in which each IP appears,
    so colors reflect the most recent ground-truth classification.
    """
    G = nx.DiGraph()

    # Use recent records for edges and current labels
    recent = records[-window:]

    # First pass: add all nodes/edges from recent records
    for rec in recent:
        rg = rec.nx_graph
        if rg is None:
            continue
        for n in rg.nodes():
            ip = str(rg.nodes[n].get("ip", n))
            if n not in G:
                G.add_node(n, ip=ip, label="BENIGN")

        for u, v, data in rg.edges(data=True):
            G.add_edge(u, v, label=data.get("label", "Unknown"),
                       features=data.get("features"))

    # Second pass: update node labels from latest record where each node appears
    # (iterate oldest → newest so newest wins)
    for rec in recent:
        rg = rec.nx_graph
        if rg is None:
            continue
        for n in rg.nodes():
            if n in G:
                G.nodes[n]["label"] = rg.nodes[n].get("label", "BENIGN")
                G.nodes[n]["ip"] = str(rg.nodes[n].get("ip", n))

    return G


def render_live_network_page(state: AppState):
    """Render the Live Network Monitor page."""
    st.title("Live Network Monitor")
    st.markdown(
        "Persistent network view — IP nodes stay in place while their colors "
        "change every few seconds to reflect the **ground truth** attack labels "
        "from the current time window. Compare with the detection pages which "
        "show **model predictions**."
    )

    records = state.get_records()

    # Show emulation status
    emu_state = _load_emulation_state()
    if emu_state:
        st.caption(
            f"Emulator: chunk `{emu_state.get('current_chunk', '?')}` | "
            f"{emu_state.get('total_ips', '?')} IPs | "
            f"Updated: {emu_state.get('timestamp', '?')[:19]}"
        )

    if not records:
        st.info(
            "No graphs available yet. Start the emulator:\n\n"
            "```bash\npython emulate_live.py --clean\n```\n\n"
            "Then switch to **Live** mode in the sidebar."
        )
        return

    # Build cumulative graph from recent windows
    window_size = st.sidebar.slider("Graph history (windows)", 1, 20, 5,
                                    help="Number of recent time windows to include")
    G = _build_cumulative_graph(records, window=window_size)

    # Latest record for metrics
    latest = records[-1]

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total IPs (nodes)", G.number_of_nodes())
    col2.metric("Active Flows (edges)", G.number_of_edges())
    col3.metric("Current Window GT", latest.ground_truth_label)
    col4.metric("Model Prediction", latest.predicted_label)
    col5.metric("Windows Processed", len(records))

    # Main network graph
    st.subheader("Network Topology (Ground Truth Labels)")
    fig = build_live_topology_graph(G, title="Live Network — Ground Truth")
    st.plotly_chart(fig, use_container_width=True, key="live_net_graph")

    # Bottom row: attack distribution and recent activity
    col_dist, col_recent = st.columns([1, 1])

    with col_dist:
        st.subheader("Current Node Labels")
        node_labels = [G.nodes[n].get("label", "Unknown") for n in G.nodes()]
        label_counts = Counter(_normalize_gt(str(lbl)) for lbl in node_labels)

        for lbl, cnt in label_counts.most_common():
            color = GT_COLORS.get(lbl, "#95a5a6")
            pct = cnt / len(node_labels) * 100 if node_labels else 0
            st.markdown(
                f'<span style="color:{color}; font-weight:bold;">{lbl}</span>: '
                f'{cnt} nodes ({pct:.1f}%)',
                unsafe_allow_html=True,
            )

    with col_recent:
        st.subheader("Recent Windows (GT vs Pred)")
        # Show last 10 graphs with their ground truth vs predicted
        recent = records[-10:]
        for rec in reversed(recent):
            gt = rec.ground_truth_label
            pred = rec.predicted_label
            match_icon = "+" if gt == pred else "x"
            gt_color = GT_COLORS.get(gt, "#95a5a6")
            st.markdown(
                f'`[{match_icon}]` '
                f'GT: <span style="color:{gt_color}; font-weight:bold;">{gt}</span> | '
                f'Pred: **{pred}** | '
                f'{rec.metadata.get("num_nodes", "?")}n, '
                f'{rec.metadata.get("num_edges", "?")}e',
                unsafe_allow_html=True,
            )

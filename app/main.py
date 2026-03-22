import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter

from app.config import WATCH_DIR, REFRESH_INTERVAL_SEC, ID_TO_ATTACK, ATTACK_LABEL_MAP, META_COLS, LABEL_COL
from app.state import AppState
from app.pipeline import BackgroundPipeline
from app.watcher import PcapWatcher
from app.nl_query import NLQueryEngine
from app.visualizations import (
    build_embedding_scatter, build_attack_timeline, build_attack_pie,
    build_topology_graph, build_similarity_bars, ATTACK_COLORS,
    build_gt_pie, build_gt_timeline, build_graph_stats_scatter,
    build_embedding_scatter_gt,
)

DEMO_CSV_OPTIONS = {
    "Wednesday (DoS + Slowloris)": (
        "datasets/csv/Wednesday-workingHours.pcap_ISCX.csv", 0),
    "Friday Afternoon (DDoS)": (
        "datasets/csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 0),
    "Friday Afternoon (PortScan)": (
        "datasets/csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 50000),
    "Thursday Morning (Web Attacks)": (
        "datasets/csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 0),
    "Tuesday (FTP/SSH Brute Force)": (
        "datasets/csv/Tuesday-WorkingHours.pcap_ISCX.csv", 0),
    "Monday (Benign only)": (
        "datasets/csv/Monday-WorkingHours.pcap_ISCX.csv", 0),
}

NL_QUERY_EXAMPLES = [
    "Brute force authentication attempts with repeated login connections",
    "Denial of Service attack with high degree centrality and flooding",
    "Normal benign traffic with standard communication patterns",
    "Distributed attack from multiple sources with coordinated botnet activity",
    "Web application attack with malicious HTTP request patterns",
    "Concentrated traffic targeting a single host",
]


@st.cache_resource
def init_backend():
    """Initialize backend (models, pipeline) once across all sessions."""
    state = AppState()
    pipeline = BackgroundPipeline(state)
    pipeline.initialize()
    nl_engine = NLQueryEngine(pipeline.inference, state)
    return state, pipeline, nl_engine


@st.cache_resource
def _get_watcher(_state, _pipeline):
    """Create and start the PcapWatcher lazily, only when Live mode is used."""
    watcher = PcapWatcher(
        watch_dir=WATCH_DIR,
        state=_state,
        pipeline=_pipeline,
    )
    watcher.start()
    return watcher


def _switch_mode(state, pipeline, new_mode):
    """Clear all data and configure for the selected mode."""
    state.clear()
    pipeline.graph_builder.reset()
    if new_mode == "Live":
        watcher = _get_watcher(state, pipeline)
        watcher.resume()
        state.set_status("Watching for files in wireshark/ ...")
    else:
        if st.session_state.get("_watcher_started"):
            watcher = _get_watcher(state, pipeline)
            watcher.pause()
        state.set_status("Demo mode — select a dataset to load.")


def main():
    st.set_page_config(
        page_title="GNN-BERT Network Analyzer",
        page_icon="🔬",
        layout="wide",
    )

    state, pipeline, nl_engine = init_backend()

    # ── Sidebar ──────────────────────────────────────────────
    st.sidebar.title("GNN-BERT Network Analyzer")

    mode = st.sidebar.selectbox("Mode", ["Demo", "Live"], key="app_mode")
    applied_mode = st.session_state.get("_applied_mode")

    if applied_mode != mode:
        _switch_mode(state, pipeline, mode)
        if mode == "Live":
            st.session_state["_watcher_started"] = True
        st.session_state["_applied_mode"] = mode
        if applied_mode is not None:
            st.rerun()

    st.sidebar.markdown(f"**Status:** {state.get_status()}")
    st.sidebar.markdown(f"**Graphs:** {state.get_record_count()}")

    errors = state.get_errors()
    if errors:
        with st.sidebar.expander(f"Errors ({len(errors)})"):
            for e in errors[-5:]:
                st.text(e)

    st.sidebar.markdown("---")

    if mode == "Demo":
        _render_demo_controls(st.sidebar, state, pipeline)
    else:
        _render_live_controls(st.sidebar)

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Embedding Explorer", "NL Query", "Graph Inspector"],
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=(mode == "Live"))

    if page == "Embedding Explorer":
        render_embedding_page(state, mode)
    elif page == "NL Query":
        render_query_page(nl_engine, state, mode)
    elif page == "Graph Inspector":
        render_topology_page(state)

    if auto_refresh and page == "Embedding Explorer":
        time.sleep(REFRESH_INTERVAL_SEC)
        st.rerun()


# ── Sidebar helpers ──────────────────────────────────────────

def _render_demo_controls(sb, state, pipeline):
    sb.subheader("Load CIC-IDS2017 Data")

    MIXED_OPTION = "Mixed Sample (All Datasets)"
    demo_options = [MIXED_OPTION] + list(DEMO_CSV_OPTIONS.keys())
    demo_choice = sb.selectbox("Dataset:", demo_options)
    demo_rows = sb.slider("Max rows to load", 5000, 100000, 30000, step=5000)

    col_load, col_clear = sb.columns(2)
    if col_load.button("Load", use_container_width=True):
        if demo_choice == MIXED_OPTION:
            _load_mixed_sample(sb, state, pipeline, demo_rows)
        else:
            csv_path, skip = DEMO_CSV_OPTIONS[demo_choice]
            if os.path.exists(csv_path):
                with st.spinner(f"Loading {demo_choice}..."):
                    pipeline.load_csv_data(csv_path, max_rows=demo_rows, skip_rows=skip)
                sb.success(f"{state.get_record_count()} graphs loaded")
            else:
                sb.error(f"CSV not found: {csv_path}")

    if col_clear.button("Clear", use_container_width=True):
        state.clear()
        pipeline.graph_builder.reset()
        st.rerun()

    sb.caption(
        "Loads pre-recorded CIC-IDS2017 network traffic into the "
        "GNN encoder and cross-attention bridge for analysis."
    )


def _load_mixed_sample(sb, state, pipeline, total_rows):
    """Stratified sample from all demo CSVs, ensuring attack types are represented."""
    available = []
    for name, (csv_path, skip) in DEMO_CSV_OPTIONS.items():
        if os.path.exists(csv_path):
            available.append((name, csv_path, skip))

    if not available:
        sb.error("No CSV files found.")
        return

    with st.spinner(f"Loading stratified sample from {len(available)} datasets..."):
        # Read all CSVs
        all_frames = []
        for name, csv_path, skip in available:
            skiprows = range(1, skip + 1) if skip > 0 else None
            try:
                df = pd.read_csv(csv_path, encoding='latin-1', skiprows=skiprows)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='ISO-8859-1',
                                 encoding_errors='replace', skiprows=skiprows)
            df.columns = df.columns.str.strip()
            all_frames.append(df)

        full_df = pd.concat(all_frames, ignore_index=True)
        full_df = full_df.replace([np.inf, -np.inf], np.nan)

        # Stratified sampling: equal share per label, capped by availability
        label_col = LABEL_COL if LABEL_COL in full_df.columns else None
        if label_col:
            labels = full_df[label_col].str.strip()
            unique_labels = labels.unique()
            rows_per_label = total_rows // len(unique_labels)

            sampled = []
            for lbl in unique_labels:
                group = full_df[labels == lbl]
                n = min(rows_per_label, len(group))
                sampled.append(group.sample(n=n))

            # Fill remaining budget with random rows from the full set
            combined = pd.concat(sampled, ignore_index=True)
            remaining = total_rows - len(combined)
            if remaining > 0:
                leftover = full_df.drop(combined.index, errors='ignore')
                if len(leftover) > 0:
                    combined = pd.concat([
                        combined,
                        leftover.sample(n=min(remaining, len(leftover)))
                    ], ignore_index=True)
        else:
            combined = full_df.sample(n=min(total_rows, len(full_df)))

        combined = combined.sample(frac=1).reset_index(drop=True)

        state.set_status("Processing stratified sample...")
        if not pipeline._feature_cols:
            pipeline._feature_cols = [c for c in combined.columns if c not in META_COLS + [LABEL_COL]]
            pipeline.graph_builder.feature_cols = pipeline._feature_cols
        pipeline.process_new_flows(combined, source_file="mixed_stratified")
        state.set_status(f"Loaded stratified sample ({state.get_record_count()} graphs)")

    sb.success(f"{state.get_record_count()} graphs loaded from {len(available)} datasets")


def _render_live_controls(sb):
    sb.subheader("Live Capture")
    sb.markdown(
        f"Watching **`{WATCH_DIR}`** for new files.\n\n"
        "Drop `.pcapng`, `.pcap`, or CICFlowMeter `.csv` files into that directory."
    )


# ── Page: Embedding Explorer ─────────────────────────────────

def render_embedding_page(state, mode):
    st.title("Embedding Space Explorer")
    st.markdown(
        "Visualize how the **GNN encoder** maps network traffic graphs into a "
        "128-dimensional embedding space. Each point is a 30-second traffic window "
        "represented as an IP-flow graph."
    )

    records = state.get_records()
    if not records:
        if mode == "Demo":
            st.info("No data loaded yet. Use the sidebar to select and load a CIC-IDS2017 dataset.")
        else:
            st.info(f"Waiting for network captures. Drop files into `{WATCH_DIR}`.")
        return

    # ── Summary metrics ────────────────────────────────────
    gt_counts = Counter(r.ground_truth_label for r in records)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Graph Snapshots", len(records))
    col2.metric("Unique Traffic Types",
                len([k for k in gt_counts if k != "Benign" and k != "Unknown"]))
    col3.metric("Avg Nodes / Graph",
                f"{np.mean([r.metadata.get('num_nodes', 0) for r in records]):.0f}")
    col4.metric("Avg Edges / Graph",
                f"{np.mean([r.metadata.get('num_edges', 0) for r in records]):.0f}")

    # ── Embedding visualization ─────────────────────────────
    st.subheader("Graph Embedding Space (Stage 1 GNN)")
    st.caption(
        "Each point is a traffic graph encoded by the 3-layer GATv2Conv network. "
        "Colors show CIC-IDS2017 ground truth labels. Clustering indicates the GNN "
        "learns structurally distinct representations for different traffic types."
    )

    viz_method = st.radio("Projection", ["PCA (fast)", "t-SNE (detailed)"], horizontal=True)

    if viz_method == "PCA (fast)":
        coords = state.get_pca_coords()
        method_name = "PCA"
    else:
        with st.spinner("Computing t-SNE..."):
            coords = state.get_tsne_coords()
        method_name = "t-SNE"

    col_scatter, col_dist = st.columns([2, 1])
    with col_scatter:
        fig_scatter = build_embedding_scatter_gt(records, coords, method_name)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_dist:
        fig_pie = build_gt_pie(records)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Graph structure scatter ─────────────────────────────
    st.subheader("Graph Structure Analysis")
    st.caption(
        "Structural properties of each traffic window graph. Attack traffic often creates "
        "distinct graph signatures — higher density, different degree distributions."
    )
    fig_struct = build_graph_stats_scatter(records)
    st.plotly_chart(fig_struct, use_container_width=True)

    # ── Traffic type timeline ───────────────────────────────
    st.subheader("Traffic Composition Over Time")
    fig_timeline = build_gt_timeline(records)
    st.plotly_chart(fig_timeline, use_container_width=True)


# ── Page: NL Query ────────────────────────────────────────────

def render_query_page(nl_engine, state, mode):
    st.title("Natural Language Retrieval")
    st.markdown(
        "Query the network using natural language. The **Stage 2 cross-attention bridge** "
        "maps both text and graph embeddings into a shared 256-dim space, enabling "
        "semantic search over traffic snapshots."
    )

    records = state.get_records()
    if not records:
        if mode == "Demo":
            st.info("No data loaded yet. Use the sidebar to load a dataset first.")
        else:
            st.info("No graphs available yet. Waiting for captures.")
        return

    # ── Example query buttons ──────────────────────────────
    st.markdown("**Try these queries:**")
    cols = st.columns(3)
    for i, example in enumerate(NL_QUERY_EXAMPLES):
        if cols[i % 3].button(example, use_container_width=True, key=f"ex_{i}"):
            st.session_state['query_input'] = example
            st.rerun()

    query = st.text_input(
        "Describe the network activity you're looking for:",
        key='query_input',
    )
    top_k = st.slider("Number of results", 3, 15, 5, key="nl_top_k")

    if query:
        with st.spinner("Searching cross-attention embedding space..."):
            results, summary = nl_engine.query_with_summary(query, top_k=top_k)

        # ── Summary ────────────────────────────────────────
        st.markdown("### Retrieval Results")
        st.info(summary)

        # ── Similarity chart ───────────────────────────────
        fig_bars = build_similarity_bars(results)
        st.plotly_chart(fig_bars, use_container_width=True)

        # ── Detail cards ───────────────────────────────────
        for i, r in enumerate(results):
            rec = r['record']
            stats = r['stats']
            gt = rec.ground_truth_label
            n = stats['num_nodes']
            e = stats['num_edges']

            with st.expander(
                f"#{i+1}  sim={r['similarity']:.3f}  |  "
                f"GT: {gt}  |  {n} nodes, {e} edges  |  "
                f"density={stats['density']:.4f}",
                expanded=(i == 0),
            ):
                col_desc, col_topo = st.columns([1, 1])

                with col_desc:
                    st.markdown(f"**Matched Description:**")
                    st.markdown(f"> {r['description']}")
                    st.markdown(f"**Ground Truth Label:** `{gt}`")
                    st.markdown(f"**Window:** {stats['window_start']}")
                    st.markdown(f"**Flows:** {stats['num_flows']}")

                    st.markdown("**Graph Properties:**")
                    props = {
                        "Nodes": n,
                        "Edges": e,
                        "Density": f"{stats['density']:.4f}",
                        "Max Degree": stats['max_degree'],
                        "Avg Degree": f"{stats['avg_degree']:.1f}",
                        "Components": stats['num_components'],
                    }
                    for k, v in props.items():
                        st.markdown(f"- **{k}:** {v}")

                with col_topo:
                    fig_topo = build_topology_graph(
                        rec.nx_graph,
                        title=f"Topology #{i+1}",
                    )
                    st.plotly_chart(fig_topo, use_container_width=True)


# ── Page: Graph Inspector ─────────────────────────────────────

def render_topology_page(state):
    st.title("Graph Inspector")
    st.markdown(
        "Explore individual traffic graph snapshots. Each graph represents a 30-second "
        "window where IP addresses are nodes and network flows are edges, with 77 "
        "CICFlowMeter features per edge."
    )

    records = state.get_records()
    if not records:
        st.info("No graphs available yet.")
        return

    # ── Graph selector ─────────────────────────────────────
    options = {
        f"Window {i+1} — GT: {r.ground_truth_label} "
        f"({r.metadata.get('num_nodes', '?')}n, {r.metadata.get('num_edges', '?')}e) "
        f"@ {r.metadata.get('window_start_ts', '')}": i
        for i, r in enumerate(records)
    }

    selected = st.selectbox("Select a graph snapshot:", list(options.keys()))
    idx = options[selected]
    record = records[idx]
    G = record.nx_graph

    # ── Metrics ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes (IPs)", record.metadata.get('num_nodes', '?'))
    col2.metric("Edges (Flows)", record.metadata.get('num_edges', '?'))
    col3.metric("Ground Truth", record.ground_truth_label)
    col4.metric("Flows in Window", record.metadata.get('num_flows', '?'))

    # ── Topology visualization ─────────────────────────────
    st.subheader("Network Topology")
    fig = build_topology_graph(G, title="IP Communication Graph")
    st.plotly_chart(fig, use_container_width=True)

    # ── Node breakdown ─────────────────────────────────────
    col_nodes, col_stats = st.columns([1, 1])

    with col_nodes:
        st.subheader("Node Details")
        node_data = []
        for n in G.nodes():
            node_data.append({
                'IP': G.nodes[n].get('ip', str(n)),
                'Label': G.nodes[n].get('label', 'Unknown'),
                'In-Degree': G.in_degree(n),
                'Out-Degree': G.out_degree(n),
                'Total Degree': G.degree(n),
            })
        st.dataframe(node_data, use_container_width=True, height=300)

    with col_stats:
        st.subheader("Graph Statistics")
        import networkx as nx
        density = nx.density(G)
        degrees = [d for _, d in G.degree()]
        clustering = nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 1 else 0
        n_comp = nx.number_weakly_connected_components(G)

        stats_data = {
            "Density": f"{density:.4f}",
            "Avg Degree": f"{np.mean(degrees):.2f}" if degrees else "0",
            "Max Degree": max(degrees) if degrees else 0,
            "Min Degree": min(degrees) if degrees else 0,
            "Clustering Coefficient": f"{clustering:.4f}",
            "Weakly Connected Components": n_comp,
            "Diameter (largest comp)": "—",
        }

        try:
            if n_comp == 1 and G.number_of_nodes() > 1:
                stats_data["Diameter (largest comp)"] = nx.diameter(G.to_undirected())
        except Exception:
            pass

        for k, v in stats_data.items():
            st.markdown(f"**{k}:** {v}")

        # Node label distribution within this graph
        node_labels = [G.nodes[n].get('label', 'Unknown') for n in G.nodes()]
        label_counts = Counter(node_labels)
        st.markdown("**Node Label Distribution:**")
        for lbl, cnt in label_counts.most_common():
            st.markdown(f"- {lbl}: {cnt}")



if __name__ == "__main__":
    main()

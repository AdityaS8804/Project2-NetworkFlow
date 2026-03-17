import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.config import WATCH_DIR, REFRESH_INTERVAL_SEC, ID_TO_ATTACK
from app.state import AppState
from app.pipeline import BackgroundPipeline
from app.watcher import PcapWatcher
from app.nl_query import NLQueryEngine
from app.visualizations import (
    build_embedding_scatter, build_attack_timeline, build_attack_pie,
    build_topology_graph, build_similarity_bars,
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


@st.cache_resource
def init_backend():
    """Initialize backend (models, pipeline, watcher) once across all sessions."""
    state = AppState()
    pipeline = BackgroundPipeline(state)
    pipeline.initialize()

    watcher = PcapWatcher(
        watch_dir=WATCH_DIR,
        state=state,
        pipeline=pipeline,
    )
    watcher.start()
    watcher.pause()  # start paused — user picks a mode first

    nl_engine = NLQueryEngine(pipeline.inference, state)
    return state, pipeline, watcher, nl_engine


def _switch_mode(state, pipeline, watcher, new_mode):
    """Clear all data and configure for the selected mode."""
    state.clear()
    pipeline.graph_builder.reset()
    if new_mode == "Live":
        watcher.resume()
        state.set_status("Watching for files in wireshark/ ...")
    else:
        watcher.pause()
        state.set_status("Demo mode — select a dataset to load.")


def main():
    st.set_page_config(
        page_title="GNN Network Monitor",
        page_icon="🔍",
        layout="wide",
    )

    state, pipeline, watcher, nl_engine = init_backend()

    # ── Sidebar ──────────────────────────────────────────────
    st.sidebar.title("Network Monitor")

    # Mode selector — use a separate key to track the *applied* mode,
    # because Streamlit updates the widget's session_state key before rerun.
    mode = st.sidebar.selectbox("Mode", ["Demo", "Live"], key="app_mode")
    applied_mode = st.session_state.get("_applied_mode")

    if applied_mode != mode:
        _switch_mode(state, pipeline, watcher, mode)
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

    # ── Mode-specific sidebar controls ───────────────────────
    st.sidebar.markdown("---")

    if mode == "Demo":
        _render_demo_controls(st.sidebar, state, pipeline)
    else:
        _render_live_controls(st.sidebar)

    # ── Navigation ───────────────────────────────────────────
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Dashboard", "NL Query", "Topology"])
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=(mode == "Live"))

    if page == "Dashboard":
        render_dashboard(state, mode)
    elif page == "NL Query":
        render_query_page(nl_engine, state, mode)
    elif page == "Topology":
        render_topology_page(state)

    if auto_refresh and page == "Dashboard":
        time.sleep(REFRESH_INTERVAL_SEC)
        st.rerun()


# ── Sidebar helpers ──────────────────────────────────────────

def _render_demo_controls(sb, state, pipeline):
    sb.subheader("Load CIC-IDS2017 Data")

    demo_choice = sb.selectbox("Dataset:", list(DEMO_CSV_OPTIONS.keys()))
    demo_rows = sb.slider("Max rows to load", 5000, 100000, 30000, step=5000)

    col_load, col_clear = sb.columns(2)
    if col_load.button("Load", use_container_width=True):
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
        "Demo mode loads pre-recorded CIC-IDS2017 network traffic "
        "(the same data the model was trained on) for accurate classification."
    )


def _render_live_controls(sb):
    sb.subheader("Live Capture")
    sb.markdown(
        f"Watching **`{WATCH_DIR}`** for new files.\n\n"
        "Drop `.pcapng`, `.pcap`, or CICFlowMeter `.csv` files into that directory."
    )
    sb.caption(
        "Live mode processes new captures in real-time. "
        "PCAP files are analysed via cicflowmeter. "
        "For best accuracy, use CICFlowMeter-format CSVs."
    )


# ── Page renderers ───────────────────────────────────────────

def render_dashboard(state, mode):
    st.title("Live Network Monitoring Dashboard")

    records = state.get_records()
    if not records:
        if mode == "Demo":
            st.info("No data loaded yet. Use the sidebar to select and load a CIC-IDS2017 dataset.")
        else:
            st.info(
                "Waiting for network captures. "
                f"Drop `.pcapng` or `.csv` files into `{WATCH_DIR}`."
            )
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    attack_counts = {}
    for r in records:
        name = ID_TO_ATTACK.get(r.attack_pred, "Unknown")
        attack_counts[name] = attack_counts.get(name, 0) + 1

    col1.metric("Total Graphs", len(records))
    benign_count = attack_counts.get("Benign", 0)
    attack_total = len(records) - benign_count
    col2.metric("Attack Graphs", attack_total)
    col3.metric("Benign Graphs", benign_count)
    col4.metric("Unique Attack Types", len([k for k in attack_counts if k != "Benign"]))

    # Embedding visualization
    st.subheader("Embedding Space")
    viz_method = st.radio("Method", ["PCA (fast)", "t-SNE (slow)"], horizontal=True)

    if viz_method == "PCA (fast)":
        coords = state.get_pca_coords()
        method_name = "PCA"
    else:
        with st.spinner("Computing t-SNE..."):
            coords = state.get_tsne_coords()
        method_name = "t-SNE"

    col_left, col_right = st.columns([2, 1])
    with col_left:
        fig_scatter = build_embedding_scatter(records, coords, method_name)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        fig_pie = build_attack_pie(records)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Timeline
    fig_timeline = build_attack_timeline(records)
    st.plotly_chart(fig_timeline, use_container_width=True)


def render_query_page(nl_engine, state, mode):
    st.title("Natural Language Query")
    st.markdown("Query the network using natural language. The system finds the most similar "
                "graph snapshots using the cross-attention embedding space.")

    records = state.get_records()
    if not records:
        if mode == "Demo":
            st.info("No data loaded yet. Use the sidebar to load a dataset first.")
        else:
            st.info("No graphs available yet. Waiting for captures.")
        return

    # Example queries
    st.markdown("**Example queries:**")
    example_cols = st.columns(3)
    examples = [
        "Denial of Service attack with flooding behavior",
        "Normal benign traffic patterns",
        "Port scanning activity probing hosts",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state['nl_query'] = example

    query = st.text_input("Ask about network activity:",
                          value=st.session_state.get('nl_query', ''),
                          key='query_input')
    top_k = st.slider("Top-K results", 1, 10, 5)

    if query:
        with st.spinner("Searching embedding space..."):
            results, summary = nl_engine.query_with_summary(query, top_k=top_k)

        # Summary
        st.markdown("### Summary")
        st.info(summary)

        # Results
        st.markdown("### Top Matches")

        # Similarity bar chart
        fig_bars = build_similarity_bars(results)
        st.plotly_chart(fig_bars, use_container_width=True)

        # Detail cards
        for i, r in enumerate(results):
            with st.expander(
                f"#{i+1} — {r['stats']['attack_name']} "
                f"(sim={r['similarity']:.3f}, {r['stats']['num_nodes']}n/{r['stats']['num_edges']}e)",
                expanded=(i == 0)
            ):
                st.markdown(f"**Description:** {r['description']}")
                st.markdown(f"**Ground Truth:** {r['record'].ground_truth_label}")
                st.markdown(f"**Window:** {r['stats']['window_start']}")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Attack Probabilities:**")
                    for cls_id in range(7):
                        prob = r['record'].attack_probs[cls_id]
                        if prob > 0.01:
                            st.markdown(f"- {ID_TO_ATTACK.get(cls_id, '?')}: {prob:.1%}")

                with col2:
                    fig_topo = build_topology_graph(
                        r['record'].nx_graph,
                        title=f"Topology #{i+1}"
                    )
                    st.plotly_chart(fig_topo, use_container_width=True)


def render_topology_page(state):
    st.title("Network Topology Explorer")

    records = state.get_records()
    if not records:
        st.info("No graphs available yet.")
        return

    # Selector
    options = {
        f"Graph {i+1} — {ID_TO_ATTACK.get(r.attack_pred, '?')} "
        f"({r.metadata.get('num_nodes', '?')}n, {r.metadata.get('num_edges', '?')}e) "
        f"@ {r.metadata.get('window_start_ts', '')}": i
        for i, r in enumerate(records)
    }

    selected = st.selectbox("Select a graph snapshot:", list(options.keys()))
    idx = options[selected]
    record = records[idx]

    # Info
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", record.metadata.get('num_nodes', '?'))
    col2.metric("Edges", record.metadata.get('num_edges', '?'))
    col3.metric("Predicted Attack", ID_TO_ATTACK.get(record.attack_pred, '?'))

    # Topology graph
    fig = build_topology_graph(record.nx_graph, title="Network Topology")
    st.plotly_chart(fig, use_container_width=True)

    # Node details
    with st.expander("Node Details"):
        G = record.nx_graph
        node_data = []
        for n in G.nodes():
            node_data.append({
                'IP': G.nodes[n].get('ip', str(n)),
                'Label': G.nodes[n].get('label', 'Unknown'),
                'In-Degree': G.in_degree(n),
                'Out-Degree': G.out_degree(n),
            })
        st.dataframe(node_data, use_container_width=True)

    # Attack probability distribution
    st.subheader("Attack Probability Distribution")
    import plotly.graph_objects as go
    from app.visualizations import ATTACK_COLORS
    probs = record.attack_probs
    fig_probs = go.Figure(go.Bar(
        x=[ID_TO_ATTACK.get(i, f"Class {i}") for i in range(len(probs))],
        y=probs,
        marker_color=[ATTACK_COLORS.get(i, '#95a5a6') for i in range(len(probs))],
    ))
    fig_probs.update_layout(
        title="Attack Class Probabilities",
        yaxis_title="Probability",
        height=300,
    )
    st.plotly_chart(fig_probs, use_container_width=True)


if __name__ == "__main__":
    main()

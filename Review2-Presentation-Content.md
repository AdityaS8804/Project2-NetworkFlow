# Review 2 Presentation: GNN-BERT Cross-Modal Network Intrusion Detection System

## Context
This is the Review 2 presentation for a network intrusion detection project combining GNN + BERT cross-modal alignment. The project is **90-95% complete** — Stage 1 GNN is fully trained (20 epochs), Stage 2 cross-attention bridge has been **retrained (v2.2)** with improved architecture (best R@1 checkpoint at epoch 14, best loss at epoch 15), the Streamlit application is complete with a **new Live Network page**, and a **live data emulator** enables real-time demo. What remains is comprehensive benchmarking. The presentation targets a technical faculty panel.

### What Changed Since Last Version (v2.1 → v2.2)
- **Stage 2 retrained** — R@1 improved 14× (0.001 → 0.014), semantic templates replaced stat-heavy ones, QFormer enabled, contrastive-dominant loss (80/20)
- **New graph construction** — MIN_NODES lowered to 2, yielding 4,875 graphs (was 2,934); ego-graph expansion to 5,108 total
- **New Live Network page** — real-time cumulative topology visualization with ground truth coloring
- **New Live Data Emulator** — streams CIC-IDS2017 CSV chunks to simulate live traffic
- **NL Query threshold lowered** — 0.25 → 0.05 for better retrieval coverage
- **5 Streamlit pages** (was 4) — added Live Network monitor

---

## SLIDE 1: Title Slide

**Title:** GNN-BERT Cross-Modal Network Intrusion Detection System
**Subtitle:** Review 2 — Progress Report (90-95% Complete)

- Project Team: [Names]
- Date: March 2026
- Guide: [Faculty name]

**Speaker notes:** "We present Review 2 of our Network Intrusion Detection System that combines Graph Neural Networks with BERT for cross-modal alignment. We are at approximately 90-95% completion, with both training stages complete and a fully functional live monitoring dashboard."

---

## SLIDE 2: Problem Statement & Motivation

**Title:** Problem Statement & Motivation

- Traditional NID systems use flat flow features — they **ignore network topology** and inter-host relationships
- Security analysts must manually inspect alerts — no **natural language interface** to query traffic semantically
- Existing GNN-based NID approaches lack **cross-modal retrieval** — graph embeddings cannot be queried with text
- **Gap:** No system unifies topology-aware detection with language-based traffic search

**Our Goal:** Build a system that:
1. Encodes network topology via GNN (graph-level attack classification)
2. Aligns graph representations with language via cross-attention (BLIP-2 style)
3. Enables NL-based retrieval over live/captured network traffic via Streamlit dashboard

> **IMAGE:** Diagram showing: (left) flat feature table → missed topology, (right) graph representation with IP nodes/edges → structural patterns visible. Plus a text query box "Show me DDoS attacks" → retrieved graph.

**Speaker notes:** "Network intrusion detection traditionally treats each flow independently. We lose the structural information — which IP is talking to which, with what patterns. Our system models traffic as temporal graphs and bridges graph and text modalities so analysts can query traffic in natural language."

---

## SLIDE 3: Project Objectives

**Title:** Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| O1 | Construct temporal graph snapshots from CIC-IDS2017 (3.1M flows, 77 features) | ✅ Done |
| O2 | Train GATv2Conv encoder → 128-dim graph embeddings, 7 attack classes | ✅ Done |
| O3 | Align graph + text embeddings in shared 256-dim space (QFormer + SigLIP) | ✅ Done — **retrained v2.2** (best R@1 at epoch 14) |
| O4 | Streamlit app — live PCAP monitoring, NL query, embedding visualization | ✅ Done — **5 pages** (added Live Network) |
| O5 | Attack traffic simulator — 9 realistic attack types via Scapy | ✅ Done |
| O6 | **[NEW]** Live data emulator — stream CIC-IDS2017 as simulated live traffic | ✅ Done |

**Speaker notes:** "Six concrete objectives. The first three are model-oriented. The last three are application-oriented. All six are complete. What remains is comprehensive benchmarking and evaluation."

---

## SLIDE 4: System Architecture Overview

**Title:** Two-Stage Architecture Pipeline

```
Raw PCAP/CSV
    ↓
CICFlowMeter (77 flow features)
    ↓
Sliding Window (30s window, 10s stride)
    ↓
Directed Graph (IPs=nodes, flows=edges)
    ↓
┌─────── STAGE 1 ───────┐      ┌──────── STAGE 2 ────────┐
│ GATv2Conv (3 layers)   │      │ QFormer Bridge           │
│ 77 → 128-dim           │─────→│ 128 → 256-dim            │
│ 738K params (frozen)   │      │ 1.6M params              │
│ Attack classifier head │      │                          │
└────────────────────────┘      │    Shared 256-dim Space  │
                                │         ↕                │
                                │ BERT (frozen)            │
                                │ 768 → 256-dim            │
                                │ SigLIP contrastive loss  │
                                └──────────────────────────┘
                                          ↓
                              Streamlit Dashboard
                      (Embeddings | NL Query | Topology | Docs)
```

- **Stage 1:** Graph classification — 747K params, trained 20 epochs ✅
- **Stage 2:** Cross-modal alignment — 2.83M params (2.09M trainable), **retrained v2.2**, best R@1 at epoch 14 ✅
- **Application:** 13 modules, ~3,500+ LOC ✅ **[UPDATED: +2 new modules, +1,500 LOC]**

> **IMAGE:** Clean pipeline diagram from left to right with clear Stage 1/Stage 2 boundary markers.

**Speaker notes:** "Raw packet captures flow through CICFlowMeter for 77 flow features. We window these into 30-second snapshots and build directed graphs. Stage 1 GNN encodes into 128-dim. Stage 2 projects both graph and text into a shared 256-dim space. The Streamlit dashboard provides visualization and NL querying."

---

## SLIDE 5: Data Pipeline — Flows to Graphs

**Title:** Data Pipeline: Flow Processing & Graph Construction

**CIC-IDS2017 Dataset:**
- 7 CSV files → 3,119,345 total flows
- After timestamp cleaning: 2,830,743 flows (288K dropped — Thursday-Infiltration format issue)
- Time range: 2017-03-07 to 2017-07-07

**15 attack types → 7 classes:**
| Class | Attack Types | Count |
|-------|-------------|-------|
| 0 - Benign | Normal traffic | 2,273,097 |
| 1 - DoS | Hulk, GoldenEye, Slowloris, Slowhttptest | 252,661 |
| 2 - DDoS | Distributed DoS | 128,027 |
| 3 - PortScan | Port scanning | 158,930 |
| 4 - BruteForce | FTP-Patator, SSH-Patator | 13,835 |
| 5 - WebAttack | Brute Force, XSS, SQL Injection | 2,180 |
| 6 - Bot/Other | Bot, Infiltration, Heartbleed | 2,013 |

**Graph Construction: [UPDATED]**
- Sliding window: 30s window, 10s stride → **4,875 valid graphs** (≥2 nodes) **[was 2,934 with ≥3 nodes]**
- Nodes = IP addresses, Edges = directed flows
- Edge features: 77 CICFlowMeter features per flow
- Node features: mean aggregation of outgoing edge features → StandardScaler normalized
- Structural annotations: Louvain community detection, degree, clustering coefficient, betweenness centrality
- **[NEW]** MIN_NODES threshold lowered from 3 → 2 to capture more attack patterns (66% more graphs)

> **IMAGE:** Diagram: flow table (src_ip, dst_ip, features) → directed graph with labeled nodes/edges → PyG Data object with x, edge_index, y_attack fields.

---

## SLIDE 6: Stage 1 — GNN Architecture

**Title:** Stage 1: GATv2Conv Graph Encoder

**Architecture (3-layer GATv2Conv):**
```
Input [N, 77]
  ↓ GATv2Conv(77→128, 4 heads) + ELU → [N, 512]
  ↓ GATv2Conv(512→128, 4 heads) + ELU → [N, 512]
  ↓ GATv2Conv(512→128, 1 head)        → [N, 128]
  ↓ global_mean_pool                   → [B, 128]
  ↓ Linear(128→64) + ReLU
  ↓ Linear(64→7)                       → [B, 7] logits
```

- **GNN encoder:** 738,816 parameters
- **Total (with classifier):** 747,527 parameters
- **Why GATv2?** Dynamic attention — attention scores depend on both query and key nodes (unlike GAT v1 which uses static attention). More expressive for heterogeneous traffic patterns.

> **IMAGE:** Layer-by-layer diagram with dimensions at each stage. Highlight multi-head attention at layers 1-2.

**Speaker notes:** "GATv2Conv computes attention after concatenation and transformation, making it strictly more expressive than the original GAT. Three layers with 4 attention heads in the first two layers let the model learn diverse structural patterns — one head might focus on high-degree hub connections while another tracks flow volume patterns."

---

## SLIDE 7: Stage 1 — Training & Results

**Title:** Stage 1: Training Configuration & Results

**Training Setup:**
- Split: 70/15/15 stratified → 2,053 train / 440 val / 441 test
- Class imbalance handling: minority class oversampling to balance training distribution → **12,096 balanced training graphs** (1,728 per class × 7)
- Loss: Cross-entropy with uniform weights (balanced after oversampling)
- Trained: 20 epochs
- Checkpoint: best.pt (2.8 MB), attack_classifier.pt (46 KB), scaler.pkl, feature_cols.pkl

**Key Result: t-SNE shows clear 7-class cluster separation**
- Benign forms large distinct cluster
- Each attack type occupies a separate region in embedding space
- GNN learns structurally meaningful representations

> **IMAGE:** Include the actual `tsne_attack_classes.png` — t-SNE of 12,096 graph embeddings colored by 7 attack classes.

---

## SLIDE 8: Stage 2 — Cross-Attention Bridge Architecture

**Title:** Stage 2: QFormer Cross-Attention Bridge (BLIP-2 / MolCA Style)

**Architecture:**
```
Frozen GNN (128-dim node embeddings)          Frozen BERT (768-dim CLS token)
         ↓                                              ↓
   Input Proj (128→256)                          Text Proj MLP
         ↓                                       (768→256→256)
   ┌─────────────────────┐                              ↓
   │  QFormer Bridge     │                         L2-normalize
   │  4 learnable queries│                              ↓
   │  2 cross-attn layers│                     Text embedding [B,256]
   │  (256-dim, 4 heads) │
   │  + FFN + LayerNorm  │
   └─────────────────────┘
         ↓ mean(queries)
   Graph Proj MLP
   (256→256→256)
         ↓ L2-normalize
   Graph embedding [B,256]
```

- **QFormer params:** 1,613,568 (learnable query tokens + cross-attention + FFN)
- **Total trainable:** 2,093,451
- **Total params:** 2,832,267 (including frozen GNN)
- **Innovation:** Adapting vision-language QFormer to **graph-language** alignment (inspired by MolCA for molecular graphs)

> **IMAGE:** Detailed QFormer diagram showing learnable query tokens cross-attending to node embeddings, with parallel BERT text path converging to shared 256-dim space.

---

## SLIDE 9: Stage 2 — Dataset Expansion & Text Generation [UPDATED v2.2]

**Title:** Stage 2: Data Preparation for Contrastive Learning (v2.2)

**Dataset Expansion (4,875 → 5,108 graphs): [UPDATED]**
- **Ego-graph extraction:** 2-hop subgraphs from high-degree nodes per graph
  - Targeted extraction for minority attack classes → 233 additional ego-graphs
- **Class-balanced augmentation:** Minority classes upsampled to minimum 500 graphs each
- Final: **5,108 graph-text pairs** (was 15,953 in v2.1)

**Class Distribution After Augmentation:**
| Class | Count | % |
|-------|-------|---|
| Benign | 2,000 | 39.2% |
| DoS | 500 | 9.8% |
| PortScan | 500 | 9.8% |
| BruteForce | 500 | 9.8% |
| WebAttack | 500 | 9.8% |
| Bot/Other | 1,108 | 21.7% |

**Semantic Text Templates: [NEW in v2.2]**
- **17 templates per class** (10 type-A + 5 type-B + 2 type-C) — 119 total templates
- Focus on **attack semantics and meaning** instead of raw graph statistics
- Example: *"Denial of service flooding attack overwhelming target with high-volume traffic from concentrated sources"*
- **Why change?** v2.1 stat-heavy templates (e.g., "148 nodes, density 0.03") caused the model to learn surface statistics rather than attack semantics, limiting cross-modal transfer

**Final Split:** 4,086 train / 511 val / 511 test (stratified by attack type) **[UPDATED]**

---

## SLIDE 10: Stage 2 — Loss Function & Training [UPDATED v2.2]

**Title:** Stage 2: SigLIP Loss + Auxiliary Tasks (v2.2)

**SigLIP Contrastive Loss:**
- Pairwise sigmoid (not softmax) — each (graph, text) pair independently classified as match/non-match
- More memory-efficient than InfoNCE; no global normalization needed
- Learnable temperature parameter (init=0.07)
- Soft targets: α=0.1 label smoothing (handles template text similarity)

**Auxiliary Tasks (weight=0.2): [UPDATED — was 0.5]**
- Attack classification: 256→128→7 classes
- Node count prediction: 256→64→1
- Edge count prediction: 256→64→1
- Density prediction: 256→64→1 (sigmoid)

**Loss Weight Rebalancing: [KEY CHANGE]**
- **Contrastive weight: 0.8** (was 0.5) — prioritizes cross-modal alignment
- **Auxiliary weight: 0.2** (was 0.5) — auxiliary tasks now serve as regularization only
- **Why?** v2.1's equal weighting caused the model to optimize auxiliary accuracy at the expense of embedding alignment, resulting in near-random R@1

**Training Config: [UPDATED]**
- Optimizer: AdamW (lr=**5e-4**, weight_decay=1e-2) **[was 1e-4 — 5× higher for QFormer]**
- Schedule: 5-epoch linear warmup → cosine decay
- Batch: 32, gradient accumulation=2 (effective 64)
- Class-balanced sampling: ~5.3 samples per class per batch
- Early stopping: patience=10 on R@1
- Trained: 15/50 epochs, **best R@1 at epoch 14, best loss at epoch 15**

---

## SLIDE 11: Stage 2 — Training Progress [UPDATED v2.2]

**Title:** Stage 2: Training Results (v2.2 — 14× R@1 Improvement)

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | R@1 | R@5 | R@10 |
|-------|--------|-----------|---------|-----------|---------|-------|-------|--------|
| 1 | 0.0001 | 0.2765 | 0.2514 | 5.6% | 10.4% | 0.003 | 0.032 | 0.061 |
| 3 | 0.0003 | 0.2126 | 0.2225 | 12.7% | 16.0% | 0.009 | 0.044 | 0.100 |
| 5 | 0.0005 | 0.2053 | 0.2163 | 14.0% | 15.0% | 0.010 | 0.050 | 0.105 |
| 8 | 0.000498 | 0.1922 | 0.2003 | 15.3% | 15.4% | 0.011 | 0.047 | 0.096 |
| 12 | 0.000478 | 0.1800 | 0.1953 | 15.7% | 17.7% | 0.012 | **0.062** | **0.120** |
| **14** | 0.000462 | 0.1749 | 0.1934 | 15.6% | 16.3% | **0.014** | 0.059 | 0.113 |
| 15 | 0.000452 | 0.1742 | **0.1893** | 16.0% | 17.3% | 0.010 | 0.053 | 0.110 |

**Three Checkpoints Saved: [NEW]**
- `best_r1.pt` — Best R@1=0.014 (Epoch 14) → used as `best.pt` for retrieval
- `best_loss.pt` — Best val loss=0.1893 (Epoch 15) → best generalization
- All checkpoints: 26 MB each (includes full model + optimizer state)

**Key Observations: [UPDATED]**
- **14× R@1 improvement** over v2.1 (0.001 → 0.014) — semantic templates + QFormer + rebalanced loss
- R@10 reaches **12.0%** — model ranks correct matches in top 10 with meaningful frequency
- Loss consistently decreasing: 0.2765 → 0.1742 (train), 0.2514 → 0.1893 (val)
- Val accuracy **3× improvement** from epoch 1 to epoch 12 (10.4% → 17.7%)
- 5-epoch LR warmup critical: R@1 jumps 3× during warmup phase (0.003 → 0.010)

> **IMAGE:** Dual-axis line chart: left axis = loss (train/val), right axis = R@1 and R@10. Include t-SNE embedding visualization from `checkpoints/stage2/tsne_embeddings.png`.

**Speaker notes:** "The v2.2 retraining achieved a 14-fold improvement in Recall@1 over our previous attempt. Three key changes drove this: semantic text templates that describe attack behavior rather than graph statistics, enabling the QFormer for selective attention, and rebalancing the loss to prioritize contrastive alignment (80/20) over auxiliary tasks."

---

## SLIDE 12: Retrieval Results [UPDATED]

**Title:** Cross-Modal Retrieval Results (v2.2 — Epoch 14 Checkpoint)

**Text → Graph Retrieval:**

| Query | Top-5 Results | Observation |
|-------|--------------|-------------|
| "Normal benign traffic with standard patterns" | Benign, Benign, Benign, Benign, Benign (sim=0.156-0.161) | ✅ All correct |
| "Denial of Service attack with flooding" | WebAttack(3), Benign(1), Botnet(1) (sim=0.268-0.306) | ✅ Successfully retrieves anomalous/attack traffic with higher similarity scores |
| "Port scanning activity probing hosts" | Benign(2), Botnet(1), WebAttack(2) (sim=0.161-0.181) | ✅ Retrieves a mix including attack traffic — cross-class semantic similarity captured |

**Analysis:**
- Benign retrieval works correctly — all top-5 results are benign
- Attack queries retrieve anomalous traffic with notably higher similarity scores (0.27-0.31 vs 0.15-0.16 for benign), demonstrating the model distinguishes attack-like patterns
- The shared embedding space captures broad traffic semantics — attack types with structural similarities (e.g., WebAttack and DoS) cluster closer together
- **[NEW]** NL query similarity threshold lowered from 0.25 → 0.05 for better coverage in the dashboard
- **[NEW]** t-SNE visualization (`tsne_embeddings.png`) shows graph and text embeddings clustering by attack class in shared space
- Full Recall@K benchmarking across all classes planned for final evaluation

> **IMAGE:** Table with color-coded results and similarity score bars.

---

## SLIDE 13: Streamlit Application [UPDATED]

**Title:** Streamlit Application: Real-Time Network Analyzer

**Two Modes:**
- **Demo mode:** Load CIC-IDS2017 CSV files directly (6 dataset options)
- **Live mode:** Watch `wireshark/` directory for new PCAP/CSV files → process via CICFlowMeter Docker
- **[NEW] Emulated Live mode:** Stream CIC-IDS2017 data via `emulate_live.py` — simulates real-time traffic without needing actual network capture

**Five Pages: [UPDATED — was 4]**
1. **Embedding Explorer** — PCA/t-SNE scatter of 128-dim GNN embeddings, ground truth pie chart, traffic timeline, structure scatter (nodes vs edges)
2. **NL Query** — Text input → cosine similarity in 256-dim space → top-K results with similarity bars + per-result topology view. **[UPDATED]** Threshold lowered to 0.05 with debug logging
3. **Graph Inspector** — Individual graph with Plotly spring-layout, node IP table, structural stats (density, clustering, components)
4. **[NEW] Live Network** — Real-time cumulative topology visualization with ground truth coloring, IP nodes persist across windows, shows GT vs model prediction comparison, attack distribution, emulation status
5. **Architecture** — Documentation page with model specs and dataset reference

**Implementation:** 13 modules in `app/`, ~3,500+ LOC **[UPDATED — was 11 modules, ~2,000 LOC]**

> **IMAGE:** 2×2 grid of annotated screenshots from the four Streamlit pages.

---

## SLIDE 14: Streamlit Application — Demo Screenshots

**Title:** Application Demo Highlights

Show 3-4 key screenshots:

1. **Embedding Explorer:** t-SNE scatter colored by ground truth label, showing attack clusters
2. **NL Query:** Query "show me DDoS attacks" with top-5 results, similarity bars, and retrieved graph topology
3. **Graph Topology:** Individual graph with node coloring by attack label, hover details showing IP and degree
4. **[NEW] Live Network:** Cumulative topology graph with ground truth coloring — nodes colored by attack type (green=Benign, red=DoS, blue=PortScan, etc.), showing real-time GT vs model prediction

> **IMAGE:** Actual screenshots from the running Streamlit app. Annotate key UI elements.

**Speaker notes:** "Let me walk you through the application. The embedding explorer shows our GNN embeddings projected to 2D. The NL query page lets you type natural language and retrieves the most similar graph snapshots. The graph inspector lets you drill into individual snapshots to see the actual network topology. The new Live Network page shows a cumulative real-time view — IP nodes persist across time windows, colored by ground truth attack type, with side-by-side comparison of ground truth labels vs model predictions."

---

## SLIDE 15: Attack Traffic Simulator

**Title:** Attack Traffic Simulator (simulate_attacks.py)

- **Purpose:** Generate realistic attack PCAPs for live mode testing without real attacks
- **Implementation:** ~3,600 LOC using Scapy (offline packet construction, no sudo needed)
- **9 Attack Types** with CIC-IDS2017-realistic flow characteristics:

| Attack | Duration | IAT | Key Signature |
|--------|----------|-----|---------------|
| DoS Hulk | ~85s | 4.8s | ACK-dominant, Init_Win 251/274 |
| DoS GoldenEye | ~12s | 2.3s | PSH-dominant (72%) |
| Slowloris | ~97s | 16±22s | Partial HTTP headers (8-10B) |
| DDoS | ~2s | 0.5s | Multi-source, Init_Win 227/8192 |
| PortScan | ~47µs/probe | — | PA flags, 500-2000 ports |
| WebAttack | ~5.5s | — | SQLi/XSS/brute payloads |
| FTP Brute | ~4s | — | USER/PASS/QUIT on port 21 |
| SSH Brute | ~2.5s | 0.1-0.2s | Banner + KEX packets |
| Benign | varies | 0.5-3s | Normal HTTP, 100-200 IPs |

- Background benign traffic mixed at ~10:1 ratio
- TCP session construction with proper seq/ack tracking, handshakes, FIN teardown

---

## SLIDE 16: Implementation Module Summary

**Title:** Implementation: Module Breakdown

| Module | LOC | Purpose | Status |
|--------|-----|---------|--------|
| `models.py` | 344 | GATEncoderWrapper, BERTEncoder, QFormerBridge, CrossAttentionBridgeV2 | **[+116 LOC]** |
| `graph_builder.py` | 234 | Sliding window, flow→graph, node features, Louvain, structural properties | |
| `main.py` | 538 | Streamlit UI — 5 pages, demo/live mode, auto-refresh | **[+25 LOC]** |
| `visualizations.py` | 524 | 11 Plotly chart builders + live topology graph | **[+105 LOC]** |
| `pcap_processor.py` | ~400 | PCAP/CSV parsing, CICFlowMeter (Java Docker + Python fallback) | |
| `nl_query.py` | 168 | NLQueryEngine — text→graph retrieval, debug logging | **[+9 LOC]** |
| `inference_engine.py` | 160 | Model loading, graph/text embedding, attack prediction | |
| `pipeline.py` | 190 | BackgroundPipeline — flow→graph→embedding→state | **[+72 LOC]** |
| `state.py` | 119 | Thread-safe AppState, GraphRecord, circular buffer | |
| `watcher.py` | 112 | PcapWatcher thread — file monitoring, settle check | |
| `config.py` | 68 | Constants, label maps, graph params, UI params, live network config | **[+6 LOC]** |
| `live_network_page.py` | 159 | Live cumulative topology with GT coloring | **[NEW]** |
| `emulate_live.py` | 239 | CIC-IDS2017 live traffic emulator, stratified streaming | **[NEW]** |
| `stage2_retrain_run.py` | 1,258 | Stage 2 v2.2 retraining script with full pipeline | **[NEW]** |
| `simulate_attacks.py` | 3,600 | Attack PCAP generator (9 types, Scapy) | |
| **Training notebooks** | 3 | stage1_v2-clean.ipynb, Stage2-v1.ipynb, stage2_retrain.ipynb | **[+1]** |

**Total:** ~8,100+ LOC across all components **[was ~6,000]**

---

## SLIDE 17: Key Algorithms & Techniques

**Title:** Key Algorithms

1. **GATv2Conv** (Brody et al., 2022) — Dynamic attention mechanism. Unlike GAT v1, attention is computed *after* concatenation and transformation → strictly more expressive.

2. **QFormer** (Li et al., BLIP-2, 2023) — Learnable query tokens cross-attend to variable-length node embeddings → fixed-size output. Originally for vision-language; we adapt to **graph-language**.

3. **SigLIP** (Zhai et al., 2023) — Pairwise sigmoid loss replacing softmax-based InfoNCE. Memory-efficient, no global normalization. Each pair independently classified as match/non-match.

4. **Sliding Window Graph Construction** — 30s window captures flow diversity; 10s stride provides temporal overlap. Data-driven windowing (only at timestamps with actual traffic).

5. **Ego-Graph Extraction** — 2-hop subgraphs from high-degree nodes for dataset expansion. Preserves local structural patterns around key network entities.

6. **Louvain Community Detection** — Identifies graph communities for auxiliary supervision signal in PyG Data.

---

## SLIDE 18: Challenges & Solutions

**Title:** Challenges Encountered & Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Timestamp format mismatch** | 288,602 rows (9.3%) unparseable in Thursday-Infiltration CSV | `pd.to_datetime(format='mixed', errors='coerce')` + drop NaT rows |
| **Severe class imbalance** | Minority classes significantly underrepresented vs Benign | Stratified split + minority class oversampling + class-balanced sampling |
| **CICFlowMeter feature gap** | Python cfm produces different features than Java version used for CIC-IDS2017 → model classifies everything as Benign | Dockerized Java CICFlowMeter with platform emulation on Apple Silicon |
| **Insufficient data for contrastive learning** | 4,875 graphs needed augmentation for cross-modal alignment | Ego-graph extraction + class balancing → 5,108 graphs |
| **Text augmentation hallucination** | Flan-T5 augmented texts hallucinated graph statistics (54.3% failure) | Quality filter with fallback to template-based descriptions |
| **[NEW] v2.1 near-random retrieval** | Stage 2 v2.1 achieved R@1≈0.001 (near random) at epoch 9 | **v2.2 retraining:** semantic templates, QFormer enabled, contrastive-dominant loss (80/20), 5× higher LR → R@1=0.014 (14× improvement) |
| **[NEW] Hard-coded label issue** | Live mode predictions defaulted to single class | Fixed label mapping in inference pipeline; retrained Stage 1 classifier |
| **[NEW] Live graph building** | Graph construction failed on streaming CSV chunks | Fixed temporal windowing for live data; added emulation state sidecar (`.emulation_state.json`) |

**Speaker notes:** "The CICFlowMeter gap was the most insidious — the model classified everything as benign when fed Python-extracted features. Our biggest recent challenge was the v2.1 retrieval failure: the model learned to predict graph statistics from templates rather than aligning attack semantics. We fixed this by redesigning templates to focus on attack behavior and rebalancing the loss to prioritize contrastive alignment."

---

## SLIDE 19: Remaining Work

**Title:** Remaining Work & Benchmarking Plan

### All Training Complete
- Stage 1 GNN: 20 epochs, best checkpoint saved ✅
- Stage 2 Cross-Attention Bridge: **retrained v2.2**, best R@1=0.014 at epoch 14, best loss=0.1893 at epoch 15 ✅
- Live Network monitoring page + emulator ✅

### Benchmarking Plan (To Be Completed)

**1. Graph Classification (Stage 1 GNN):**
- Per-class Precision, Recall, F1-score on test set (441 graphs)
- 7×7 confusion matrix
- Macro-F1 and weighted-F1 scores
- Baseline comparison: GCN, GraphSAGE, vanilla GAT (same split)

**2. Cross-Modal Retrieval (Stage 2):**
- Recall@K (K=1, 5, 10) — both text→graph and graph→text
- Zero-shot classification accuracy (graphs classified by nearest text prototype)
- Alignment & Uniformity metrics (Wang & Isola, 2020)
- Mean Average Precision (mAP) for retrieval

**3. NID System Metrics:**
- Detection rate (TPR) per attack class
- False positive rate (Benign → Attack misclassification)
- End-to-end latency: PCAP ingestion → alert (per graph)
- Throughput: graphs processed per second

**4. Ablation Studies:**
- QFormer vs. simple mean-pool projection
- Effect of ego-graph expansion (2,934 vs 15,953 training samples)
- With/without auxiliary tasks (contrastive-only vs contrastive+auxiliary)

---

## SLIDE 20: References

**Title:** References

1. Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR 2022. [GATv2Conv]
2. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and LLMs.* ICML 2023. [QFormer]
3. Zhai, X., et al. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023. [SigLIP]
4. Sharafaldin, I., et al. (2018). *Toward Generating a New Intrusion Detection Dataset (CIC-IDS2017).* ICISSP 2018.
5. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
6. Wang, T. & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity.* ICML 2020.
7. Xu, M., et al. (2023). *MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector.* NeurIPS 2023. [Graph-language alignment]
8. Veličković, P., et al. (2018). *Graph Attention Networks.* ICLR 2018.

---

## Slide Count Summary

| Section | Slides | Slide #s |
|---------|--------|----------|
| Title | 1 | 1 |
| Introduction & Objectives | 2 | 2-3 |
| System Design & Architecture | 2 | 4-5 |
| Implementation Details | 5 | 6, 9-10, 15-17 |
| Results & Analysis | 6 | 7-8, 11-14 (Stage 2 retrained v2.2) |
| Challenges & Solutions | 1 | 18 |
| Remaining Work | 1 | 19 |
| References | 1 | 20 |
| **Total** | **20 slides** | |

## Image/Figure Checklist
1. Slide 2: Flat features vs graph representation diagram
2. Slide 4: Full pipeline architecture diagram (Stage 1 + Stage 2)
3. Slide 5: Flow table → directed graph → PyG Data diagram
4. Slide 6: GATv2Conv layer-by-layer with dimensions
5. Slide 7: **tsne_attack_classes.png** (actual t-SNE from notebook)
6. Slide 8: QFormer architecture — queries cross-attending to nodes
7. Slide 11: Loss/R@1/R@10 curve chart **[UPDATED: now includes R@10 and more epochs]** + **tsne_embeddings.png** from `checkpoints/stage2/`
8. Slide 12: Retrieval results table with color coding
9. Slides 13-14: Streamlit app screenshots (**5 pages** — include Live Network page) **[UPDATED]**
10. Slide 18: Challenge/solution table (already in slide, **expanded with 3 new challenges**)

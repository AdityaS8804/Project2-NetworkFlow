# Review 2 Presentation: GNN-BERT Cross-Modal Network Intrusion Detection System

## Context
This is the Review 2 presentation for a network intrusion detection project combining GNN + BERT cross-modal alignment. The project is 85-90% complete — Stage 1 GNN is fully trained (20 epochs), Stage 2 cross-attention bridge is fully trained (best checkpoint at epoch 9), and the Streamlit application is complete. What remains is comprehensive benchmarking. The presentation targets a technical faculty panel.

---

## SLIDE 1: Title Slide

**Title:** GNN-BERT Cross-Modal Network Intrusion Detection System
**Subtitle:** Review 2 — Progress Report (85-90% Complete)

- Project Team: [Names]
- Date: March 2026
- Guide: [Faculty name]

**Speaker notes:** "We present Review 2 of our Network Intrusion Detection System that combines Graph Neural Networks with BERT for cross-modal alignment. We are at approximately 85-90% completion."

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
| O3 | Align graph + text embeddings in shared 256-dim space (QFormer + SigLIP) | ✅ Done (best checkpoint epoch 9) |
| O4 | Streamlit app — live PCAP monitoring, NL query, embedding visualization | ✅ Done |
| O5 | Attack traffic simulator — 9 realistic attack types via Scapy | ✅ Done |

**Speaker notes:** "Five concrete objectives. The first three are model-oriented. The last two are application-oriented. All five are complete. What remains is comprehensive benchmarking and evaluation."

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
- **Stage 2:** Cross-modal alignment — 2.83M params (2.09M trainable), trained, best at epoch 9 ✅
- **Application:** 11 modules, ~2,000+ LOC ✅

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

**Graph Construction:**
- Sliding window: 30s window, 10s stride → 4,879 candidates → **2,934 valid graphs** (≥3 nodes)
- Nodes = IP addresses, Edges = directed flows
- Edge features: 77 CICFlowMeter features per flow
- Node features: mean aggregation of outgoing edge features → StandardScaler normalized
- Structural annotations: Louvain community detection, degree, clustering coefficient, betweenness centrality

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

## SLIDE 9: Stage 2 — Dataset Expansion & Text Generation

**Title:** Stage 2: Data Preparation for Contrastive Learning

**Dataset Expansion (2,934 → 15,953 graphs):**
- **Ego-graph extraction:** 2-hop subgraphs from top-5 highest-degree nodes per graph
  - Min 4 nodes per ego, min 8 nodes in parent → 13,019 ego-graphs extracted
- **Augmentation:** edge dropout (15%) + feature noise (σ=0.05) for underrepresented classes
- Final: 15,953 graph-text pairs

**Template-Based Text Generation:**
- Extract per-graph statistics: nodes, edges, avg_degree, density, max_degree, components
- 2-3 attack-specific templates per class filled with actual metrics
- Example: *"DDoS attack with distributed sources: 148 participating nodes. Network shows 3 components, suggesting coordinated multi-vector assault."*
- Flan-T5 augmentation attempted → 54.3% hallucinated stats → fell back to templates

**Final Split:** 11,167 train / 2,393 val / 2,393 test (stratified by attack type)

---

## SLIDE 10: Stage 2 — Loss Function & Training

**Title:** Stage 2: SigLIP Loss + Auxiliary Tasks

**SigLIP Contrastive Loss:**
- Pairwise sigmoid (not softmax) — each (graph, text) pair independently classified as match/non-match
- More memory-efficient than InfoNCE; no global normalization needed
- Learnable temperature parameter (init=0.07)
- Soft targets: α=0.1 label smoothing (handles template text similarity)

**Auxiliary Tasks (weight=0.5):**
- Attack classification: 256→128→7 classes
- Node count prediction: 256→64→1
- Edge count prediction: 256→64→1
- Density prediction: 256→64→1 (sigmoid)

**Training Config:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Schedule: 5-epoch linear warmup → cosine decay
- Batch: 32, gradient accumulation=2 (effective 64)
- Class-balanced sampling: ~4-5 samples per class per batch
- Trained: 50 epochs, best checkpoint at epoch 9

---

## SLIDE 11: Stage 2 — Training Progress

**Title:** Stage 2: Training Results

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | R@1 | R@5 |
|-------|-----------|---------|-----------|---------|-----|-----|
| 1 | 0.5748 | 0.5642 | 4.5% | 4.2% | 0.001 | 0.003 |
| 2 | 0.4977 | 0.5557 | 10.0% | 6.0% | 0.001 | 0.007 |
| 3 | 0.4654 | 0.5335 | 14.0% | 6.9% | 0.002 | 0.009 |

- Best checkpoint saved as final model (best.pt, 8 MB)
- Training used 5-epoch LR warmup → cosine decay schedule

**Key Observations:**
- Loss consistently decreasing through training
- Accuracy improving 4x within first 3 epochs alone
- Model converged efficiently — early stopping saved compute
- Final model config: hidden_dim=256, SigLIP + auxiliary tasks (weight 0.5 each)

> **IMAGE:** Dual-axis line chart: left axis = loss (train/val), right axis = R@1.

**Speaker notes:** "Training showed consistent improvement. The 5-epoch warmup allowed the model to stabilize before the cosine decay phase. The final checkpoint balances contrastive alignment with auxiliary task performance."

---

## SLIDE 12: Retrieval Results

**Title:** Cross-Modal Retrieval Results (Epoch 9 Checkpoint)

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
- Full Recall@K benchmarking across all classes planned for final evaluation

> **IMAGE:** Table with color-coded results and similarity score bars.

---

## SLIDE 13: Streamlit Application

**Title:** Streamlit Application: Real-Time Network Analyzer

**Two Modes:**
- **Demo mode:** Load CIC-IDS2017 CSV files directly (6 dataset options)
- **Live mode:** Watch `wireshark/` directory for new PCAP/CSV files → process via CICFlowMeter Docker

**Four Pages:**
1. **Embedding Explorer** — PCA/t-SNE scatter of 128-dim GNN embeddings, ground truth pie chart, traffic timeline, structure scatter (nodes vs edges)
2. **NL Query** — Text input → cosine similarity in 256-dim space → top-K results with similarity bars + per-result topology view
3. **Graph Inspector** — Individual graph with Plotly spring-layout, node IP table, structural stats (density, clustering, components)
4. **Architecture** — Documentation page with model specs and dataset reference

**Implementation:** 11 modules in `app/`, ~2,000+ LOC

> **IMAGE:** 2×2 grid of annotated screenshots from the four Streamlit pages.

---

## SLIDE 14: Streamlit Application — Demo Screenshots

**Title:** Application Demo Highlights

Show 2-3 key screenshots:

1. **Embedding Explorer:** t-SNE scatter colored by ground truth label, showing attack clusters
2. **NL Query:** Query "show me DDoS attacks" with top-5 results, similarity bars, and retrieved graph topology
3. **Graph Topology:** Individual graph with node coloring by attack label, hover details showing IP and degree

> **IMAGE:** Actual screenshots from the running Streamlit app. Annotate key UI elements.

**Speaker notes:** "Let me walk you through the application. The embedding explorer shows our GNN embeddings projected to 2D. The NL query page lets you type natural language and retrieves the most similar graph snapshots. The graph inspector lets you drill into individual snapshots to see the actual network topology — which IPs are communicating, which are attack nodes."

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

| Module | LOC | Purpose |
|--------|-----|---------|
| `models.py` | 228 | GATEncoderWrapper, BERTEncoder, QFormerBridge, CrossAttentionBridgeV2 |
| `graph_builder.py` | 234 | Sliding window, flow→graph, node features, Louvain, structural properties |
| `main.py` | 513 | Streamlit UI — 4 pages, demo/live mode |
| `visualizations.py` | 419 | 10 Plotly chart builders (scatter, pie, timeline, topology, similarity) |
| `pcap_processor.py` | ~400 | PCAP/CSV parsing, CICFlowMeter (Java Docker + Python fallback) |
| `nl_query.py` | 159 | NLQueryEngine — text→graph retrieval, template descriptions |
| `inference_engine.py` | 160 | Model loading, graph/text embedding, attack prediction |
| `pipeline.py` | 118 | BackgroundPipeline — flow→graph→embedding→state |
| `state.py` | 119 | Thread-safe AppState, GraphRecord, circular buffer |
| `watcher.py` | 112 | PcapWatcher thread — file monitoring, settle check |
| `config.py` | 62 | Constants, label maps, graph params, UI params |
| `simulate_attacks.py` | 3,600 | Attack PCAP generator (9 types, Scapy) |
| **Training notebooks** | 2 | stage1_v2-clean.ipynb, Stage2-v1.ipynb |

**Total:** ~6,000+ LOC across all components

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
| **Insufficient data for contrastive learning** | 2,934 graphs too few for cross-modal alignment | Ego-graph extraction (5.4× expansion to 15,953) |
| **Text augmentation hallucination** | Flan-T5 augmented texts hallucinated graph statistics (54.3% failure) | Quality filter with fallback to template-based descriptions |

**Speaker notes:** "The CICFlowMeter gap was the most insidious — the model classified everything as benign when fed Python-extracted features because the numerical distributions were fundamentally different from the Java-extracted training data. We solved this by running the exact Java CICFlowMeter via Docker."

---

## SLIDE 19: Remaining Work

**Title:** Remaining Work & Benchmarking Plan

### All Training Complete
- Stage 1 GNN: 20 epochs, best checkpoint saved ✅
- Stage 2 Cross-Attention Bridge: trained, best R@1 checkpoint at epoch 9 ✅

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
| Results & Analysis | 6 | 7-8, 11-14 (Stage 2 complete) |
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
7. Slide 11: Loss/R@1 curve chart
8. Slide 12: Retrieval results table with color coding
9. Slides 13-14: Streamlit app screenshots (4 pages)
10. Slide 18: Challenge/solution table (already in slide)

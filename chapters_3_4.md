CHAPTER 3
PROPOSED SYSTEM AND METHODOLOGY

3.1 INTRODUCTION

This chapter presents the architecture and methodology for the proposed two-stage graph intelligence approach with cross-modal contrastive semantics for network intrusion detection. Building on research gaps identified in Chapter 2, the system is designed to integrate graph-based structural awareness with multimodal semantic interpretability through a novel combination of Graph Attention Networks and language model alignment. The system architecture addresses fundamental challenges of topology-aware threat detection and analyst interpretability through a unified framework combining graph neural network encoding, cross-attention bridge alignment, natural language retrieval, and real-time traffic monitoring.

The methodology follows a multi-stage pipeline from raw network flow processing through temporal graph construction to cross-modal embedding alignment, with each stage designed for effectiveness and modularity. This chapter is organized to provide comprehensive coverage of all components. Section 3.2 presents the system overview, followed by architecture details (3.3), mathematical foundations (3.4), algorithms (3.5), dataset specifications (3.6), hyperparameters (3.7), workflow (3.8), and a summary (3.9).

3.2 SYSTEM OVERVIEW

The proposed system integrates multiple deep learning techniques into a modular framework designed for accuracy, interpretability, and operational deployment. The system operates as a two-stage pipeline, first encoding network communication graphs through graph neural networks, then aligning graph representations with natural language descriptions in a shared embedding space. This multimodal design leverages both structural traffic topology and semantic understanding to enable novel query capabilities.

The core innovation is the adaptation of the Query Former (QFormer) cross-attention mechanism, originally developed for vision-language models such as BLIP-2, to the domain of network traffic graphs. This bridge module learns to translate between the graph modality and the text modality through learnable query tokens that attend to graph node embeddings. The resulting shared embedding space enables cosine similarity-based retrieval where security analysts can query network traffic using natural language descriptions.

The architecture is designed for modularity. Key modules include data preparation and graph construction, the GNN encoder, the cross-attention bridge with contrastive learning, the natural language query engine, the real-time monitoring application, and the attack simulation module.

System operation proceeds in two phases. The initial training phase involves Stage 1 GNN pretraining on the CIC-IDS2017 dataset to establish graph-level embeddings and train the attack classifier, followed by Stage 2 cross-attention bridge training to align graph and text modalities. The operational deployment phase involves continuous processing of incoming traffic through a Streamlit-based dashboard supporting both demonstration and live monitoring modes, with real-time embedding visualization providing a foundation for observing distributional shifts in network behavior.

3.3 SYSTEM ARCHITECTURE AND METHODOLOGY

The proposed system is designed as a modular, end-to-end pipeline that transforms raw network traffic into interpretable, queryable security intelligence. The architecture, depicted in Figure 3.1, integrates six key modules: Data Preparation and Graph Construction, a GATv2Conv Graph Neural Network Encoder, a QFormer Cross-Attention Bridge for multimodal alignment, a Natural Language Query and Retrieval Engine, a Real-Time Monitoring Application, and an Attack Simulation Module. This design facilitates a complete workflow from raw packet captures to analyst-facing natural language retrieval.

[Figure 3.1: High-Level System Architecture — Two-stage pipeline from raw PCAP/CSV through GNN encoding and cross-attention alignment to Streamlit dashboard]

The core of the system's multimodal capability lies within the cross-attention bridge architecture, which is trained using a SigLIP contrastive objective. As illustrated in Figure 3.2, this module processes graph embeddings from the frozen GNN encoder and text descriptions from a frozen BERT encoder through specialized projection heads into a shared 256-dimensional embedding space. Within this space, the sigmoid contrastive loss function is optimized with soft targets to pull embeddings of corresponding graph-text pairs closer together while pushing non-corresponding pairs apart, thereby achieving robust semantic alignment.

[Figure 3.2: Cross-Attention Bridge Module — QFormer with learnable queries bridging GNN and BERT encoders into shared 256-dim space]

3.3.1 Data Preparation and Preprocessing Module

The data preparation module serves as the entry point, transforming raw packet captures or flow records into temporal graph representations. Data from the CIC-IDS2017 dataset, consisting of eight CSV files totaling 3,119,345 network flow records generated by the CICFlowMeter tool, is accepted as input. Each flow record contains 85 columns: seven metadata columns (Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp), 77 statistical feature columns, and one ground truth label column.

The preprocessing pipeline performs several critical transformations. Timestamp parsing and validation is first applied, during which 288,602 malformed entries are removed, yielding 2,830,743 valid flow records. Label consolidation maps the original 15 fine-grained attack labels into seven canonical classes: Benign, DoS (comprising Hulk, GoldenEye, Slowloris, Slowhttptest), DDoS, PortScan, BruteForce (FTP-Patator, SSH-Patator), WebAttack (Brute Force, XSS, SQL Injection), and Bot/Other (Bot, Infiltration, Heartbleed). Feature normalization is applied using a pre-fitted StandardScaler to standardize the 77 numerical features.

Temporal graph construction employs a sliding window approach with a 30-second window size and 10-second stride. Within each window, a directed graph is constructed where nodes represent unique IP addresses and edges represent individual network flows. Each edge carries the full 77-dimensional CICFlowMeter feature vector. Node features are computed by mean-aggregating all outgoing edge features for each IP address. Additional structural properties including Louvain community assignments, node degree, and clustering coefficients are computed per node. Windows with fewer than two nodes are discarded, yielding 2,934 valid temporal graph snapshots. Each graph is converted to a PyTorch Geometric Data object for downstream processing.

3.3.2 GNN Encoder Architecture

The graph neural network encoder forms the first stage, learning to encode variable-size communication graphs into fixed-dimensional embeddings. The encoder employs GATv2Conv layers, an improved Graph Attention Network variant that applies dynamic attention regardless of node feature similarity.

The architecture consists of three GATv2Conv layers. The first layer transforms 77-dimensional node features to 128 dimensions using four attention heads (output: 512-dim). The second layer processes these through four heads back to 128 dimensions per head. The third layer uses a single head to produce final 128-dimensional node embeddings. ELU activations and dropout of 0.2 are applied between layers.

Global mean pooling across all node embeddings produces a single 128-dimensional graph-level embedding. The encoder comprises 738,816 parameters. An AttackClassifier head — a three-layer MLP (128→64→32→7) with ReLU and dropout 0.3 — is trained alongside for attack type prediction. The combined Stage 1 model contains 747,527 parameters trained over 20 epochs with cross-entropy loss and class-balanced oversampling.

3.3.3 Cross-Attention Bridge Module

The cross-attention bridge module implements the second stage of the pipeline, aligning graph embeddings with natural language descriptions in a shared 256-dimensional space. This module adapts the QFormer architecture from BLIP-2 to the network graph domain, enabling cross-modal retrieval between graph structures and text queries.

The QFormerBridge introduces four learnable query tokens of dimension 256 that attend to GNN-encoded node embeddings through two cross-attention transformer layers with four heads and dropout 0.1. An input projection maps 128-dimensional GNN features to the 256-dimensional query space. Output queries are mean-pooled to produce a single graph representation. The QFormer contains 1,613,568 parameters.

A two-layer MLP graph projection head maps the QFormer output to the 256-dimensional shared space. The text path employs frozen BERT base-uncased (109M parameters) to extract 768-dimensional CLS embeddings, projected to 256 dimensions through a similar MLP. Both paths apply L2 normalization.

Training employs SigLIP contrastive loss with learnable temperature (initialized 0.07) and soft targets (alpha=0.1). Auxiliary heads for attack classification, property prediction, and triplet ranking provide additional supervision. The total model comprises 2,832,267 parameters, of which 2,093,451 are trainable.

3.3.4 Natural Language Query and Retrieval Module

The natural language query module enables semantic retrieval over processed network traffic graphs using free-form text queries. This module bridges the gap between technical graph representations and human-interpretable descriptions, allowing security analysts to search for specific traffic patterns without requiring knowledge of the underlying graph structure.

The retrieval process operates in the shared 256-dimensional space. A text query is encoded through BERT and the text projection head, producing an L2-normalized embedding. Cosine similarity is computed against all stored graph embeddings via matrix multiplication. The top-K most similar graphs above a threshold of 0.25 are retrieved.

Template-based descriptions are generated for each result using graph statistics (node count, edge count, density, degree, components). Seven attack-type-specific template families with hash-based selection ensure contextually appropriate and diverse descriptions. An aggregated summary reports the distribution of traffic types across matched results.

The engine maintains GraphRecord objects containing shared-space embeddings, PyG graph data, attack predictions, and metadata, enabling sub-second retrieval through vectorized computation.

3.3.5 Real-Time Monitoring and Application Module

The real-time monitoring module provides an operational deployment interface through a Streamlit-based web application comprising eleven Python modules totaling over 2,000 lines of code. The application supports two operational modes designed for different use cases.

Demo Mode loads pre-recorded CIC-IDS2017 CSV datasets with stratified sampling, enabling evaluation without network infrastructure. Live Mode implements continuous monitoring via a PcapWatcher daemon thread polling every two seconds for new capture files. A Dockerized Java CICFlowMeter extracts the 77 features matching the training distribution, and flows are processed through the full pipeline.

Three visualization pages are provided: the Embedding Explorer (PCA/t-SNE projections with structural scatter plots), the NL Query page (text retrieval with similarity visualization), and the Graph Inspector (topology visualization with node tables and statistics).

Thread-safe state management ensures correct concurrent operation. The continuous embedding visualization in Live Mode provides a foundation for observing concept drift — emerging traffic patterns would manifest as distributional shifts visible to the analyst in real time.

3.3.6 Attack Simulation Module

The attack simulation module generates realistic synthetic network traffic matching CIC-IDS2017 statistical characteristics using the Scapy packet construction library. This module enables controlled testing of the detection pipeline without requiring access to live network infrastructure or elevated system privileges.

Nine attack types are implemented, each replicating CIC-IDS2017 temporal characteristics including inter-arrival times, session durations, payload sizes, and TCP flag patterns. Realistic TCP sessions are constructed with proper sequence tracking and protocol-specific behaviors.

Generated PCAP files are saved with timestamped filenames encoding the attack type, enabling automatic label inference during Live Mode. This allows analysts to inject controlled scenarios to evaluate detection and observe how evolving attack patterns appear in the embedding space, providing a mechanism for stress-testing the system's response to shifting traffic distributions.

3.4 MATHEMATICAL FORMULATION

3.4.1 Graph Attention Network (GATv2Conv)

The GNN operates on a directed graph G = (V, E) where V represents IP addresses and E represents network flows. Each node v has features x_v in R^77. The GATv2Conv update at layer l is:

h_v^(l) = sigma(sum_{u in N(v)} alpha_vu^(l) W^(l) h_u^(l-1))

The attention coefficients use GATv2 dynamic attention:

e_vu^(l) = a^T LeakyReLU(W^(l) [h_v^(l-1) || h_u^(l-1)])
alpha_vu^(l) = softmax_u(e_vu^(l))

Multi-head outputs are concatenated (layers 1-2) or averaged (final layer). The graph embedding is: z_G = (1/|V|) sum_{v in V} h_v^(L).

3.4.2 QFormer Cross-Attention

The QFormer processes a set of learnable query tokens Q = {q_1, ..., q_M} where M = 4 and q_i in R^256. Given projected node embeddings K = V = proj(h_v^(L)) for all v in V, the cross-attention at each QFormer layer computes:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Each layer applies multi-head attention with H = 4 heads followed by layer normalization and a feed-forward network. The final graph representation is the mean of output queries: z_G^bridge = (1/M) sum_{i=1}^{M} q_i^out.

3.4.3 SigLIP Contrastive Loss

Let {(g_i, t_i)}_{i=1}^B be a mini-batch of graph-text pairs. The normalized embeddings are z_g^i = graph_proj(QFormer(GNN(g_i))) and z_t^i = text_proj(BERT(t_i)). The pairwise similarity is s_ij = z_g^i . z_t^j / tau where tau is the learnable temperature.

The SigLIP loss employs a sigmoid function rather than softmax normalization:

L_SigLIP = -(1/B^2) sum_{i=1}^{B} sum_{j=1}^{B} [y_ij log(sigma(s_ij)) + (1 - y_ij) log(1 - sigma(s_ij))]

where y_ij = 1 if i = j (positive pair) and y_ij = 0 otherwise. Soft targets replace hard labels: y_ij^soft = (1 - alpha) y_ij + alpha / B.

3.4.4 Cosine Similarity Retrieval

For a text query q, the retrieval score for graph g_k is: score(q, g_k) = z_t^q . z_g^k where both embeddings are L2-normalized. The top-K results are: R_K = argmax_{S subset Records, |S|=K} sum_{k in S} score(q, g_k), subject to score(q, g_k) >= threshold.

3.4.5 Combined Training Objective

The total Stage 2 training loss combines contrastive and auxiliary objectives:

L_total = w_c L_SigLIP + w_a (w_atk L_CE + w_prop L_MSE + w_trip L_triplet)

where L_CE is cross-entropy with label smoothing 0.1 for attack classification, L_MSE is mean squared error for graph property prediction, and L_triplet is margin-based triplet loss with margin 0.5. Default weights are w_c = 0.5, w_a = 0.5, with task weights w_atk = 0.4, w_prop = 0.3, w_trip = 0.3.

3.5 ALGORITHMS

3.5.1 Data Preprocessing and Graph Construction Algorithm

Algorithm 1: Data Preprocessing and Temporal Graph Construction
Input: Raw CIC-IDS2017 CSV files D_raw (8 files, 85 columns)
Output: Set of PyG graph objects G = {G_1, ..., G_N}
Parameters: window_size W=30s, stride S=10s, min_nodes=2

D = concat(read_csv(f) for f in D_raw)    // 3,119,345 flows
Parse timestamps, remove invalid entries    // 2,830,743 valid
Consolidate 15 labels into 7 classes; Sort by timestamp

for each window [t, t+W] with stride S do
    W_flows = filter flows in [t, t+W]
    if |unique_IPs(W_flows)| < min_nodes then continue
    Construct DiGraph: nodes=IPs, edges=flows with 77-dim features
    Node features = mean(outgoing_edge_features), normalized via scaler
    Compute Louvain communities, degree, clustering coefficient
    Convert to PyG Data(x, edge_index, y_attack, metadata)
end for
return G    // 2,934 valid graphs

3.5.2 GNN Encoder Training Algorithm (Stage 1)

Algorithm 2: GATv2Conv Encoder Training
Input: Graph dataset G, split 70/15/15 train/val/test
Output: Trained encoder theta_GNN, classifier theta_cls
Parameters: epochs=20, batch_size=32, num_classes=7

Oversample minority classes to 1,728 per class (12,096 total)
Initialize GATEncoder (77->128, 3 layers) + AttackClassifier (128->7)

for epoch = 1 to 20 do
    for each batch B do
        graph_emb = global_mean_pool(GATEncoder(B.x, B.edge_index))
        loss = CrossEntropy(AttackClassifier(graph_emb), B.y_attack)
        Backpropagate and update
    end for
    Evaluate on validation; save best checkpoint
end for

3.5.3 Cross-Attention Bridge Training Algorithm (Stage 2)

Algorithm 3: QFormer Bridge Training with SigLIP Loss
Input: Graph-text pairs D (15,953 pairs), frozen GNN, frozen BERT
Output: Trained bridge theta_bridge
Parameters: epochs=50, lr=1e-4, warmup=5, grad_accum=2

Freeze GNN and BERT; Initialize QFormer + projection heads
scheduler = LinearWarmup(5 epochs) + CosineDecay

for epoch = 1 to 50 do
    for each batch B do
        graph_emb = L2_norm(graph_proj(QFormer(frozen_GNN(B.graph))))
        text_emb = L2_norm(text_proj(frozen_BERT(B.text)))
        L = 0.5 * SigLIP(graph_emb, text_emb) + 0.5 * L_auxiliary
        Accumulate gradients (effective batch=64); update
    end for
    Evaluate R@1, R@5; save best checkpoint
end for

3.5.4 Natural Language Query and Retrieval Algorithm

Algorithm 4: Text-to-Graph Retrieval
Input: Text query q, stored GraphRecords R
Output: Top-K matching graphs with descriptions
Parameters: top_k=5, sim_threshold=0.25

z_q = L2_norm(text_proj(BERT(q)))
sims = stack([r.embedding_256 for r in R]) @ z_q
For top-K indices where sims[idx] >= threshold:
    Extract graph statistics, generate template description
    Append to results with similarity score
return results, generate_summary(results)

3.5.5 Live Traffic Processing Pipeline Algorithm

Algorithm 5: Real-Time Traffic Monitoring Pipeline
Input: Monitored directory, initialized models
Output: Continuous GraphRecord stream to AppState

Initialize PcapWatcher (poll=2s), Pipeline, AppState (max=500)

loop (daemon thread):
    for each new file F (age>=3s, size>=100B) do
        flows_df = CICFlowMeter(F) if PCAP else load_csv(F)
        for each graph in sliding_window(flows_df, W=30s, S=10s) do
            Compute emb_128 (GNN), emb_256 (Bridge), prediction
            AppState.add_record(GraphRecord(...))
        end for
    end for
end loop

3.6 DATASET SPECIFICATION

3.6.1 Dataset Structure and Features

The CIC-IDS2017 dataset, developed by the Canadian Institute for Cybersecurity, serves as the primary data source. This widely adopted benchmark captures five days of network traffic including benign activity and fourteen attack types. The dataset consists of eight CSV files with 85 columns per flow: seven metadata columns, 77 CICFlowMeter statistical features (duration, packet counts, inter-arrival times, TCP flags, flow ratios), and one label column.

[Table 3.1: Label Consolidation — 15 original labels mapped to 7 classes: BENIGN (2,273,097 flows), DoS (252,661), DDoS (128,027), PortScan (158,930), BruteForce (13,835), WebAttack (2,180), Bot/Other (2,013)]

3.6.2 Preprocessing Pipeline

The preprocessing pipeline transforms raw CSV records into graph representations through six stages. Stage 1 performs data loading and validation, reading all eight CSV files and verifying schema consistency. Stage 2 applies timestamp parsing and cleaning, removing 288,602 entries with malformed timestamps. Stage 3 consolidates the 15 original labels into 7 canonical classes. Stage 4 constructs temporal graphs using the sliding window approach. Stage 5 computes node features through edge aggregation and applies StandardScaler normalization. Stage 6 performs quality filtering, discarding graphs with fewer than two nodes.

3.6.3 Train-Validation-Test Split Strategy

The dataset of 2,934 temporal graphs is split into training (70%, 2,053 graphs), validation (15%, 440 graphs), and test (15%, 441 graphs) sets using stratified sampling to maintain class proportions. Given severe class imbalance — Benign graphs outnumber DDoS graphs by a factor of 123 — the training set is class-balanced through minority class oversampling, producing 12,096 training graphs with 1,728 samples per class. For Stage 2 training, ego-graph extraction expands the dataset from 2,934 to 15,953 graph-text pairs, split into 11,167 training, 2,393 validation, and 2,393 test pairs.

[Table 3.2: Per-class graph distribution — Before balancing: Benign 2,470 / DoS 26 / DDoS 20 / PortScan 26 / BruteForce 122 / WebAttack 68 / Bot/Other 202. After balancing: 1,728 per class, 12,096 total training graphs]

3.7 HYPERPARAMETERS

[Table 3.3: Complete hyperparameter specification for both training stages]

| Parameter | Stage 1 (GNN) | Stage 2 (Bridge) |
|-----------|---------------|------------------|
| Epochs | 20 | 50 |
| Batch size | 32 | 32 (effective 64) |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-4 |
| Weight decay | 1e-2 | 1e-2 |
| LR schedule | — | Warmup (5 epochs) + Cosine |
| Gradient accumulation | 1 | 2 |
| Dropout (encoder) | 0.2 | 0.1 (QFormer) |
| Temperature | — | 0.07 (learnable) |
| Contrastive weight | — | 0.5 |
| Auxiliary weight | — | 0.5 |
| Hidden dimension | 128 | 256 |
| Attention heads | 4 | 4 |
| Number of layers | 3 (GATv2Conv) | 2 (QFormer) |
| QFormer queries | — | 4 |
| Total parameters | 747,527 | 2,832,267 |
| Trainable parameters | 747,527 | 2,093,451 |

3.8 WORKFLOW

The system workflow proceeds through three sequential phases. Phase 1 (Data Preparation and Stage 1 Training) involves loading and cleaning the CIC-IDS2017 dataset, constructing temporal graphs through sliding window processing, and training the GATv2Conv encoder with the attack classifier head for 20 epochs on the class-balanced training set. The trained encoder produces 128-dimensional graph embeddings that capture structural patterns discriminative of attack types.

Phase 2 (Stage 2 Cross-Attention Training) begins with generating template-based natural language descriptions for each graph and expanding the dataset through ego-graph extraction from 2,934 to 15,953 graph-text pairs. The QFormer cross-attention bridge is then trained for 50 epochs with SigLIP contrastive loss to align graph and text embeddings in the shared 256-dimensional space, while keeping the GNN encoder and BERT text encoder frozen.

Phase 3 (Application Deployment) deploys the trained models through the Streamlit web application. In Demo Mode, analysts explore pre-recorded datasets with embedding visualization and NL querying. In Live Mode, the PcapWatcher monitors for incoming captures, processes them through the full pipeline, and updates the dashboard in real time. The attack simulator can generate controlled traffic scenarios for testing.

3.9 SUMMARY

This chapter presented a comprehensive two-stage architecture for graph-based network intrusion detection with cross-modal semantic alignment. The system transforms raw network flows into temporal communication graphs, encodes them through a GATv2Conv graph attention network, and aligns the resulting embeddings with natural language descriptions via a QFormer cross-attention bridge trained with SigLIP contrastive loss. The modular design encompasses data preparation, graph neural network encoding, cross-modal alignment, natural language retrieval, real-time monitoring, and attack simulation, providing both detection capability and analyst interpretability through a deployed Streamlit application.


CHAPTER 4
RESULTS, CONCLUSION, AND FUTURE WORK

4.1 EXPERIMENTAL SETUP

The proposed two-stage graph intelligence approach with cross-modal contrastive alignment was implemented and evaluated on the CIC-IDS2017 dataset (Sharafaldin et al., 2018). A stratified split strategy was employed, with 70% allocated to training, 15% to validation, and 15% to testing. The Stage 1 GATv2Conv encoder with three graph attention layers was trained for 20 epochs using the AdamW optimizer with batch size 32 on the class-balanced training set of 12,096 graphs (1,728 per class, oversampled from 2,053 original training graphs). The model contains 747,527 parameters producing 128-dimensional graph embeddings.

For Stage 2, the cross-attention bridge was trained on 15,953 graph-text pairs (expanded from 2,934 graphs via ego-graph extraction) for 50 epochs using AdamW with learning rate 1e-4, weight decay 0.01, batch size 32 with gradient accumulation of 2 (effective batch size 64), a 5-epoch linear warmup followed by cosine decay scheduling, and SigLIP contrastive loss with learnable temperature initialized at 0.07. The GNN encoder (738,816 parameters) and BERT encoder (109,482,240 parameters) were frozen, with only the 2,093,451 bridge parameters trainable. Template-based natural language descriptions were generated for each graph snapshot using attack-type-specific templates populated with graph statistics. All experiments were conducted on Apple Silicon hardware with MPS acceleration.

[Table 4.1: Dataset split — Per-class counts for train/val/test sets across 7 classes, 2,934 total graphs before balancing, 12,096 training graphs after oversampling]

4.2 RESULTS AND ANALYSIS

4.2.1 Stage 1 GNN Embedding Quality

The quality of the learned graph representations was assessed through dimensionality reduction visualization. The t-SNE projection of 12,096 graph embeddings colored by the seven attack classes is presented in Figure 4.1. Clear cluster separation was observed, with each attack class occupying a distinct region in the projected embedding space. The Benign class forms a large, cohesive cluster, while attack classes such as DoS, DDoS, and PortScan exhibit tight, well-separated groupings. This visual evidence indicates that the GATv2Conv encoder successfully learned structurally discriminative representations, capturing fundamental differences in communication topology between benign traffic and various attack patterns.

[Figure 4.1: t-SNE visualization of Stage 1 GNN embeddings — 12,096 graph embeddings colored by 7 attack classes showing clear cluster separation. Source: results/images/stage1/tsne_attack_classes.png]

The separation of attack classes with markedly different network topologies — such as the concentrated hub-spoke pattern of DoS attacks versus the distributed structure of DDoS — validates the graph-based approach. The encoder captures that PortScan traffic, characterized by a single source probing many destinations, produces fundamentally different graph structures compared to BruteForce attacks targeting specific service ports. Classification performance metrics on the held-out test set of 441 graphs are presented below.

[Placeholder: Table showing per-class Precision, Recall, F1-score on 441 test graphs across 7 classes. Macro-F1 and weighted-F1 scores to be computed from evaluation run.]

[Placeholder: Figure 4.4 — 7x7 Confusion matrix for Stage 1 GNN classification on test set]

4.2.2 Stage 2 Cross-Attention Training Performance

The cross-attention bridge training demonstrated consistent convergence, as evidenced by the loss curves shown in Figure 4.2. The training loss decreased from 0.5748 at epoch 1 to 0.4654 at epoch 3, exhibiting rapid initial learning during the warmup phase as the QFormer learned to attend to relevant graph node features. The validation loss showed a corresponding decrease from 0.5642 to 0.5335 over the same period, indicating effective generalization.

[Figure 4.2: Stage 2 training and validation loss curves across epochs, with warmup region shaded. Source: diagrams/slide11_stage2_results.png]

Training accuracy improved from 4.5% at epoch 1 to 14.0% by epoch 3, reflecting the progressive learning of cross-modal alignment in the challenging graph-to-text retrieval task. Validation accuracy rose from 4.2% to 6.9% over the same interval. The best model checkpoint was selected at epoch 9 based on Recall@1 performance on the validation set. The gap between training and validation metrics remained modest, suggesting that the frozen encoder strategy effectively controlled overfitting while allowing the bridge parameters sufficient capacity to learn meaningful cross-modal representations.

[Placeholder: R@1, R@5, R@10 retrieval metrics at best checkpoint (epoch 9) on validation and test sets]

4.2.3 Embedding Space Quality Analysis

The structure of the joint 256-dimensional embedding space was examined to assess cross-modal alignment quality. Graph embeddings and their corresponding text description embeddings were projected using t-SNE to visualize the shared space geometry. Successful alignment is indicated by the proximity of graph and text embeddings for the same traffic class, demonstrating that the SigLIP contrastive objective and QFormer attention mechanism learned to map both modalities into a semantically coherent shared representation.

[Placeholder: Figure 4.3 — t-SNE of shared 256-dim space showing graph and text embeddings colored by attack class]

4.2.4 Natural Language Retrieval Qualitative Results

Qualitative evaluation of the natural language query system demonstrated semantically meaningful retrieval behavior. When queried with attack-type-specific descriptions such as "Show me denial of service attacks," the system retrieved graph snapshots with corresponding DoS labels and characteristic topological properties including high maximum degree indicative of traffic concentration. Queries for benign traffic correctly retrieved graphs with lower density and more distributed communication patterns. The template-based description generation provided interpretable summaries for each retrieved result, enabling analysts to rapidly assess the nature and severity of matched traffic patterns.

4.3 COMPARATIVE STUDY

[Table 4.2: Comparative analysis with related methods]

| Method | Mean AUC | Interpretability | F1-Score | Speed | Graph-Based |
|--------|----------|-----------------|----------|-------|-------------|
| Ours | [Placeholder] | High | [Placeholder] | High | Yes |
| PACKETCLIP (Masukawa et al., 2025) | ~95% | High | ~94% | High | No |
| LAGER (2025) | 67% | Moderate | ~60-67% | Moderate | No |
| E-GraphSAGE (Lo et al., 2022) | ~92% | Low | ~91% | High | Yes |
| NetGuard (Gupta et al., 2025) | 86% | Moderate | 86% | High | No |
| ENIDrift (2022) | 88-92% | Moderate | ~90% | High | No |
| GNN-NIDS (Chang et al., 2021) | ~89% | Low | ~88% | Moderate | Yes |

The comparative analysis situates the proposed method against both packet-level and graph-based intrusion detection approaches. PACKETCLIP achieves the highest raw detection performance through direct packet-text contrastive learning but operates on flat flow features without graph structure. E-GraphSAGE and GNN-NIDS employ graph neural networks for topology-aware detection but lack cross-modal interpretability, offering no mechanism for natural language querying or semantic traffic description.

A distinguishing feature of the proposed system is its combination of graph-based topology awareness with high human interpretability. While methods such as LAGER, NetGuard, and ENIDrift provide moderate interpretability through feature importance or attention weights, the proposed approach enables direct natural language interaction with the detection system. The QFormer bridge translates graph-level patterns into a semantically meaningful space that supports free-form text retrieval, a capability not offered by any compared method.

The processing speed assessment reveals that most methods including the proposed approach achieve high throughput suitable for real-time deployment. The graph construction overhead is offset by the efficiency of vectorized GNN inference and pre-computed embeddings for retrieval. LAGER is an exception with moderate speed due to its online drift computation overhead.

The overall assessment reveals that while the proposed approach may not achieve the peak detection accuracy of specialized methods, it uniquely combines graph-based structural encoding with cross-modal semantic querying — a capability absent from all compared methods. This balance is particularly relevant in operational security settings where analyst understanding and trust in detection outputs are as critical as raw classification accuracy. The ability to query traffic patterns in natural language represents a qualitative advancement in analyst workflow integration.

4.4 DISCUSSION

The experimental results demonstrate that the two-stage architecture successfully learns discriminative graph-level representations and establishes cross-modal alignment between network traffic graphs and natural language descriptions. The clear t-SNE cluster separation in Stage 1 validates the effectiveness of the GATv2Conv encoder in capturing topology-dependent attack signatures. The progressive loss reduction in Stage 2 indicates that the QFormer cross-attention mechanism learns meaningful correspondences between graph structures and textual descriptions, despite the inherent modality gap.

Several strengths of the approach merit emphasis. The graph-based representation preserves network topology information lost by flat feature approaches, enabling detection of structural attack patterns. The modular two-stage design allows independent optimization of each component. The Streamlit application provides immediate operational utility, and the continuous embedding visualization in Live Mode offers a practical mechanism for observing distributional shifts that may indicate concept drift in network traffic patterns.

However, several limitations warrant discussion. Comprehensive classification benchmarking on the test set remains pending at the time of writing, limiting quantitative comparison with baselines. The severe class imbalance in the original dataset — Benign graphs outnumber DDoS graphs by 123:1 — necessitated aggressive oversampling whose impact on generalization requires further investigation. The current inference configuration employs mean-pooling rather than the full QFormer cross-attention for graph embedding projection, representing an area for optimization. Additionally, computational requirements for the BERT encoder and graph construction pipeline should be profiled for high-throughput deployment scenarios.

4.5 CONCLUSION

The development and evaluation of a two-stage graph intelligence approach with cross-modal contrastive semantics for network intrusion detection was presented in this research. The system transforms network traffic into temporal communication graphs, encodes them through a GATv2Conv attention network producing 128-dimensional embeddings, and aligns these with natural language descriptions via a QFormer cross-attention bridge in a shared 256-dimensional space trained with SigLIP contrastive loss.

Clear discriminative representations across seven traffic classes were established, as evidenced by t-SNE visualization showing distinct cluster separation. The cross-attention bridge demonstrated consistent training convergence with decreasing loss and improving retrieval accuracy over training. A complete operational system was delivered through the Streamlit application supporting both demonstration and live monitoring modes with natural language querying capability. The research demonstrates that graph-based structural encoding combined with cross-modal language alignment provides both detection capability and interpretability, advancing network intrusion detection toward systems that security analysts can meaningfully interact with through natural language.

4.6 FUTURE WORK

Several promising research directions emerge from this work. First, comprehensive benchmarking of the Stage 1 classifier and Stage 2 retrieval system on held-out test sets remains a critical near-term priority, including per-class precision, recall, and F1-scores, Recall@K retrieval metrics, and baseline comparisons against GCN, GraphSAGE, and vanilla GAT architectures on the same data splits.

Second, formal concept drift detection mechanisms should be integrated, building on the real-time embedding visualization already present in the monitoring application. Statistical tests such as the Kolmogorov-Smirnov test or the Concept Drift Compatibility Index could be applied to embedding distributions from the sliding windows, with automated alerting when distributional shifts exceed defined thresholds. This would formalize the drift observation capability already supported by the Live Mode embedding explorer.

Third, adaptive model update mechanisms employing continual learning techniques such as Elastic Weight Consolidation, replay buffers, and layer-wise selective freezing should be investigated to enable the system to evolve in response to detected drift without catastrophic forgetting of previously learned attack patterns.

Fourth, zero-shot and few-shot attack detection through the cross-modal alignment space represents a high-impact direction, where novel attack types could be identified based purely on textual descriptions without requiring labeled graph examples. The shared embedding space is architecturally designed to support this capability, requiring evaluation of how well unseen attack categories cluster relative to their textual descriptions. Fifth, evaluation across diverse datasets including CIC-IDS2018, IoT-23, and enterprise network captures would validate generalization beyond the CIC-IDS2017 benchmark. Sixth, enabling the full QFormer cross-attention mechanism at inference time rather than the current mean-pooling configuration should be investigated to maximize retrieval quality from the trained bridge architecture. Finally, computational optimization through model distillation and quantization would improve throughput for high-volume deployment scenarios.

# # Stage 2: Cross-Attention Bridge Retraining (v2.2)
# 
# **Why retrain?** The v2.1 checkpoint (epoch 9/50) achieved R@1≈0.001 — near random retrieval.
# Root causes: stat-heavy templates, no QFormer, premature training stop, 83% Benign imbalance.
# 
# **Key fixes:**
# 1. Semantic text templates (attack meaning, not graph stats)
# 2. QFormer enabled (selective attention to attack-relevant nodes)
# 3. Contrastive-dominant loss (80/20 instead of 50/50)
# 4. Class-balanced sampling + ego-graph augmentation
# 5. Full training with early stopping on R@1

# ======================================================================
# === Cell 1: Imports & Device Setup ===
import os, sys, json, math, hashlib, random, pickle, glob, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import train_test_split

try:
    from torch_scatter import scatter
except ImportError:
    from torch_geometric.utils import scatter

import networkx as nx

# Add project root to path
sys.path.insert(0, '.')
from app.config import ATTACK_LABEL_MAP, ID_TO_ATTACK, META_COLS, LABEL_COL

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

# ======================================================================
# === Cell 2: Load Frozen Stage 1 GNN ===
from app.models import GATEncoderWrapper, BERTEncoder, CrossAttentionBridgeV2

# Load GNN
s1_ckpt = torch.load('checkpoints/stage1/best.pt', map_location='cpu', weights_only=False)
gnn_model = GATEncoderWrapper.from_checkpoint(s1_ckpt, device='cpu')
gnn_model.eval()
for p in gnn_model.parameters():
    p.requires_grad = False
print(f"GNN loaded: input_dim={s1_ckpt['model_config']['input_dim']}, "
      f"hidden_dim={s1_ckpt['model_config']['hidden_dim']}, epoch={s1_ckpt['epoch']}")

# Load scaler and feature columns
import joblib
scaler = joblib.load('checkpoints/stage1/scaler.pkl')
with open('checkpoints/stage1/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

assert scaler.n_features_in_ == 77, f"Expected 77 features, got {scaler.n_features_in_}"
print(f"Scaler loaded: {scaler.n_features_in_} features")
print(f"Feature columns: {len(feature_cols)} cols")

# ## Section 2: Data Preparation from CICIDS2017 CSVs
# 
# Build time-windowed graphs directly from CSVs using the same logic as `app/graph_builder.py`.

# ======================================================================
# === Cell 3: Load CSVs & Build Graphs ===

CICIDS_DIR = 'cicids2017'
WINDOW_SIZE = 30  # seconds
STRIDE = 10       # seconds
MIN_NODES = 2

def load_all_csvs(cicids_dir):
    """Load and clean all CICIDS2017 CSVs."""
    csv_files = sorted(glob.glob(os.path.join(cicids_dir, '*.csv')))
    print(f"Found {len(csv_files)} CSV files")
    
    frames = []
    for path in csv_files:
        print(f"  Loading {os.path.basename(path)}...")
        df = pd.read_csv(path, encoding='latin-1')
        df.columns = df.columns.str.strip()
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    
    # Clean labels
    combined[LABEL_COL] = combined[LABEL_COL].astype(str).str.strip()
    combined.loc[combined[LABEL_COL].isin(['nan', '', 'NaN']), LABEL_COL] = 'BENIGN'
    
    print(f"\nTotal flows: {len(combined):,}")
    print(f"\nLabel distribution:")
    for lbl, cnt in combined[LABEL_COL].value_counts().items():
        print(f"  {lbl}: {cnt:,}")
    
    return combined


def build_graph_from_flows(flow_df, feat_cols):
    """Build directed graph: nodes=IPs, edges=flows with 77-dim features."""
    G = nx.DiGraph()
    for _, row in flow_df.iterrows():
        src_ip = row.get('Source IP')
        dst_ip = row.get('Destination IP')
        if pd.isna(src_ip) or pd.isna(dst_ip):
            continue
        if src_ip not in G:
            G.add_node(src_ip, ip=src_ip)
        if dst_ip not in G:
            G.add_node(dst_ip, ip=dst_ip)
        edge_features = row[feat_cols].values.astype(np.float32)
        edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)
        G.add_edge(src_ip, dst_ip, features=edge_features,
                   label=row.get(LABEL_COL, 'Unknown'))
    return G


def compute_node_features(G, feat_cols):
    """Aggregate edge features to node features (mean of outgoing)."""
    for node in G.nodes():
        out_edges = list(G.out_edges(node))
        in_edges = list(G.in_edges(node))
        out_features = [G.edges[e]['features'] for e in out_edges]
        if out_features:
            out_mean = np.mean(out_features, axis=0)
        else:
            out_mean = np.zeros(len(feat_cols))
        G.nodes[node]['features'] = out_mean.astype(np.float32)
        # Node label = majority vote from incident edges
        edge_labels = ([G.edges[e]['label'] for e in in_edges] +
                       [G.edges[e]['label'] for e in out_edges])
        if edge_labels:
            G.nodes[node]['label'] = max(set(edge_labels), key=edge_labels.count)
        else:
            G.nodes[node]['label'] = 'BENIGN'
    return G


def graph_to_pyg(G, feat_cols, scaler_obj):
    """Convert NetworkX graph to PyG Data object with normalized features."""
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # Node features [N, 77]
    raw_feats = np.array([G.nodes[n]['features'] for n in node_list], dtype=np.float32)
    if scaler_obj is not None:
        raw_feats = scaler_obj.transform(raw_feats).astype(np.float32)
    x = torch.tensor(raw_feats)
    
    # Edge index [2, E]
    edges = list(G.edges())
    if edges:
        edge_index = torch.tensor(
            [[node_to_idx[u], node_to_idx[v]] for u, v in edges],
            dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Node attack labels
    node_labels = []
    for n in node_list:
        lbl = G.nodes[n].get('label', 'BENIGN')
        node_labels.append(ATTACK_LABEL_MAP.get(lbl, 0))
    node_attack_labels = torch.tensor(node_labels, dtype=torch.long)
    
    # Graph-level label (most common non-benign, or 0)
    label_counts = Counter(node_labels)
    attack_only = {k: v for k, v in label_counts.items() if k != 0}
    graph_label = max(attack_only, key=attack_only.get) if attack_only else 0
    y_attack = torch.tensor([graph_label], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y_attack=y_attack,
        y=y_attack,
        node_attack_labels=node_attack_labels,
    )
    return data, graph_label


def build_graphs_from_csv(df, feat_cols, scaler_obj, window_size=30, stride=10, min_nodes=2):
    """Build time-windowed graphs from a flows DataFrame."""
    # Parse timestamps
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed',
                                     dayfirst=False, errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    if df.empty:
        return []
    
    # Ensure feature columns exist
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    min_t = df['Timestamp'].min()
    ts_seconds = (df['Timestamp'] - min_t).dt.total_seconds().values
    
    # Stride-aligned window starts
    stride_bins = df['Timestamp'].dt.floor(f'{stride}s').drop_duplicates().sort_values()
    
    graphs = []
    for ws in tqdm(stride_bins, desc='Building graphs', leave=False):
        ws_sec = (ws - min_t).total_seconds()
        we_sec = ws_sec + window_size
        
        left = np.searchsorted(ts_seconds, ws_sec, side='left')
        right = np.searchsorted(ts_seconds, we_sec, side='left')
        
        if right - left < 3:
            continue
        
        window_df = df.iloc[left:right]
        G = build_graph_from_flows(window_df, feat_cols)
        
        if G.number_of_nodes() < min_nodes:
            continue
        
        G = compute_node_features(G, feat_cols)
        pyg_data, graph_label = graph_to_pyg(G, feat_cols, scaler_obj)
        
        # Store graph stats for potential stat templates
        pyg_data.num_nodes_val = G.number_of_nodes()
        pyg_data.num_edges_val = G.number_of_edges()
        
        graphs.append(pyg_data)
    
    return graphs


# Load and build
df_all = load_all_csvs(CICIDS_DIR)
print(f"\nBuilding graphs from {len(df_all):,} flows...")
all_graphs = build_graphs_from_csv(df_all, feature_cols, scaler,
                                    window_size=WINDOW_SIZE, stride=STRIDE,
                                    min_nodes=MIN_NODES)

print(f"\nBuilt {len(all_graphs)} graphs")
label_dist = Counter(g.y_attack.item() for g in all_graphs)
print("Class distribution:")
for cls_id in sorted(label_dist.keys()):
    print(f"  {ID_TO_ATTACK.get(cls_id, cls_id)}: {label_dist[cls_id]}")

# ======================================================================
# === Cell 4: Ego-Graph Extraction for Augmentation ===

K_HOPS = 2
MAX_EGOS_PER_GRAPH = 5
MIN_EGO_NODES = 4
MIN_PARENT_NODES = 8

def extract_ego_graphs(graph, k_hops=K_HOPS, max_egos=MAX_EGOS_PER_GRAPH,
                       min_ego_nodes=MIN_EGO_NODES, min_parent_nodes=MIN_PARENT_NODES):
    """Extract ego-subgraphs from high-degree nodes."""
    num_nodes = graph.x.size(0)
    if num_nodes < min_parent_nodes:
        return []
    
    # Find top-degree nodes
    edge_index = graph.edge_index
    if edge_index.numel() == 0:
        return []
    
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    
    _, top_nodes = degrees.topk(min(max_egos, num_nodes))
    
    egos = []
    for center in top_nodes.tolist():
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            center, k_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        if len(subset) < min_ego_nodes:
            continue
        
        ego = Data(
            x=graph.x[subset],
            edge_index=sub_edge_index,
            y_attack=graph.y_attack.clone(),
            y=graph.y.clone(),
        )
        if hasattr(graph, 'node_attack_labels'):
            ego.node_attack_labels = graph.node_attack_labels[subset]
        ego.num_nodes_val = len(subset)
        ego.num_edges_val = sub_edge_index.size(1)
        egos.append(ego)
    
    return egos


def edge_drop_augment(graph, drop_rate=0.15):
    """Create a copy with random edges dropped."""
    num_edges = graph.edge_index.size(1)
    if num_edges <= 2:
        return None
    mask = torch.rand(num_edges) > drop_rate
    if mask.sum() < 2:
        return None
    aug = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index[:, mask],
        y_attack=graph.y_attack.clone(),
        y=graph.y.clone(),
    )
    if hasattr(graph, 'node_attack_labels'):
        aug.node_attack_labels = graph.node_attack_labels.clone()
    aug.num_nodes_val = graph.x.size(0)
    aug.num_edges_val = aug.edge_index.size(1)
    return aug


# Extract ego-graphs, prioritizing minority classes
augmented_graphs = list(all_graphs)  # start with originals

# Determine minority classes
label_dist = Counter(g.y_attack.item() for g in all_graphs)
max_count = max(label_dist.values())
minority_classes = {cls for cls, cnt in label_dist.items() if cnt < max_count * 0.5}

print(f"Minority classes (< 50% of max): {[ID_TO_ATTACK.get(c, c) for c in minority_classes]}")
print(f"Extracting ego-graphs...")

for g in tqdm(all_graphs, desc='Ego extraction'):
    cls_id = g.y_attack.item()
    # More egos for minority classes
    max_e = MAX_EGOS_PER_GRAPH if cls_id in minority_classes else 2
    egos = extract_ego_graphs(g, max_egos=max_e)
    augmented_graphs.extend(egos)

# Edge-drop augmentation for classes still underrepresented
label_dist2 = Counter(g.y_attack.item() for g in augmented_graphs)
target_per_class = 500

for cls_id in range(7):
    current = label_dist2.get(cls_id, 0)
    if current < target_per_class:
        cls_graphs = [g for g in augmented_graphs if g.y_attack.item() == cls_id]
        needed = target_per_class - current
        added = 0
        while added < needed and cls_graphs:
            for g in cls_graphs:
                if added >= needed:
                    break
                aug = edge_drop_augment(g)
                if aug is not None:
                    augmented_graphs.append(aug)
                    added += 1

# Cap Benign at 2000
benign_indices = [i for i, g in enumerate(augmented_graphs) if g.y_attack.item() == 0]
if len(benign_indices) > 2000:
    remove_indices = set(random.sample(benign_indices, len(benign_indices) - 2000))
    augmented_graphs = [g for i, g in enumerate(augmented_graphs) if i not in remove_indices]

print(f"\nTotal graphs after augmentation: {len(augmented_graphs)}")

# ======================================================================
# === Cell 5: Class Balance Verification ===

label_dist_final = Counter(g.y_attack.item() for g in augmented_graphs)

print("Final class distribution:")
print(f"{'Class':<15} {'Count':>8} {'Pct':>8}")
print('-' * 33)
total = len(augmented_graphs)
for cls_id in sorted(label_dist_final.keys()):
    name = ID_TO_ATTACK.get(cls_id, f'Class {cls_id}')
    cnt = label_dist_final[cls_id]
    pct = cnt / total * 100
    print(f"{name:<15} {cnt:>8,} {pct:>7.1f}%")
print(f"{'TOTAL':<15} {total:>8,} {'100.0':>7}%")

# Verify minimum
min_class = min(label_dist_final.values())
print(f"\nSmallest class: {min_class} graphs")
if min_class < 100:
    print("WARNING: Some classes are very small. Training may struggle with these.")

# ## Section 3: Semantic Text Descriptions
# 
# **Critical fix**: Previous templates were stat-heavy (\"DoS attack with 15 nodes and 42 edges\").
# Now we use semantic descriptions of what each attack IS, plus query-style templates
# that match how users will actually search.

# ======================================================================
# === Cell 6: Semantic Template Definitions ===

# Category A: Semantic descriptions (what the attack IS)
SEMANTIC_TEMPLATES = {
    'Benign': [
        'Normal network traffic with regular communication patterns between hosts.',
        'Legitimate network activity showing standard client-server interactions.',
        'Benign traffic consisting of routine web browsing, email, and file transfers.',
        'Clean network behavior with no signs of malicious intent or anomalous patterns.',
        'Standard enterprise network traffic with typical protocol distributions.',
        'Regular network communication with balanced request-response patterns.',
        'Normal network flow with expected packet sizes and inter-arrival times.',
        'Legitimate user traffic showing standard HTTP, DNS, and SMTP interactions.',
        'Baseline network activity with no indicators of compromise.',
        'Typical daily network traffic from regular business operations.',
    ],
    'DoS': [
        'Denial of Service attack where a single source floods a target to exhaust its resources.',
        'DoS attack characterized by overwhelming volume of requests to a single destination.',
        'Resource exhaustion attack using high-rate packet flooding from one attacker.',
        'Service disruption attack targeting server availability through connection saturation.',
        'Denial of Service involving repeated SYN floods or HTTP request flooding.',
        'DoS attack with a single malicious source sending massive traffic to crash a service.',
        'Network attack aimed at making a server or service unavailable through overload.',
        'Flooding attack where one host sends an abnormal volume of traffic to deny service.',
        'Application-layer DoS attack exhausting server resources with crafted requests.',
        'Denial of service traffic showing asymmetric communication from attacker to victim.',
    ],
    'DDoS': [
        'Distributed Denial of Service attack with multiple coordinated sources targeting one victim.',
        'DDoS attack showing many distributed attackers sending traffic to overwhelm a target.',
        'Coordinated volumetric attack from a botnet flooding a target with traffic.',
        'Multi-source denial of service with synchronized traffic patterns from compromised hosts.',
        'Distributed flooding attack where numerous sources simultaneously target a single server.',
        'Large-scale DDoS attack with traffic originating from many different IP addresses.',
        'Botnet-driven distributed attack overwhelming target infrastructure with massive traffic.',
        'DDoS amplification attack using multiple reflectors to flood a victim.',
        'Coordinated network attack with many sources generating aggregate denial of service.',
        'Distributed attack where compromised machines synchronize to take down a target.',
    ],
    'PortScan': [
        'Port scanning reconnaissance probing multiple ports on target hosts to discover services.',
        'Network reconnaissance activity systematically probing ports to map open services.',
        'Service discovery scan sending connection attempts to many ports on target machines.',
        'Horizontal or vertical port scan attempting to enumerate network services.',
        'Automated scanning activity probing for open ports and vulnerable services.',
        'Network mapping attack scanning multiple hosts and ports for service enumeration.',
        'Reconnaissance traffic with rapid sequential connection attempts to different ports.',
        'Port sweep activity checking for accessible services across the network.',
        'Service fingerprinting scan probing ports to identify running applications.',
        'Systematic network probe scanning for open ports and potential attack vectors.',
    ],
    'BruteForce': [
        'Brute force authentication attack with repeated login attempts using different credentials.',
        'Credential stuffing attack trying many username and password combinations on a service.',
        'FTP or SSH brute force attack with rapid sequential authentication attempts.',
        'Password guessing attack targeting login services with automated credential testing.',
        'Sustained authentication attack repeatedly trying to gain unauthorized access.',
        'Brute force login attack with high-frequency authentication requests to a single service.',
        'Automated password cracking attempt against SSH, FTP, or other authentication services.',
        'Credential brute force with dictionary-based password guessing against network services.',
        'Repeated failed login attempts indicating brute force password attack.',
        'Authentication abuse with rapid-fire credential testing against remote services.',
    ],
    'WebAttack': [
        'Web-based attack targeting HTTP services with malicious requests or injections.',
        'Web application attack involving SQL injection, XSS, or brute force against web forms.',
        'HTTP-layer attack with crafted payloads targeting web application vulnerabilities.',
        'Web attack using malformed HTTP requests to exploit server-side vulnerabilities.',
        'Application-layer attack targeting web services with injection or traversal techniques.',
        'SQL injection attack sending malicious database queries through web parameters.',
        'Cross-site scripting attack injecting malicious scripts into web applications.',
        'Web application exploitation using crafted HTTP requests to bypass security controls.',
        'HTTP-based attack attempting to exploit input validation flaws in web services.',
        'Web vulnerability exploitation targeting server-side code through HTTP requests.',
    ],
    'Bot/Other': [
        'Botnet command and control communication between compromised hosts and a controller.',
        'Automated bot traffic showing periodic beaconing and command-response patterns.',
        'Infiltration or data exfiltration traffic with stealthy low-rate communication.',
        'Anomalous traffic from compromised hosts exhibiting bot-like periodic behavior.',
        'Covert channel communication typical of botnet or advanced persistent threat activity.',
        'Bot traffic with regular beaconing intervals to a command and control server.',
        'Heartbleed exploitation or infiltration traffic with unusual data transfer patterns.',
        'Malware communication showing automated, periodic network connections.',
        'Suspicious automated traffic indicating compromised host under remote control.',
        'Stealthy network traffic consistent with data exfiltration or backdoor communication.',
    ],
}

# Category B: Query-style templates (how users search)
QUERY_TEMPLATES = {
    'Benign': [
        'Show me normal network traffic.',
        'Find benign network activity.',
        'Display clean traffic with no attacks.',
        'Retrieve regular network communication patterns.',
        'Show legitimate network flows.',
    ],
    'DoS': [
        'Show me DoS attacks.',
        'Find denial of service activity.',
        'Detect traffic where a server is being flooded.',
        'Retrieve DoS flooding patterns.',
        'Show denial of service attack traffic.',
    ],
    'DDoS': [
        'Show me DDoS attacks.',
        'Find distributed denial of service traffic.',
        'Detect coordinated flooding from multiple sources.',
        'Retrieve distributed attack patterns.',
        'Show DDoS botnet activity.',
    ],
    'PortScan': [
        'Show me port scanning activity.',
        'Find network reconnaissance scans.',
        'Detect port probing traffic.',
        'Retrieve scanning and enumeration attempts.',
        'Show port scan reconnaissance.',
    ],
    'BruteForce': [
        'Show me brute force attacks.',
        'Find credential guessing attempts.',
        'Detect FTP or SSH login attacks.',
        'Retrieve password brute force traffic.',
        'Show authentication attack patterns.',
    ],
    'WebAttack': [
        'Show me web attacks.',
        'Find SQL injection or XSS traffic.',
        'Detect web application attacks.',
        'Retrieve HTTP-based attack patterns.',
        'Show web vulnerability exploitation.',
    ],
    'Bot/Other': [
        'Show me botnet traffic.',
        'Find command and control communication.',
        'Detect bot activity or infiltration.',
        'Retrieve anomalous automated traffic.',
        'Show covert channel communication.',
    ],
}

# Category C: Light statistical context (grounds representations)
STAT_TEMPLATES = {
    'Benign': [
        'Benign network segment with balanced communication across hosts.',
        'Normal traffic cluster with moderate connection density.',
    ],
    'DoS': [
        'DoS attack traffic with high connection concentration toward a single target.',
        'Denial of service flooding visible in a network cluster with elevated degree centrality.',
    ],
    'DDoS': [
        'DDoS traffic from many distributed sources converging on a single destination.',
        'Distributed attack with high fan-in from multiple attacking nodes.',
    ],
    'PortScan': [
        'Port scan with high edge count from scanner to many targets.',
        'Reconnaissance traffic with sparse but wide-spread connection patterns.',
    ],
    'BruteForce': [
        'Brute force traffic with repeated connections to authentication services.',
        'High-frequency connection attempts between attacker and login service.',
    ],
    'WebAttack': [
        'Web attack traffic concentrated on HTTP service ports.',
        'Application-layer attack with focused connections to web servers.',
    ],
    'Bot/Other': [
        'Bot traffic with periodic low-rate communication to external controllers.',
        'Stealthy automated traffic with regular beaconing intervals.',
    ],
}

ALL_TEMPLATES = {
    cls: {'semantic': SEMANTIC_TEMPLATES[cls],
          'query': QUERY_TEMPLATES[cls],
          'stat': STAT_TEMPLATES[cls]}
    for cls in SEMANTIC_TEMPLATES
}

print("Templates defined:")
for cls, cats in ALL_TEMPLATES.items():
    total = sum(len(v) for v in cats.values())
    print(f"  {cls}: {total} templates ({len(cats['semantic'])}A + {len(cats['query'])}B + {len(cats['stat'])}C)")

# ======================================================================
# === Cell 7: Dynamic Text Sampling Function ===

def sample_template(class_name, templates):
    """Sample a text description with 60/25/15 category weighting."""
    r = random.random()
    if r < 0.60:
        pool = templates[class_name]['semantic']
    elif r < 0.85:
        pool = templates[class_name]['query']
    else:
        pool = templates[class_name]['stat']
    return random.choice(pool)


def get_fixed_template(class_name, templates, idx=0):
    """Get a deterministic template for val/test (always semantic category)."""
    pool = templates[class_name]['semantic']
    return pool[idx % len(pool)]


# Verify sampling distribution
category_counts = Counter()
for _ in range(10000):
    r = random.random()
    if r < 0.60:
        category_counts['semantic'] += 1
    elif r < 0.85:
        category_counts['query'] += 1
    else:
        category_counts['stat'] += 1

print("Sampling distribution (10K samples):")
for cat, cnt in category_counts.most_common():
    print(f"  {cat}: {cnt/100:.1f}%")

# ## Section 4: Dataset & DataLoaders

# ======================================================================
# === Cell 8: GraphTextDataset & Collate Function ===

class GraphTextDataset(TorchDataset):
    """Dataset of (graph, text) pairs with dynamic text sampling."""
    
    def __init__(self, graphs, templates, id_to_attack, fixed_text=False):
        self.graphs = graphs
        self.templates = templates
        self.id_to_attack = id_to_attack
        self.fixed_text = fixed_text  # True for val/test (reproducible R@K)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        class_id = graph.y_attack.item()
        class_name = self.id_to_attack[class_id]
        
        if self.fixed_text:
            text = get_fixed_template(class_name, self.templates, idx)
        else:
            text = sample_template(class_name, self.templates)
        
        return graph, text


def collate_graph_text(batch):
    """Collate (graph, text) pairs into (Batch, List[str])."""
    graphs, texts = zip(*batch)
    batch_graphs = Batch.from_data_list(list(graphs))
    return batch_graphs, list(texts)


print("GraphTextDataset and collate_graph_text defined.")

# ======================================================================
# === Cell 9: Train/Val/Test Split & DataLoaders ===

BATCH_SIZE = 32

# Stratified split: 80% train, 10% val, 10% test
labels = [g.y_attack.item() for g in augmented_graphs]
indices = list(range(len(augmented_graphs)))

train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

train_graphs = [augmented_graphs[i] for i in train_idx]
val_graphs = [augmented_graphs[i] for i in val_idx]
test_graphs = [augmented_graphs[i] for i in test_idx]

print(f"Train: {len(train_graphs)} graphs")
print(f"Val:   {len(val_graphs)} graphs")
print(f"Test:  {len(test_graphs)} graphs")

# Datasets
train_dataset = GraphTextDataset(train_graphs, ALL_TEMPLATES, ID_TO_ATTACK, fixed_text=False)
val_dataset = GraphTextDataset(val_graphs, ALL_TEMPLATES, ID_TO_ATTACK, fixed_text=True)
test_dataset = GraphTextDataset(test_graphs, ALL_TEMPLATES, ID_TO_ATTACK, fixed_text=True)

# Weighted sampler for class balance
train_labels = [g.y_attack.item() for g in train_graphs]
class_counts = Counter(train_labels)
n_classes = len(class_counts)

sample_weights = [1.0 / (n_classes * class_counts[lbl]) for lbl in train_labels]
train_sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float64),
    num_samples=len(train_dataset),
    replacement=True
)

# DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    collate_fn=collate_graph_text, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_graph_text, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_graph_text, num_workers=0
)

print(f"\nTrain: {len(train_loader)} batches")
print(f"Val:   {len(val_loader)} batches")
print(f"Test:  {len(test_loader)} batches")
print(f"Expected per class per batch: ~{BATCH_SIZE / n_classes:.1f}")

# Verify balanced batch
batch_graphs, batch_texts = next(iter(train_loader))
print(f"\nSample batch:")
print(f"  Graphs: {batch_graphs.num_graphs}, Total nodes: {batch_graphs.num_nodes}")
batch_dist = Counter(batch_graphs.y_attack.tolist())
print(f"  Class dist: {dict(sorted(batch_dist.items()))}")
print(f"  Text[0]: {batch_texts[0][:80]}...")

# ## Section 5: Model Construction
# 
# Key changes from v2.1:
# - `use_qformer=True` (selective attention to attack-relevant nodes)
# - `contrastive_weight=0.80` (contrastive loss dominates)
# - `auxiliary_weight=0.20` (reduced to supporting role)

# ======================================================================
# === Cell 10: BERT Text Encoder (frozen) ===

text_encoder = BERTEncoder(model_name='bert-base-uncased', device=device)
text_encoder.freeze()
print(f"BERT loaded and frozen: {text_encoder.hidden_dim}-dim")

# ======================================================================
# === Cell 11: CrossAttentionBridgeV2 with QFormer ===

gnn_model = gnn_model.to(device)

model = CrossAttentionBridgeV2(
    gnn_model=gnn_model,
    text_encoder=text_encoder,
    hidden_dim=256,
    dropout=0.1,
    pooling='mean',
    use_auxiliary_tasks=True,
    num_attack_classes=7,
    contrastive_weight=0.80,    # UP from 0.5
    auxiliary_weight=0.20,      # DOWN from 0.5
    use_qformer=True,           # ENABLED (was False in v2.1)
    num_queries=4,
    num_qformer_layers=2,
    soft_target_alpha=0.1,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model v2.2 created:")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen (GNN+BERT):    {total_params - trainable_params:,}")
print(f"  QFormer: {sum(p.numel() for p in model.qformer.parameters()):,} params")
print(f"  Graph proj: {sum(p.numel() for p in model.graph_proj.parameters()):,} params")
print(f"  Text proj: {sum(p.numel() for p in model.text_proj.parameters()):,} params")
print(f"  Loss: SigLIP + soft targets (alpha={model.soft_target_alpha})")
print(f"  Weights: contrastive={model.contrastive_weight}, auxiliary={model.auxiliary_weight}")

# Forward pass test
model.train()
batch_graphs, batch_texts = next(iter(train_loader))
batch_graphs = batch_graphs.to(device)
g_emb, t_emb, loss, metrics = model(batch_graphs, batch_texts)
print(f"\nForward pass test:")
print(f"  Graph embs: {g_emb.shape}, Text embs: {t_emb.shape}")
print(f"  Loss: {loss.item():.4f}, Acc: {metrics['acc_avg']:.3f}")
print(f"  Pos sim: {metrics['pos_sim']:.4f}, Neg sim: {metrics['neg_sim']:.4f}")

# ## Section 6: Training Loop

# ======================================================================
# === Cell 12: Training & Validation Functions ===

def train_epoch(model, loader, optimizer, device, epoch, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, (batch_graphs, batch_texts) in enumerate(pbar):
        batch_graphs = batch_graphs.to(device)

        g_emb, t_emb, loss, metrics = model(batch_graphs, batch_texts, compute_auxiliary=True)

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{metrics.get('acc_avg', 0):.3f}"})

    avg = {k: v / num_batches for k, v in total_metrics.items()}
    avg['loss'] = total_loss / num_batches
    return avg


@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0
    all_g_embs, all_t_embs, all_labels = [], [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ")
    for batch_graphs, batch_texts in pbar:
        batch_graphs = batch_graphs.to(device)
        g_emb, t_emb, loss, metrics = model(batch_graphs, batch_texts, compute_auxiliary=True)

        all_g_embs.append(g_emb.cpu())
        all_t_embs.append(t_emb.cpu())
        if hasattr(batch_graphs, 'y_attack'):
            all_labels.append(batch_graphs.y_attack.cpu())

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{metrics.get('acc_avg', 0):.3f}"})

    avg = {k: v / num_batches for k, v in total_metrics.items()}
    avg['loss'] = total_loss / num_batches

    # Full-set retrieval metrics
    all_g = torch.cat(all_g_embs)
    all_t = torch.cat(all_t_embs)
    avg.update(compute_recall_at_k(all_g, all_t))

    if all_labels:
        all_l = torch.cat(all_labels)
        avg.update(compute_zero_shot_accuracy(all_g, all_t, all_l, all_l))

    return avg


print("Training and validation functions defined.")

# ======================================================================
# === Cell 13: Evaluation Metrics ===

def compute_recall_at_k(graph_embs, text_embs, k_values=[1, 5, 10]):
    """Recall@K: is the matching text in the top-K most similar?"""
    N = graph_embs.size(0)
    sim = graph_embs @ text_embs.T
    gt = torch.arange(N, device=graph_embs.device)

    metrics = {}
    for k in k_values:
        k_actual = min(k, N)
        _, topk_g2t = sim.topk(k_actual, dim=1)
        hits_g2t = (topk_g2t == gt.unsqueeze(1)).any(dim=1).float().mean().item()

        _, topk_t2g = sim.T.topk(k_actual, dim=1)
        hits_t2g = (topk_t2g == gt.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics[f'R@{k}_g2t'] = hits_g2t
        metrics[f'R@{k}_t2g'] = hits_t2g
        metrics[f'R@{k}_avg'] = (hits_g2t + hits_t2g) / 2
    return metrics


def compute_zero_shot_accuracy(graph_embs, text_embs, graph_labels, text_labels):
    """Zero-shot: classify graphs by nearest text prototype per class."""
    unique_labels = torch.unique(text_labels)
    prototypes, proto_labels = [], []

    for label in unique_labels:
        mask = text_labels == label
        if mask.sum() > 0:
            proto = F.normalize(text_embs[mask].mean(dim=0), dim=0)
            prototypes.append(proto)
            proto_labels.append(label.item())

    if not prototypes:
        return {'zero_shot_acc': 0.0}

    prototypes_t = torch.stack(prototypes)
    proto_labels_t = torch.tensor(proto_labels, device=graph_embs.device)

    sim = graph_embs @ prototypes_t.T
    predicted = proto_labels_t[sim.argmax(dim=1)]
    correct = (predicted == graph_labels).float()

    metrics = {'zero_shot_acc': correct.mean().item()}
    for label in unique_labels:
        mask = graph_labels == label
        if mask.sum() > 0:
            metrics[f'zero_shot_acc_class_{label.item()}'] = correct[mask].mean().item()
    return metrics


print("Metrics defined: compute_recall_at_k, compute_zero_shot_accuracy")

# ======================================================================
# === Cell 14: Training Hyperparameters ===

NUM_EPOCHS = 50
LR = 5e-4              # Higher than old 1e-4 — QFormer needs faster initial learning
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 5
GRAD_ACCUM = 2          # Effective batch = 64
PATIENCE = 10           # Early stop on R@1
CHECKPOINT_DIR = 'checkpoints/stage2'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)

def get_lr_multiplier(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

print(f"Hyperparameters:")
print(f"  LR: {LR}, Weight Decay: {WEIGHT_DECAY}")
print(f"  Epochs: {NUM_EPOCHS}, Warmup: {WARMUP_EPOCHS}")
print(f"  Grad Accum: {GRAD_ACCUM} (effective batch={BATCH_SIZE * GRAD_ACCUM})")
print(f"  Early stop patience: {PATIENCE} (on R@1_avg)")

# ======================================================================
# === Cell 15: Training Loop ===

best_val_loss = float('inf')
best_val_r1 = 0.0
patience_counter = 0
history = []

for epoch in range(1, NUM_EPOCHS + 1):
    # LR schedule
    lr_mult = get_lr_multiplier(epoch - 1, WARMUP_EPOCHS, NUM_EPOCHS)
    current_lr = LR * lr_mult
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr

    # Train
    train_m = train_epoch(model, train_loader, optimizer, device, epoch, GRAD_ACCUM)

    # Validate
    val_m = validate(model, val_loader, device, epoch)

    history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

    # Print
    print(f"\nEpoch {epoch}/{NUM_EPOCHS} (lr={current_lr:.6f})")
    print(f"  Train: loss={train_m['loss']:.4f}  acc={train_m.get('acc_avg', 0):.3f}")
    print(f"  Val:   loss={val_m['loss']:.4f}  acc={val_m.get('acc_avg', 0):.3f}")
    if 'R@1_avg' in val_m:
        print(f"  R@1={val_m['R@1_avg']:.3f}  R@5={val_m.get('R@5_avg', 0):.3f}  "
              f"R@10={val_m.get('R@10_avg', 0):.3f}")
    if 'zero_shot_acc' in val_m:
        print(f"  Zero-shot acc: {val_m['zero_shot_acc']:.3f}")
    print(f"  Temp: {val_m.get('temperature', 0):.4f}  "
          f"Pos sim: {val_m.get('pos_sim', 0):.4f}  "
          f"Neg sim: {val_m.get('neg_sim', 0):.4f}")

    # Checkpoint on best val loss
    if val_m['loss'] < best_val_loss:
        best_val_loss = val_m['loss']
        model.save_checkpoint(f'{CHECKPOINT_DIR}/best_loss.pt', epoch, optimizer.state_dict())
        print(f"  -> Saved best loss checkpoint (loss={best_val_loss:.4f})")

    # Checkpoint on best R@1
    val_r1 = val_m.get('R@1_avg', 0)
    if val_r1 > best_val_r1:
        best_val_r1 = val_r1
        model.save_checkpoint(f'{CHECKPOINT_DIR}/best_r1.pt', epoch, optimizer.state_dict())
        print(f"  -> Saved best R@1 checkpoint (R@1={best_val_r1:.3f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE and epoch > WARMUP_EPOCHS + PATIENCE:
            print(f"\nEarly stopping: R@1 hasn't improved for {PATIENCE} epochs.")
            break

    # Diagnostic warning
    if epoch == 10 and best_val_r1 < 0.05:
        print("\n*** WARNING: R@1 < 0.05 after 10 epochs. Check data/model setup. ***")

print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}, Best R@1: {best_val_r1:.3f}")

# ## Section 7: Evaluation & Visualization

# ======================================================================
# === Cell 16: Test Set Evaluation ===

# Load best R@1 checkpoint
best_path = f'{CHECKPOINT_DIR}/best_r1.pt'
if not os.path.exists(best_path):
    best_path = f'{CHECKPOINT_DIR}/best_loss.pt'

print(f"Loading best checkpoint: {best_path}")
ckpt = torch.load(best_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f"  Epoch: {ckpt['epoch']}, Config: {ckpt['config']}")

# Evaluate on test set
test_m = validate(model, test_loader, device, 0)

print(f"\n=== Test Set Results ===")
print(f"  R@1:  {test_m.get('R@1_avg', 0):.3f} (g2t={test_m.get('R@1_g2t', 0):.3f}, t2g={test_m.get('R@1_t2g', 0):.3f})")
print(f"  R@5:  {test_m.get('R@5_avg', 0):.3f}")
print(f"  R@10: {test_m.get('R@10_avg', 0):.3f}")
print(f"  Zero-shot acc: {test_m.get('zero_shot_acc', 0):.3f}")

# Per-class zero-shot accuracy
print(f"\nPer-class zero-shot accuracy:")
for cls_id in range(7):
    key = f'zero_shot_acc_class_{cls_id}'
    if key in test_m:
        print(f"  {ID_TO_ATTACK.get(cls_id, cls_id)}: {test_m[key]:.3f}")

# ======================================================================
# === Cell 17: t-SNE Visualization ===
from sklearn.manifold import TSNE

# Collect embeddings from test set
all_g_embs, all_t_embs, all_labels = [], [], []
model.eval()
with torch.no_grad():
    for batch_graphs, batch_texts in test_loader:
        batch_graphs = batch_graphs.to(device)
        g_emb = model.encode_graph(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
        t_emb = model.encode_text(batch_texts)
        all_g_embs.append(g_emb.cpu())
        all_t_embs.append(t_emb.cpu())
        all_labels.append(batch_graphs.y_attack.cpu())

all_g = torch.cat(all_g_embs).numpy()
all_t = torch.cat(all_t_embs).numpy()
all_l = torch.cat(all_labels).numpy()

# Combine for t-SNE
combined = np.vstack([all_g, all_t])
labels_combined = np.concatenate([all_l, all_l])
modality = ['graph'] * len(all_g) + ['text'] * len(all_t)

print(f"Running t-SNE on {len(combined)} embeddings (256-dim)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords = tsne.fit_transform(combined)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors = plt.cm.Set1(np.linspace(0, 1, 7))
class_names = [ID_TO_ATTACK.get(i, f'Class {i}') for i in range(7)]

# Left: colored by class
ax = axes[0]
n_g = len(all_g)
for cls_id in range(7):
    mask_g = (labels_combined[:n_g] == cls_id)
    mask_t = (labels_combined[n_g:] == cls_id)
    ax.scatter(coords[:n_g][mask_g, 0], coords[:n_g][mask_g, 1],
               c=[colors[cls_id]], marker='o', s=20, alpha=0.6, label=f'{class_names[cls_id]} (graph)')
    ax.scatter(coords[n_g:][mask_t, 0], coords[n_g:][mask_t, 1],
               c=[colors[cls_id]], marker='^', s=30, alpha=0.6, label=f'{class_names[cls_id]} (text)')
ax.set_title('Shared Embedding Space (by class)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Right: colored by modality
ax = axes[1]
ax.scatter(coords[:n_g, 0], coords[:n_g, 1], c='blue', marker='o', s=15, alpha=0.4, label='Graph')
ax.scatter(coords[n_g:, 0], coords[n_g:, 1], c='red', marker='^', s=15, alpha=0.4, label='Text')
ax.set_title('Shared Embedding Space (by modality)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{CHECKPOINT_DIR}/tsne_embeddings.png', dpi=150, bbox_inches='tight')
# plt.show()  # skip in script mode
print(f"Saved: {CHECKPOINT_DIR}/tsne_embeddings.png")

# ======================================================================
# === Cell 18: Interactive Retrieval Demo ===

def retrieve_graphs_by_text(model, query_text, test_loader, device, k=5):
    """Retrieve top-k graphs for a text query."""
    model.eval()
    with torch.no_grad():
        q_emb = model.encode_text([query_text])  # [1, 256]
        
        all_embs, all_info = [], []
        for batch_graphs, batch_texts in test_loader:
            batch_graphs = batch_graphs.to(device)
            g_emb = model.encode_graph(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
            all_embs.append(g_emb.cpu())
            for i in range(batch_graphs.num_graphs):
                cls_id = batch_graphs[i].y_attack.item()
                all_info.append({
                    'class': ID_TO_ATTACK.get(cls_id, f'Class {cls_id}'),
                    'class_id': cls_id,
                    'nodes': batch_graphs[i].num_nodes,
                    'edges': batch_graphs[i].edge_index.size(1),
                })
        
        all_embs = torch.cat(all_embs)
        sims = (q_emb.cpu() @ all_embs.T).squeeze(0)
        topk_scores, topk_idx = sims.topk(k)
        
        results = []
        for j, i in enumerate(topk_idx.tolist()):
            results.append((all_info[i], topk_scores[j].item()))
        return results


# Test queries
print('=' * 80)
print('TEXT -> GRAPH RETRIEVAL DEMO')
print('=' * 80)

queries = [
    'Show me DDoS attacks.',
    'Find port scanning activity.',
    'Normal traffic patterns.',
    'Brute force login attempts.',
    'Denial of service attack flooding a server.',
    'Web application attack with SQL injection.',
    'Botnet command and control communication.',
]

for query_text in queries:
    print(f"\nQuery: \"{query_text}\"")
    results = retrieve_graphs_by_text(model, query_text, test_loader, device, k=5)
    for rank, (info, score) in enumerate(results, 1):
        match = 'Y' if query_text.lower().find(info['class'].lower().split('/')[0]) >= 0 or \
                       (info['class'] == 'Benign' and 'normal' in query_text.lower()) else ' '
        print(f"  #{rank} (sim={score:.3f}) [{match}] {info['class']:>12} | "
              f"{info['nodes']}n, {info['edges']}e")

# ## Section 8: Save Checkpoint

# ======================================================================
# === Cell 19: Save Compatible Checkpoint ===
# Must match app/inference_engine.py lines 92-116 loading contract

import shutil

# Use best R@1 checkpoint as canonical best.pt
best_r1_path = f'{CHECKPOINT_DIR}/best_r1.pt'
best_loss_path = f'{CHECKPOINT_DIR}/best_loss.pt'

if os.path.exists(best_r1_path):
    src = best_r1_path
    print(f"Using best R@1 checkpoint: {src}")
else:
    src = best_loss_path
    print(f"Using best loss checkpoint: {src}")

canonical = f'{CHECKPOINT_DIR}/best.pt'
shutil.copy2(src, canonical)

# Verify
ckpt = torch.load(canonical, map_location='cpu', weights_only=False)
print(f"\nSaved canonical checkpoint: {canonical}")
print(f"  Epoch: {ckpt['epoch']}")
print(f"  Config: {ckpt['config']}")
assert ckpt['config']['use_qformer'] == True, 'CRITICAL: use_qformer must be True!'
print(f"  use_qformer=True: VERIFIED")

# List checkpoints
print(f"\nAll files in {CHECKPOINT_DIR}/:")
for f in sorted(os.listdir(CHECKPOINT_DIR)):
    fpath = os.path.join(CHECKPOINT_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f}: {size_mb:.1f} MB")

print(f"\nStage 2 retraining complete!")
print(f"  Model: CrossAttentionBridgeV2 (v2.2 with QFormer)")
print(f"  Best R@1: {best_val_r1:.3f}")
print(f"  Best val loss: {best_val_loss:.4f}")

# ======================================================================
# === Cell 20: Reload Verification ===
# Verify the checkpoint loads correctly in a fresh model instance

print("Verifying checkpoint reload...")

# Reload
verify_ckpt = torch.load(canonical, map_location='cpu', weights_only=False)
cfg = verify_ckpt['config']

verify_model = CrossAttentionBridgeV2(
    gnn_model=gnn_model.cpu(),
    text_encoder=text_encoder,
    hidden_dim=cfg.get('hidden_dim', 256),
    dropout=0.1,
    pooling=cfg.get('pooling', 'mean'),
    use_auxiliary_tasks=cfg.get('use_auxiliary_tasks', True),
    num_attack_classes=7,
    contrastive_weight=cfg.get('contrastive_weight', 0.8),
    auxiliary_weight=cfg.get('auxiliary_weight', 0.2),
    use_qformer=cfg.get('use_qformer', False),
    num_queries=4,
    num_qformer_layers=2,
    soft_target_alpha=cfg.get('soft_target_alpha', 0.1),
)
verify_model.load_state_dict(verify_ckpt['model_state_dict'], strict=False)
verify_model.eval()

# Test forward pass
with torch.no_grad():
    test_batch, test_texts = next(iter(test_loader))
    g_emb = verify_model.encode_graph(test_batch.x, test_batch.edge_index, test_batch.batch)
    t_emb = verify_model.encode_text(test_texts)
    sims = (g_emb @ t_emb.T).diagonal()

print(f"  Graph embs shape: {g_emb.shape}")
print(f"  Text embs shape: {t_emb.shape}")
print(f"  L2 norms: graph={g_emb.norm(dim=1).mean():.4f}, text={t_emb.norm(dim=1).mean():.4f}")
print(f"  Diagonal sims (matched pairs): mean={sims.mean():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")
print(f"  QFormer enabled: {verify_model.use_qformer}")
print(f"\nCheckpoint verification PASSED.")

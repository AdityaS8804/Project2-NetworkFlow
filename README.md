# GNN Network Monitor

Real-time network intrusion detection using Graph Neural Networks (GNN) and cross-attention alignment with natural language. Built on research into semantic interaction with network topology.

## What It Does

- **Live monitoring**: Watches for PCAP captures, extracts flow features via CICFlowMeter, builds temporal graphs, and classifies attacks in real-time
- **Demo mode**: Load pre-recorded CIC-IDS2017 CSV datasets for instant visualization
- **NL querying**: Ask natural language questions like "show me DoS attacks" — the cross-attention bridge finds matching graph snapshots by aligning GNN embeddings with text semantics
- **7 attack classes**: Benign, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot/Other

## Architecture

```
.pcapng file in wireshark/
      |
      v
CICFlowMeter (Java, via Docker)
      |
      v
Flow CSV (85 columns)
      |
      v
Graph Builder (30s sliding windows → NetworkX → PyG)
      |
      v
Stage 1 GNN (GATv2Conv → 128-dim graph embedding)
      |
      v
Stage 2 Cross-Attention Bridge (QFormer → 256-dim shared space)
      |
      v
Streamlit Dashboard (embedding scatter, attack timeline, NL query, topology)
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/main.py
```

The app starts in **Demo mode** by default. To use Demo mode, you need the CIC-IDS2017 CSV files (see [Dataset Setup](#dataset-setup) below).

## Demo Mode

Load pre-recorded CIC-IDS2017 network traffic for instant classification and visualization. The model was trained on this data, so classifications are accurate.

1. Download CSV files (see [Dataset Setup](#dataset-setup))
2. Place them in `datasets/csv/`
3. Run `streamlit run app/main.py`
4. Select "Demo" mode → pick a dataset → click "Load"

## Live Mode

Process real-time network captures.

### Prerequisites

1. **Docker** — required for Java CICFlowMeter (see [Why Java CICFlowMeter](#why-java-cicflowmeter))
2. Build the CICFlowMeter Docker image:

```bash
cd tools/cicflowmeter
docker build -t cicflowmeter .
```

> **Note**: On Apple Silicon Macs, the build uses QEMU emulation (`--platform=linux/amd64`) and takes ~5-10 minutes. On x86 Linux, it's much faster.

### Running Live Mode

1. Start the app: `streamlit run app/main.py`
2. Select "Live" mode in the sidebar
3. Drop `.pcapng`, `.pcap`, or CICFlowMeter `.csv` files into the `wireshark/` directory
4. The dashboard updates automatically as files are processed

### Using the Attack Simulator

Generate realistic attack traffic for testing:

```bash
# Run all attack types (requires sudo for packet capture)
sudo python simulate_attacks.py --attack all

# Run a specific attack
sudo python simulate_attacks.py --attack dos-hulk --duration 90

# Run continuously with 60s intervals
sudo python simulate_attacks.py --continuous --interval 60
```

Available attack types: `dos-hulk`, `dos-goldeneye`, `slowloris`, `ddos`, `portscan`, `webattack`, `ftp-brute`, `ssh-brute`, `normal`

## Attack Simulator Design

The simulator (`simulate_attacks.py`) generates traffic patterns that match the statistical characteristics of the CIC-IDS2017 dataset. Each attack type replicates the real tool behavior:

| Attack | Real Tool | Key Flow Characteristics |
|--------|-----------|-------------------------|
| DoS Hulk | Hulk HTTP flooder | ~85s duration, ~6.6s IAT, port 80, large responses |
| DoS GoldenEye | GoldenEye | ~12s duration, ~1.1s IAT, HTTP keep-alive abuse |
| Slowloris | Slowloris | ~97s duration, partial HTTP headers, ~8 byte payloads |
| DDoS | LOIC-style | ~2s bursts, ~489ms IAT, high packet rate |
| PortScan | Nmap SYN scan | ~47us per port, rapid probes across 1-4096+ ports |
| Web Attack | Brute force + SQLi | ~5.5s sessions, login attempts with injection payloads |
| FTP Brute Force | Patator-style | ~4s sessions, USER/PASS credential stuffing, port 21 |
| SSH Brute Force | Patator-style | ~2.5s sessions, SSH handshake simulation, port 22 |

### Customizing the Simulator

To add a new attack type:

1. Add a `simulate_your_attack()` function following the existing pattern:
   - Start a local server (target)
   - Start `tcpdump`/`tshark` capture
   - Generate attack traffic matching your desired flow statistics
   - Stop capture and save `.pcapng`
2. Add the attack name to the `dispatch` dict in `main()`
3. Add a filename-to-label mapping in `app/config.py` under `PCAP_LABEL_MAP`

To tune existing attacks, modify the parameters at the top of each `simulate_*()` function (thread count, timing, duration, ports, etc.).

## Why Java CICFlowMeter

The model was trained on the CIC-IDS2017 dataset, which was generated using the **Java CICFlowMeter**. The Python `cicflowmeter` package produces numerically different flow features for the same traffic due to:

- Different flow timeout defaults
- Different aggregation methods for inter-arrival times
- Different timing precision (microseconds vs milliseconds)
- Different handling of bidirectional flow merging

This **feature domain gap** causes the model to misclassify when fed Python-extracted features. Using the same Java CICFlowMeter ensures the features match the training distribution.

The Java CICFlowMeter's `jnetpcap` native library only supports Linux and Windows (no macOS `.dylib`), so we run it via Docker with `--platform=linux/amd64`. On Apple Silicon, this uses QEMU emulation.

The app falls back to the Python `cicflowmeter` if Docker is unavailable, but classification accuracy will be lower.

## Dataset Setup

The CIC-IDS2017 CSV files are too large (~1.1GB total) for GitHub. Download them from the [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html) and place in `datasets/csv/`:

```
datasets/csv/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-workingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## Model Architecture

### Stage 1: GNN Pre-training

A GATv2Conv (Graph Attention Network v2) with 3 layers is trained on CIC-IDS2017 graph snapshots for multi-task learning:

- Attack classification (7 classes)
- Link prediction
- Community detection
- Structural property prediction

Produces **128-dimensional graph embeddings** that capture network topology patterns.

### Stage 2: Cross-Attention Bridge

A QFormer-based cross-attention module aligns the GNN embeddings with BERT text embeddings in a shared 256-dimensional space. This enables:

- Natural language querying of network topology
- Cosine similarity retrieval between text queries and graph snapshots
- Template-based natural language descriptions of network activity

### Checkpoints

Pre-trained weights are included in `checkpoints/`:

| File | Size | Purpose |
|------|------|---------|
| `checkpoints/stage1/best.pt` | 2.8MB | GNN encoder weights |
| `checkpoints/stage1/attack_classifier.pt` | 46KB | Attack classification head |
| `checkpoints/stage1/scaler.pkl` | 2.4KB | Feature normalization (StandardScaler) |
| `checkpoints/stage2/best.pt` | 8.0MB | Cross-attention bridge weights |

## Training Notebooks

The `notebooks/` directory contains the Jupyter notebooks used to train the models:

- `stage1_training.ipynb` — GNN pre-training on CIC-IDS2017 graph snapshots
- `stage2_cross_attention.ipynb` — Cross-attention bridge training with graph-text pairs

These are included for reproducibility and reference. They require the full training environment (CIC-IDS2017 dataset, GPU) to re-run.

## Project Structure

```
NetworkMonitor/
├── app/                      # Streamlit application
│   ├── main.py               # UI entry point
│   ├── config.py             # Paths, label maps, constants
│   ├── models.py             # PyTorch model definitions (GAT, BERT, QFormer)
│   ├── inference_engine.py   # Model loading and inference
│   ├── pipeline.py           # Data processing orchestration
│   ├── graph_builder.py      # Temporal graph construction (sliding windows)
│   ├── pcap_processor.py     # PCAP → CSV via CICFlowMeter Docker
│   ├── watcher.py            # Directory watcher for live mode
│   ├── state.py              # Thread-safe shared state
│   ├── nl_query.py           # Natural language query engine
│   └── visualizations.py     # Plotly chart builders
├── checkpoints/              # Pre-trained model weights (~11MB)
├── notebooks/                # Training notebooks (reference)
├── tools/cicflowmeter/       # CICFlowMeter Docker build
├── wireshark/                # PCAP drop directory (live mode)
├── datasets/                 # CIC-IDS2017 CSVs (download separately)
├── simulate_attacks.py       # Attack traffic generator
└── requirements.txt          # Python dependencies
```

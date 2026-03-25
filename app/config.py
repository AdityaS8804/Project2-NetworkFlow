import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
WATCH_DIR = os.path.join(BASE_DIR, "wireshark")
CHECKPOINT_STAGE1 = os.path.join(BASE_DIR, "checkpoints/stage1/best.pt")
CHECKPOINT_STAGE2 = os.path.join(BASE_DIR, "checkpoints/stage2/best.pt")
SCALER_PATH = os.path.join(BASE_DIR, "checkpoints/stage1/scaler.pkl")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "checkpoints/stage1/attack_classifier.pt")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "checkpoints/stage1/feature_cols.pkl")

# Label maps
ATTACK_LABEL_MAP = {
    "BENIGN": 0,
    "DoS Hulk": 1, "DoS GoldenEye": 1, "DoS slowloris": 1, "DoS Slowhttptest": 1,
    "DoS": 1,
    "DDoS": 2,
    "PortScan": 3,
    "FTP-Patator": 4, "SSH-Patator": 4,
    "BruteForce": 4,
    "Web Attack \x96 Brute Force": 5, "Web Attack \x96 XSS": 5, "Web Attack \x96 Sql Injection": 5,
    "Web Attack  Brute Force": 5, "Web Attack  XSS": 5, "Web Attack  Sql Injection": 5,
    "Web Attack": 5,
    "Bot": 6, "Infiltration": 6, "Heartbleed": 6,
}

ID_TO_ATTACK = {
    0: "Benign", 1: "DoS", 2: "DDoS", 3: "PortScan",
    4: "BruteForce", 5: "WebAttack", 6: "Bot/Other",
}

NUM_CLASSES = 7

# PCAP filename -> label mapping (demo mode)
# Matches both original captures and simulated ones (simulate_attacks.py)
PCAP_LABEL_MAP = {
    "normal_traffic": "BENIGN",
    "normal_sim": "BENIGN",
    "dos_traffic": "DoS",
    "dos_sim": "DoS",
    "dos_hulk_sim": "DoS Hulk",
    "dos_goldeneye_sim": "DoS GoldenEye",
    "dos_slowloris_sim": "DoS slowloris",
    "ddos_sim": "DDoS",
    "portscan_traffic": "PortScan",
    "portscan_sim": "PortScan",
    "webattack_traffic": "Web Attack",
    "webattack_sim": "Web Attack  Brute Force",
    "ftp_bruteforce_sim": "FTP-Patator",
    "ssh_bruteforce_sim": "SSH-Patator",
}

# Graph construction
META_COLS = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
             'Destination Port', 'Protocol', 'Timestamp']
LABEL_COL = 'Label'
WINDOW_SIZE = 30  # seconds
STRIDE = 10       # seconds
MIN_NODES = 2

# Visualization
MAX_RECORDS = 500
REFRESH_INTERVAL_SEC = 5
POLL_INTERVAL_SEC = 2.0

# Live data emulation
CICIDS_DIR = os.path.join(BASE_DIR, "cicids2017")
EMULATION_STATE_PATH = os.path.join(WATCH_DIR, ".emulation_state.json")
EMULATOR_CHUNK_PREFIX = "emulated_chunk_"
LIVE_NETWORK_REFRESH_SEC = 4

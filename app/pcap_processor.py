import os
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd

from .config import PCAP_LABEL_MAP

# Mapping from Java CICFlowMeter column names to CIC-IDS2017 CSV column names.
# Java CICFlowMeter output has slightly different naming than the published CSVs.
_JAVA_CFM_TO_CIC = {
    'Flow ID': 'Flow ID',
    'Src IP': 'Source IP',
    'Src Port': 'Source Port',
    'Dst IP': 'Destination IP',
    'Dst Port': 'Destination Port',
    'Protocol': 'Protocol',
    'Timestamp': 'Timestamp',
    'Flow Duration': 'Flow Duration',
    'Total Fwd Packet': 'Total Fwd Packets',
    'Total Bwd packets': 'Total Backward Packets',
    'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
    'Total Length of Bwd Packet': 'Total Length of Bwd Packets',
    'Fwd Packet Length Max': 'Fwd Packet Length Max',
    'Fwd Packet Length Min': 'Fwd Packet Length Min',
    'Fwd Packet Length Mean': 'Fwd Packet Length Mean',
    'Fwd Packet Length Std': 'Fwd Packet Length Std',
    'Bwd Packet Length Max': 'Bwd Packet Length Max',
    'Bwd Packet Length Min': 'Bwd Packet Length Min',
    'Bwd Packet Length Mean': 'Bwd Packet Length Mean',
    'Bwd Packet Length Std': 'Bwd Packet Length Std',
    'Flow Bytes/s': 'Flow Bytes/s',
    'Flow Packets/s': 'Flow Packets/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow IAT Max': 'Flow IAT Max',
    'Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Total',
    'Fwd IAT Mean': 'Fwd IAT Mean',
    'Fwd IAT Std': 'Fwd IAT Std',
    'Fwd IAT Max': 'Fwd IAT Max',
    'Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Total',
    'Bwd IAT Mean': 'Bwd IAT Mean',
    'Bwd IAT Std': 'Bwd IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max',
    'Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': 'Bwd PSH Flags',
    'Fwd URG Flags': 'Fwd URG Flags',
    'Bwd URG Flags': 'Bwd URG Flags',
    'Fwd Header Length': 'Fwd Header Length',
    'Bwd Header Length': 'Bwd Header Length',
    'Fwd Packets/s': 'Fwd Packets/s',
    'Bwd Packets/s': 'Bwd Packets/s',
    'Packet Length Min': 'Min Packet Length',
    'Packet Length Max': 'Max Packet Length',
    'Packet Length Mean': 'Packet Length Mean',
    'Packet Length Std': 'Packet Length Std',
    'Packet Length Variance': 'Packet Length Variance',
    'FIN Flag Count': 'FIN Flag Count',
    'SYN Flag Count': 'SYN Flag Count',
    'RST Flag Count': 'RST Flag Count',
    'PSH Flag Count': 'PSH Flag Count',
    'ACK Flag Count': 'ACK Flag Count',
    'URG Flag Count': 'URG Flag Count',
    'CWR Flag Count': 'CWE Flag Count',
    'ECE Flag Count': 'ECE Flag Count',
    'Down/Up Ratio': 'Down/Up Ratio',
    'Average Packet Size': 'Average Packet Size',
    'Fwd Segment Size Avg': 'Avg Fwd Segment Size',
    'Bwd Segment Size Avg': 'Avg Bwd Segment Size',
    # Java CFM doesn't output 'Fwd Header Length.1' — duplicate of Fwd Header Length
    'Fwd Bytes/Bulk Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Packet/Bulk Avg': 'Fwd Avg Packets/Bulk',
    'Fwd Bulk Rate Avg': 'Fwd Avg Bulk Rate',
    'Bwd Bytes/Bulk Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Packet/Bulk Avg': 'Bwd Avg Packets/Bulk',
    'Bwd Bulk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets': 'Subflow Fwd Packets',
    'Subflow Fwd Bytes': 'Subflow Fwd Bytes',
    'Subflow Bwd Packets': 'Subflow Bwd Packets',
    'Subflow Bwd Bytes': 'Subflow Bwd Bytes',
    'FWD Init Win Bytes': 'Init_Win_bytes_forward',
    'Bwd Init Win Bytes': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min': 'min_seg_size_forward',
    'Active Mean': 'Active Mean',
    'Active Std': 'Active Std',
    'Active Max': 'Active Max',
    'Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': 'Idle Std',
    'Idle Max': 'Idle Max',
    'Idle Min': 'Idle Min',
}

DOCKER_IMAGE = "cicflowmeter"


def _docker_available():
    """Check if Docker is available and the cicflowmeter image exists."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _convert_to_ethernet_pcap(pcap_path, output_dir):
    """Convert any PCAP (including loopback) to Ethernet-encapsulated .pcap.

    Java CICFlowMeter's jnetpcap requires Ethernet frames (DLT_EN10MB).
    Loopback captures (DLT_NULL) must be re-wrapped with fake Ethernet headers.
    """
    from scapy.all import rdpcap, wrpcap, Ether, IP

    basename = os.path.splitext(os.path.basename(pcap_path))[0]
    out_path = os.path.join(output_dir, basename + ".pcap")

    packets = rdpcap(pcap_path)
    eth_packets = []
    for pkt in packets:
        if pkt.haslayer(IP):
            eth_pkt = Ether(dst='ff:ff:ff:ff:ff:ff', src='00:00:00:00:00:00') / pkt[IP]
            eth_pkt.time = pkt.time  # Preserve original pcap timestamp
            eth_packets.append(eth_pkt)
        elif pkt.haslayer(Ether):
            eth_packets.append(pkt)

    if eth_packets:
        wrpcap(out_path, eth_packets)
    return out_path


def _run_java_cicflowmeter(pcap_path):
    """Run Java CICFlowMeter via Docker, return DataFrame or None on failure."""
    out_dir = tempfile.mkdtemp(prefix="cfm_")
    tmp_pcap = None
    try:
        pcap_abs = os.path.abspath(pcap_path)

        # Always convert to Ethernet .pcap — handles both .pcapng format
        # and loopback encapsulation (DLT_NULL) that jnetpcap can't parse
        tmp_dir = tempfile.mkdtemp(prefix="cfm_conv_")
        tmp_pcap = _convert_to_ethernet_pcap(pcap_abs, tmp_dir)
        pcap_dir = tmp_dir
        pcap_name = os.path.basename(tmp_pcap)

        result = subprocess.run(
            [
                "docker", "run", "--rm", "--platform", "linux/amd64",
                "-v", f"{pcap_dir}:/data/in:ro",
                "-v", f"{out_dir}:/data/out",
                DOCKER_IMAGE,
                f"/data/in/{pcap_name}", "/data/out/"
            ],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            print(f"Java CICFlowMeter stderr: {result.stderr[:500]}")
            return None

        # Find the output CSV
        csvs = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
        if not csvs:
            print("Java CICFlowMeter produced no CSV output")
            return None

        csv_path = os.path.join(out_dir, csvs[0])
        df = pd.read_csv(csv_path)

        if df.empty:
            return None

        # Rename columns to match CIC-IDS2017 naming
        df.columns = df.columns.str.strip()
        rename_map = {k: v for k, v in _JAVA_CFM_TO_CIC.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Add 'Fwd Header Length.1' (duplicate of Fwd Header Length, as in CIC-IDS2017)
        if 'Fwd Header Length' in df.columns:
            df['Fwd Header Length.1'] = df['Fwd Header Length']

        return df

    except subprocess.TimeoutExpired:
        print("Java CICFlowMeter timed out")
        return None
    except Exception as e:
        print(f"Java CICFlowMeter error: {e}")
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        if tmp_pcap:
            shutil.rmtree(os.path.dirname(tmp_pcap), ignore_errors=True)


def _run_python_cicflowmeter(pcap_path):
    """Fallback: use Python cicflowmeter package."""
    from scapy.all import rdpcap
    from cicflowmeter.flow_session import FlowSession

    # Mapping from Python cicflowmeter snake_case to CIC-IDS2017 names
    _PY_CFM_TO_CIC = {
        'flow_duration': 'Flow Duration',
        'tot_fwd_pkts': 'Total Fwd Packets',
        'tot_bwd_pkts': 'Total Backward Packets',
        'totlen_fwd_pkts': 'Total Length of Fwd Packets',
        'totlen_bwd_pkts': 'Total Length of Bwd Packets',
        'fwd_pkt_len_max': 'Fwd Packet Length Max',
        'fwd_pkt_len_min': 'Fwd Packet Length Min',
        'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
        'fwd_pkt_len_std': 'Fwd Packet Length Std',
        'bwd_pkt_len_max': 'Bwd Packet Length Max',
        'bwd_pkt_len_min': 'Bwd Packet Length Min',
        'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
        'bwd_pkt_len_std': 'Bwd Packet Length Std',
        'flow_byts_s': 'Flow Bytes/s',
        'flow_pkts_s': 'Flow Packets/s',
        'flow_iat_mean': 'Flow IAT Mean',
        'flow_iat_std': 'Flow IAT Std',
        'flow_iat_max': 'Flow IAT Max',
        'flow_iat_min': 'Flow IAT Min',
        'fwd_iat_tot': 'Fwd IAT Total',
        'fwd_iat_mean': 'Fwd IAT Mean',
        'fwd_iat_std': 'Fwd IAT Std',
        'fwd_iat_max': 'Fwd IAT Max',
        'fwd_iat_min': 'Fwd IAT Min',
        'bwd_iat_tot': 'Bwd IAT Total',
        'bwd_iat_mean': 'Bwd IAT Mean',
        'bwd_iat_std': 'Bwd IAT Std',
        'bwd_iat_max': 'Bwd IAT Max',
        'bwd_iat_min': 'Bwd IAT Min',
        'fwd_psh_flags': 'Fwd PSH Flags',
        'bwd_psh_flags': 'Bwd PSH Flags',
        'fwd_urg_flags': 'Fwd URG Flags',
        'bwd_urg_flags': 'Bwd URG Flags',
        'fwd_header_len': 'Fwd Header Length',
        'bwd_header_len': 'Bwd Header Length',
        'fwd_pkts_s': 'Fwd Packets/s',
        'bwd_pkts_s': 'Bwd Packets/s',
        'pkt_len_min': 'Min Packet Length',
        'pkt_len_max': 'Max Packet Length',
        'pkt_len_mean': 'Packet Length Mean',
        'pkt_len_std': 'Packet Length Std',
        'pkt_len_var': 'Packet Length Variance',
        'fin_flag_cnt': 'FIN Flag Count',
        'syn_flag_cnt': 'SYN Flag Count',
        'rst_flag_cnt': 'RST Flag Count',
        'psh_flag_cnt': 'PSH Flag Count',
        'ack_flag_cnt': 'ACK Flag Count',
        'urg_flag_cnt': 'URG Flag Count',
        'cwr_flag_count': 'CWE Flag Count',
        'ece_flag_cnt': 'ECE Flag Count',
        'down_up_ratio': 'Down/Up Ratio',
        'pkt_size_avg': 'Average Packet Size',
        'fwd_seg_size_avg': 'Avg Fwd Segment Size',
        'bwd_seg_size_avg': 'Avg Bwd Segment Size',
        'fwd_header_len_1': 'Fwd Header Length.1',
        'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
        'fwd_pkts_b_avg': 'Fwd Avg Packets/Bulk',
        'fwd_blk_rate_avg': 'Fwd Avg Bulk Rate',
        'bwd_byts_b_avg': 'Bwd Avg Bytes/Bulk',
        'bwd_pkts_b_avg': 'Bwd Avg Packets/Bulk',
        'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
        'subflow_fwd_pkts': 'Subflow Fwd Packets',
        'subflow_fwd_byts': 'Subflow Fwd Bytes',
        'subflow_bwd_pkts': 'Subflow Bwd Packets',
        'subflow_bwd_byts': 'Subflow Bwd Bytes',
        'init_fwd_win_byts': 'Init_Win_bytes_forward',
        'init_bwd_win_byts': 'Init_Win_bytes_backward',
        'fwd_act_data_pkts': 'act_data_pkt_fwd',
        'fwd_seg_size_min': 'min_seg_size_forward',
        'active_mean': 'Active Mean',
        'active_std': 'Active Std',
        'active_max': 'Active Max',
        'active_min': 'Active Min',
        'idle_mean': 'Idle Mean',
        'idle_std': 'Idle Std',
        'idle_max': 'Idle Max',
        'idle_min': 'Idle Min',
    }

    _PY_META = {
        'src_ip': 'Source IP',
        'dst_ip': 'Destination IP',
        'src_port': 'Source Port',
        'dst_port': 'Destination Port',
        'protocol': 'Protocol',
        'timestamp': 'Timestamp',
    }

    # Time features: cicflowmeter Python outputs seconds, CIC-IDS2017 uses microseconds
    _TIME_FEATURES = {
        'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    }

    tmpfile = tempfile.mktemp(suffix='.csv')
    try:
        session = FlowSession(output_mode='csv', output=tmpfile)
        packets = rdpcap(pcap_path)
        for pkt in packets:
            session.process(pkt)
        session.flush_flows()

        if not os.path.exists(tmpfile):
            return pd.DataFrame()
        cfm_df = pd.read_csv(tmpfile)
    finally:
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)

    if cfm_df.empty:
        return pd.DataFrame()

    rows = {}
    rows['Flow ID'] = (
        cfm_df['src_ip'].astype(str) + '-' +
        cfm_df['dst_ip'].astype(str) + '-' +
        cfm_df['src_port'].astype(str) + '-' +
        cfm_df['dst_port'].astype(str) + '-' +
        cfm_df['protocol'].astype(str)
    )
    for cfm_name, cic_name in _PY_META.items():
        if cfm_name in cfm_df.columns:
            rows[cic_name] = cfm_df[cfm_name]
    for cfm_name, cic_name in _PY_CFM_TO_CIC.items():
        if cfm_name in cfm_df.columns:
            rows[cic_name] = cfm_df[cfm_name]
        else:
            rows[cic_name] = 0.0

    if 'Fwd Header Length.1' not in rows or (isinstance(rows.get('Fwd Header Length.1'), (int, float)) and rows['Fwd Header Length.1'] == 0):
        rows['Fwd Header Length.1'] = rows.get('Fwd Header Length', 0)

    rows['Label'] = 'Unknown'
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Convert time features from seconds to microseconds
    for col in _TIME_FEATURES:
        if col in df.columns:
            df[col] = df[col] * 1e6

    return df


def infer_label_from_filename(filename):
    """Infer attack label from PCAP filename."""
    basename = os.path.splitext(os.path.splitext(filename)[0])[0]
    for key, label in PCAP_LABEL_MAP.items():
        if key in basename.lower():
            return label
    return "Unknown"


def _is_loopback_capture(pcap_path):
    """Check if PCAP was captured on a loopback interface (DLT_NULL).

    Java CICFlowMeter can't parse loopback captures correctly.
    """
    try:
        from scapy.all import rdpcap
        packets = rdpcap(pcap_path, count=1)
        if packets:
            # Loopback captures use Loopback layer instead of Ether
            from scapy.layers.l2 import Loopback
            return packets[0].haslayer(Loopback) or not packets[0].haslayer('Ether')
    except Exception:
        pass
    return False


def process_pcap(pcap_path, label=None):
    """Process a .pcapng/.pcap file into a DataFrame.

    Uses Java CICFlowMeter (via Docker) for feature extraction matching
    the CIC-IDS2017 training data. Falls back to Python cicflowmeter if
    Docker is unavailable or for loopback captures.

    Args:
        pcap_path: Path to .pcapng or .pcap file
        label: Override label. If None, infers from filename.
    Returns:
        DataFrame with META_COLS + 77 feature columns + 'Label'
    """
    if label is None:
        label = infer_label_from_filename(os.path.basename(pcap_path))

    basename = os.path.basename(pcap_path)

    # Try Java CICFlowMeter first (exact match with training features).
    # Loopback captures are auto-converted to Ethernet encapsulation.
    if _docker_available():
        print(f"Using Java CICFlowMeter (Docker) for {basename}")
        df = _run_java_cicflowmeter(pcap_path)
        if df is not None and not df.empty:
            df['Label'] = label
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            return df
        print("Java CICFlowMeter produced no usable flows, falling back to Python")

    # Fallback to Python cicflowmeter (features differ from training data)
    print(f"Using Python cicflowmeter for {basename}")
    df = _run_python_cicflowmeter(pcap_path)
    if not df.empty:
        df['Label'] = label
    return df

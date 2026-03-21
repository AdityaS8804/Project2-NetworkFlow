#!/usr/bin/env python3
"""
Attack Simulation — Scapy Offline Packet Construction (CIC-IDS2017 Realistic)

Generates .pcap files with crafted packets matching CIC-IDS2017 feature
distributions. No network access, sudo, or running services required.

Each attack type produces packets with IPs, ports, payload sizes, timing,
and TCP flags that match the training data so the GNN classifier recognizes them.

Usage:
    python simulate_attacks.py --attack all
    python simulate_attacks.py --attack dos-hulk
    python simulate_attacks.py --attack portscan --duration 30
    python simulate_attacks.py --continuous --interval 60
"""

import os
import time
import random
import argparse
from datetime import datetime

from scapy.all import Ether, IP, TCP, Raw, wrpcap

# ---------------------------------------------------------------------------
# Constants — IPs and ports matching CIC-IDS2017 dataset
# ---------------------------------------------------------------------------

WIRESHARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wireshark")

VICTIM_IP = "192.168.10.50"
ATTACKER_IPS = ["172.16.0.1", "172.16.0.2", "172.16.0.3"]

# Benign IPs — CIC-IDS2017 has ~170 unique IPs per 30s window.
# We generate traffic from a pool of diverse IPs to match graph density.
BENIGN_INTERNAL_IPS = [f"192.168.10.{i}" for i in range(3, 51) if i != 50]
BENIGN_EXTERNAL_IPS = ([f"52.84.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(30)] +
                       [f"54.192.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(20)] +
                       [f"172.217.{random.randint(0,15)}.{random.randint(1,254)}" for _ in range(15)] +
                       [f"13.{random.randint(52,59)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(15)] +
                       [f"104.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(10)] +
                       [f"10.0.{random.randint(0,10)}.{random.randint(1,254)}" for _ in range(10)])
BENIGN_IPS = BENIGN_INTERNAL_IPS + BENIGN_EXTERNAL_IPS  # ~150 unique IPs

SRC_MAC = "00:0c:29:aa:bb:01"
DST_MAC = "00:0c:29:cc:dd:02"


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def jitter(mean, std, min_val=0.0):
    """Gaussian jitter clamped to min_val."""
    return max(min_val, random.gauss(mean, std))


def rand_high_port():
    return random.randint(1024, 65535)


# ---------------------------------------------------------------------------
# Core: Build a complete bidirectional TCP session as a packet list
# ---------------------------------------------------------------------------

def build_tcp_session(
    src_ip, dst_ip, sport, dport,
    fwd_payloads, bwd_payloads,
    base_time,
    fwd_iats, bwd_delays,
    init_win_fwd=8192, init_win_bwd=65535,
    include_fin=True,
    include_handshake=True,
    fwd_data_flag='PA', bwd_data_flag='PA',
):
    """Build a complete TCP session with proper seq/ack tracking.

    Args:
        src_ip, dst_ip: IP addresses (client, server)
        sport, dport: TCP ports
        fwd_payloads: list of bytes — client-to-server data packets
        bwd_payloads: list of bytes — server-to-client data packets
        base_time: float Unix epoch for first packet (SYN)
        fwd_iats: list of float — seconds between consecutive fwd data packets
        bwd_delays: list of float — delay after corresponding fwd packet for server response
        init_win_fwd: TCP window on SYN (becomes Init_Win_bytes_forward)
        init_win_bwd: TCP window on SYN-ACK (becomes Init_Win_bytes_backward)
        include_fin: whether to add FIN/ACK teardown
        include_handshake: whether to include SYN/SYN-ACK/ACK handshake
            CIC-IDS2017 shows SYN Flag Count=0 for most attack flows because
            CICFlowMeter often misses the handshake or counts flags as boolean.
            Skipping handshake → SYN=0 matching the training data.
        fwd_data_flag: TCP flag string for forward data packets (default 'PA').
            Use 'A' for ACK-only to reduce PSH Flag Count to 0.
        bwd_data_flag: TCP flag string for backward data packets (default 'PA').

    Returns:
        list of Scapy packets sorted by timestamp
    """
    packets = []
    t = base_time

    client_isn = random.randint(100000, 4000000000)
    server_isn = random.randint(100000, 4000000000)
    c_seq = client_isn
    s_seq = server_isn

    def _pkt(src, dst, sp, dp, flags, seq, ack, win, payload=None, ts=None):
        p = (Ether(src=SRC_MAC, dst=DST_MAC) /
             IP(src=src, dst=dst) /
             TCP(sport=sp, dport=dp, flags=flags, seq=seq, ack=ack, window=win))
        if payload:
            p = p / Raw(load=payload)
        p.time = ts
        return p

    # --- 3-way handshake (optional) ---
    if include_handshake:
        # SYN
        packets.append(_pkt(src_ip, dst_ip, sport, dport, 'S', c_seq, 0,
                            init_win_fwd, ts=t))
        c_seq += 1
        t += 0.0005

        # SYN-ACK
        packets.append(_pkt(dst_ip, src_ip, dport, sport, 'SA', s_seq, c_seq,
                            init_win_bwd, ts=t))
        s_seq += 1
        t += 0.0005

        # ACK
        packets.append(_pkt(src_ip, dst_ip, sport, dport, 'A', c_seq, s_seq,
                            init_win_fwd, ts=t))
    else:
        # Even without handshake, advance seq as if it happened
        c_seq += 1
        s_seq += 1

    # --- Data exchange ---
    # Interleave fwd and bwd packets
    num_exchanges = max(len(fwd_payloads), len(bwd_payloads))
    for i in range(num_exchanges):
        # Forward data packet
        if i < len(fwd_payloads):
            if i < len(fwd_iats):
                t += fwd_iats[i]
            else:
                t += fwd_iats[-1] if fwd_iats else 0.5
            payload = fwd_payloads[i]
            packets.append(_pkt(src_ip, dst_ip, sport, dport, fwd_data_flag,
                                c_seq, s_seq, init_win_fwd, payload=payload, ts=t))
            c_seq += len(payload)

        # Backward data packet (server response)
        if i < len(bwd_payloads):
            delay = bwd_delays[i] if i < len(bwd_delays) else 0.05
            t += delay
            payload = bwd_payloads[i]
            packets.append(_pkt(dst_ip, src_ip, dport, sport, bwd_data_flag,
                                s_seq, c_seq, init_win_bwd, payload=payload, ts=t))
            s_seq += len(payload)

            # Client ACK for server data
            t += 0.0002
            packets.append(_pkt(src_ip, dst_ip, sport, dport, 'A', c_seq, s_seq,
                                init_win_fwd, ts=t))

    # --- FIN/ACK teardown ---
    if include_fin:
        t += 0.01
        packets.append(_pkt(src_ip, dst_ip, sport, dport, 'FA', c_seq, s_seq,
                            init_win_fwd, ts=t))
        c_seq += 1
        t += 0.0005
        packets.append(_pkt(dst_ip, src_ip, dport, sport, 'FA', s_seq, c_seq,
                            init_win_bwd, ts=t))
        s_seq += 1
        t += 0.0005
        packets.append(_pkt(src_ip, dst_ip, sport, dport, 'A', c_seq, s_seq,
                            init_win_fwd, ts=t))

    return packets


def write_pcap(packets, output_name):
    """Sort packets by time and write to pcap file."""
    packets.sort(key=lambda p: p.time)
    path = os.path.join(WIRESHARK_DIR, f"{output_name}_{get_timestamp()}.pcap")
    wrpcap(path, packets)
    return path


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_http_request(path="/", host="192.168.10.50", extra_headers=""):
    """Build a simple HTTP GET request."""
    req = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
    if extra_headers:
        req += extra_headers
    req += "\r\n"
    return req.encode()


def make_http_response(body_size=1200, status="200 OK"):
    """Build an HTTP response with a body of specified size."""
    body = b"X" * body_size
    resp = (f"HTTP/1.1 {status}\r\n"
            f"Content-Length: {body_size}\r\n"
            f"Connection: keep-alive\r\n"
            f"\r\n").encode() + body
    return resp


# ---------------------------------------------------------------------------
# Benign background traffic — creates realistic graph density (~150+ IPs)
#
# CIC-IDS2017 has median 166 unique IPs per 30s window, and attack traffic
# ALWAYS coexists with benign traffic (benign:attack ratio median ~10:1).
# Without this background, graphs have only 3-4 nodes and the GNN produces
# out-of-distribution embeddings.
# ---------------------------------------------------------------------------

def generate_benign_background(base_time, duration, num_sessions=None):
    """Generate diverse benign HTTP/HTTPS sessions spread across many IPs.

    Returns a list of packets to mix into attack pcaps.
    Targets ~100-150 unique IPs to match CIC-IDS2017 graph density.
    """
    if num_sessions is None:
        # ~10:1 benign:attack ratio with enough IPs for graph density
        num_sessions = random.randint(120, 180)

    packets = []
    # Sample from the full IP pool — each session uses a different client IP
    ips_to_use = random.sample(BENIGN_IPS, min(num_sessions, len(BENIGN_IPS)))

    # Common server IPs that benign traffic targets (besides VICTIM_IP)
    servers = [VICTIM_IP, "192.168.10.51", "192.168.10.52"]
    pages = ["/", "/index.html", "/about", "/api/status", "/images/logo.png",
             "/css/style.css", "/js/app.js", "/contact", "/search?q=test"]

    for i in range(num_sessions):
        src_ip = ips_to_use[i % len(ips_to_use)]
        dst_ip = random.choice(servers)
        sport = rand_high_port()
        dport = random.choice([80, 80, 80, 443, 443, 8080])  # mostly port 80

        # Stagger sessions across the full duration
        session_start = base_time + random.uniform(0, duration)

        # Normal browsing: 1-3 requests per session
        n_fwd = random.randint(1, 3)
        n_bwd = random.randint(1, 3)

        fwd_payloads = [make_http_request(path=random.choice(pages), host=dst_ip)
                        for _ in range(n_fwd)]
        bwd_payloads = [make_http_response(
                            body_size=random.randint(200, 3000),
                            status=random.choice(["200 OK", "200 OK", "200 OK", "304 Not Modified"]))
                        for _ in range(n_bwd)]

        # Natural timing: 0.5-3s between requests
        fwd_iats = [jitter(1.5, 1.0, 0.1) for _ in range(n_fwd)]
        bwd_delays = [jitter(0.05, 0.03, 0.01) for _ in range(n_bwd)]

        # CIC-IDS2017 benign Init_Win distribution:
        # fwd: -1 (46%), 8192 (12%), 29200 (11%), 65535 (8%), others
        # bwd: -1 (46%), 65535 (11%), 29200 (8%), others
        # Note: -1 in CICFlowMeter means no SYN seen; we use 0 as TCP window equivalent
        benign_win_fwd = random.choices(
            [0, 8192, 29200, 65535, 64240],
            weights=[46, 12, 11, 8, 5], k=1)[0]
        benign_win_bwd = random.choices(
            [0, 65535, 29200, 64240, 8192],
            weights=[46, 11, 8, 5, 4], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, dst_ip, sport, dport,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=benign_win_fwd, init_win_bwd=benign_win_bwd,
        ))

    return packets


# ---------------------------------------------------------------------------
# DoS Hulk — Duration ~85s, IAT ~4.8s, Fwd 44B, Bwd 1281B, ACK-dominant
# ---------------------------------------------------------------------------

def simulate_dos_hulk(duration=90):
    print(f"\n[DoS Hulk] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(20, 30)

    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 2]  # 2 attacker IPs
        sport = rand_high_port()
        session_start = base + jitter(i * (duration / num_sessions), 1.0, 0)

        # ~5 fwd packets, ~4 bwd packets per session
        n_fwd = random.randint(4, 6)
        n_bwd = random.randint(3, 5)

        fwd_payloads = []
        for _ in range(n_fwd):
            rand_param = random.randint(1000, 9999)
            req = make_http_request(
                path=f"/?q={rand_param}",
                extra_headers="Connection: keep-alive\r\n"
            )
            # Pad/trim to ~44 bytes payload
            fwd_payloads.append(req[:max(40, min(50, len(req)))])

        bwd_payloads = [make_http_response(body_size=random.randint(1200, 1350))
                        for _ in range(n_bwd)]

        # IAT: mean 4.8s, std 3s (high variability like dataset)
        fwd_iats = [jitter(4.8, 3.0, 0.5) for _ in range(n_fwd)]
        bwd_delays = [jitter(0.05, 0.02, 0.01) for _ in range(n_bwd)]

        # CIC-IDS2017 DoS Hulk Init_Win distribution:
        # fwd: 251 (40%), 274 (28%), 0/-1 (26%), 29200 (6%)
        # bwd: 235 (71%), -1/0 (29%)
        win_fwd = random.choices([251, 274, 0, 29200], weights=[40, 28, 26, 6], k=1)[0]
        win_bwd = random.choices([235, 0], weights=[71, 29], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 80,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=win_fwd, init_win_bwd=win_bwd,
            include_handshake=False,  # SYN Flag Count=0 in CIC-IDS2017
            include_fin=False,        # FIN Flag Count=0
            fwd_data_flag='A',        # ACK-dominant, PSH=0 (93.7% of flows)
            bwd_data_flag='A',
        ))

    path = write_pcap(packets, "dos_hulk_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# DoS GoldenEye — Duration ~12s, IAT ~2.3s, Fwd 59B, Bwd 1257B, PSH-dominant
# ---------------------------------------------------------------------------

def simulate_dos_goldeneye(duration=60):
    print(f"\n[DoS GoldenEye] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(25, 35)

    agents = ["Mozilla/5.0", "Chrome/91.0", "Safari/537.36", "curl/7.68"]

    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 2]
        sport = rand_high_port()
        session_start = base + jitter(i * (duration / num_sessions), 0.5, 0)

        n_fwd = random.randint(5, 7)
        n_bwd = random.randint(3, 5)

        fwd_payloads = []
        for _ in range(n_fwd):
            req = make_http_request(
                path=f"/?{random.randint(1, 9999)}",
                extra_headers=(f"User-Agent: {random.choice(agents)}\r\n"
                               f"Connection: keep-alive\r\n")
            )
            # Trim to ~59 bytes
            fwd_payloads.append(req[:max(55, min(65, len(req)))])

        bwd_payloads = [make_http_response(body_size=random.randint(1180, 1330))
                        for _ in range(n_bwd)]

        fwd_iats = [jitter(2.3, 1.0, 0.3) for _ in range(n_fwd)]
        bwd_delays = [jitter(0.05, 0.02, 0.01) for _ in range(n_bwd)]

        # CIC-IDS2017 GoldenEye Init_Win: similar to Hulk
        # fwd: 251 (35%), 274 (30%), 0 (25%), 29200 (10%)
        # bwd: 235 (65%), 0 (35%)
        win_fwd = random.choices([251, 274, 0, 29200], weights=[35, 30, 25, 10], k=1)[0]
        win_bwd = random.choices([235, 0], weights=[65, 35], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 80,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=win_fwd, init_win_bwd=win_bwd,
            include_handshake=False,
            include_fin=False,
            fwd_data_flag='PA',  # PSH-dominant (72%) for GoldenEye
            bwd_data_flag='A',
        ))

    path = write_pcap(packets, "dos_goldeneye_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# DoS Slowloris — Duration ~97s, IAT extreme variability, tiny fwd, minimal bwd
# ---------------------------------------------------------------------------

def simulate_slowloris(duration=120):
    print(f"\n[Slowloris] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(15, 20)

    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 2]
        sport = rand_high_port()
        session_start = base + jitter(i * (min(duration, 97) / num_sessions), 2.0, 0)

        # 6 fwd packets (tiny partial headers), 2 bwd packets (minimal)
        n_fwd = random.randint(5, 7)
        n_bwd = random.randint(1, 2)

        # First fwd packet: partial HTTP request line
        fwd_payloads = [b"GET / HTTP/1.1\r\n"]
        # Remaining: tiny header fragments ~8-10 bytes
        for _ in range(n_fwd - 1):
            header = f"X-a: {random.randint(1, 99)}\r\n".encode()
            fwd_payloads.append(header)

        # Minimal backward: just a small response or ACK data
        bwd_payloads = [b"\r\n" * random.randint(1, 4) for _ in range(n_bwd)]

        # IAT: mean 16s, std 22s — EXTREME variability (dataset signature)
        fwd_iats = [jitter(16.0, 22.0, 1.0) for _ in range(n_fwd)]
        # Cap total duration near 97s
        total_iat = sum(fwd_iats)
        if total_iat > 100:
            scale = 97.0 / total_iat
            fwd_iats = [iat * scale for iat in fwd_iats]

        bwd_delays = [jitter(0.5, 0.3, 0.1) for _ in range(n_bwd)]

        # CIC-IDS2017 Slowloris Init_Win:
        # fwd: 29200 (60%), 251 (20%), 0 (20%)
        # bwd: 235 (50%), 0 (50%)
        win_fwd = random.choices([29200, 251, 0], weights=[60, 20, 20], k=1)[0]
        win_bwd = random.choices([235, 0], weights=[50, 50], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 80,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=win_fwd, init_win_bwd=win_bwd,
            include_handshake=False,
            include_fin=False,  # Slowloris doesn't cleanly close
            fwd_data_flag='PA',  # PSH 67% for Slowloris partial headers
            bwd_data_flag='A',
        ))

    path = write_pcap(packets, "dos_slowloris_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# DDoS — Duration ~2s, IAT ~0.5s, Fwd 7B, Bwd 1481B, multiple sources
# ---------------------------------------------------------------------------

def simulate_ddos(duration=60):
    print(f"\n[DDoS] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(50, 80)

    # DDoS uses all 3 attacker IPs (dataset shows bidirectional 2-IP pattern)
    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 3]
        sport = rand_high_port()
        # Rapid succession — many short sessions packed together
        session_start = base + jitter(i * (duration / num_sessions), 0.3, 0)

        n_fwd = random.randint(4, 5)
        n_bwd = random.randint(2, 4)

        # Tiny fwd payloads (~7 bytes)
        fwd_payloads = [b"GET / HTTP/1.1\r\n\r\n"[:random.randint(5, 10)]
                        for _ in range(n_fwd)]

        # Large bwd payloads (~1481 bytes)
        bwd_payloads = [make_http_response(body_size=random.randint(1400, 1560))
                        for _ in range(n_bwd)]

        # Short IAT ~0.5s
        fwd_iats = [jitter(0.5, 0.2, 0.1) for _ in range(n_fwd)]
        bwd_delays = [jitter(0.03, 0.01, 0.01) for _ in range(n_bwd)]

        # CIC-IDS2017 DDoS Init_Win:
        # fwd: 227 (40%), 8192 (30%), 0 (20%), 29200 (10%)
        # bwd: 235 (60%), 0 (40%)
        win_fwd = random.choices([227, 8192, 0, 29200], weights=[40, 30, 20, 10], k=1)[0]
        win_bwd = random.choices([235, 0], weights=[60, 40], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 80,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=win_fwd, init_win_bwd=win_bwd,
            include_handshake=False,
            include_fin=False,
            fwd_data_flag='A',
            bwd_data_flag='A',
        ))

    path = write_pcap(packets, "ddos_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# PortScan — Duration ~47µs per probe, SYN→RST, many ports
# ---------------------------------------------------------------------------

def simulate_portscan(duration=30):
    print(f"\n[PortScan] Generating pcap...")
    base = time.time()
    packets = generate_benign_background(base, duration)

    # Use 2 scanner IPs for graph diversity
    scanner_ips = ATTACKER_IPS[:2]
    num_ports = random.randint(500, 2000)
    ports = list(range(1, num_ports + 1))

    for i, dport in enumerate(ports):
        src_ip = scanner_ips[i % 2]
        sport = rand_high_port()
        t = base + i * 0.000047  # ~47µs between probes

        client_isn = random.randint(100000, 4000000000)
        server_isn = random.randint(100000, 4000000000)

        # CIC-IDS2017 PortScan: PSH=1 per flow, Init_Win_fwd ~1024-8192
        # Use 'PA' for the probe (not 'S') to get PSH Flag Count=1
        # This matches the CIC-IDS2017 pattern where PSH=1 for 100% of portscan flows
        win_fwd = random.choices([1024, 8192, 29200], weights=[50, 30, 20], k=1)[0]

        # Forward: single data probe with PSH flag
        probe = (Ether(src=SRC_MAC, dst=DST_MAC) /
                 IP(src=src_ip, dst=VICTIM_IP) /
                 TCP(sport=sport, dport=dport, flags='PA', seq=client_isn,
                     ack=server_isn, window=win_fwd) /
                 Raw(load=b'\x00'))  # 1 byte payload
        probe.time = t

        # Response: RST (closed) or small ACK (open) — no SYN-ACK
        resp = (Ether(src=DST_MAC, dst=SRC_MAC) /
                IP(src=VICTIM_IP, dst=src_ip) /
                TCP(sport=dport, dport=sport, flags='R', seq=server_isn,
                    ack=client_isn + 1, window=0))
        resp.time = t + 0.000020
        packets.extend([probe, resp])

    path = write_pcap(packets, "portscan_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_ports} ports scanned)")
    return path


# ---------------------------------------------------------------------------
# Web Attack — Duration ~5.5s, Fwd 17B, Bwd 42B, PSH 90%
# ---------------------------------------------------------------------------

def simulate_webattack(duration=60):
    print(f"\n[WebAttack] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(15, 25)

    usernames = ["admin", "root", "user", "test", "administrator", "guest"]
    passwords = ["password", "123456", "admin", "root", "pass123", "letmein",
                 "qwerty", "abc123", "monkey", "master", "dragon", "login"]

    sqli_paths = [
        "/login?user=admin' OR '1'='1",
        "/search?q=' UNION SELECT *--",
        "/api?id=1; DROP TABLE--",
    ]

    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 2]
        sport = rand_high_port()
        session_start = base + jitter(i * (duration / num_sessions), 1.0, 0)

        # 12 fwd packets, 6 bwd packets (more requests than responses)
        n_fwd = random.randint(10, 14)
        n_bwd = random.randint(5, 7)

        fwd_payloads = []
        for _ in range(n_fwd):
            attack_type = random.choice(["brute", "sqli", "xss"])
            if attack_type == "brute":
                path_str = f"/login?u={random.choice(usernames)[:3]}&p={random.choice(passwords)[:4]}"
            elif attack_type == "sqli":
                path_str = random.choice(sqli_paths)
            else:
                path_str = "/s?q=<script>alert(1)"
            # Trim to ~17 bytes payload (just the path portion)
            payload = path_str.encode()[:random.randint(14, 20)]
            fwd_payloads.append(payload)

        # Server responses ~42 bytes
        bwd_payloads = []
        for _ in range(n_bwd):
            resp = f"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n".encode()
            bwd_payloads.append(resp[:random.randint(38, 46)])

        # IAT ~0.4s between packets
        fwd_iats = [jitter(0.4, 0.15, 0.1) for _ in range(n_fwd)]
        bwd_delays = [jitter(0.05, 0.02, 0.01) for _ in range(n_bwd)]

        # CIC-IDS2017 WebAttack Init_Win:
        # fwd: 8192 (50%), 29200 (30%), 0 (20%)
        # bwd: 235 (55%), 29200 (25%), 0 (20%)
        win_fwd = random.choices([8192, 29200, 0], weights=[50, 30, 20], k=1)[0]
        win_bwd = random.choices([235, 29200, 0], weights=[55, 25, 20], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 80,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=win_fwd, init_win_bwd=win_bwd,
            include_handshake=False,
            include_fin=False,
            fwd_data_flag='PA',  # PSH 90% for WebAttack
            bwd_data_flag='A',
        ))

    path = write_pcap(packets, "webattack_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# FTP Brute Force — Duration ~4s, Port 21, FTP commands/responses
# ---------------------------------------------------------------------------

def simulate_ftp_bruteforce(duration=60):
    print(f"\n[FTP BruteForce] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(20, 30)

    usernames = ["admin", "root", "ftp", "anonymous", "user", "test"]
    passwords = ["password", "123456", "admin", "ftp", "pass", "letmein"]

    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 2]
        sport = rand_high_port()
        session_start = base + jitter(i * (duration / num_sessions), 0.5, 0)

        user = random.choice(usernames)
        pwd = random.choice(passwords)

        # FTP exchange: USER, PASS, QUIT (fwd) — 220, 331, 530, 221 (bwd)
        fwd_payloads = [
            f"USER {user}\r\n".encode(),
            f"PASS {pwd}\r\n".encode(),
            b"QUIT\r\n",
        ]
        # Add extra commands for some sessions to match mean 5.5 fwd packets
        if random.random() > 0.4:
            fwd_payloads.insert(2, f"PASS {random.choice(passwords)}\r\n".encode())
        if random.random() > 0.5:
            fwd_payloads.insert(0, b"FEAT\r\n")

        bwd_payloads = [
            b"220 FTP\r\n",
            b"331 Pass\r\n",
            b"530 Fail\r\n",
            b"221 Bye\r\n",
        ]
        # Extra responses to match mean 7.8 bwd packets
        while len(bwd_payloads) < len(fwd_payloads) + 2:
            bwd_payloads.insert(-1, b"530 Fail\r\n")

        # IAT: moderate spacing (~0.5-0.8s between commands)
        fwd_iats = [jitter(0.6, 0.2, 0.2) for _ in range(len(fwd_payloads))]
        bwd_delays = [jitter(0.1, 0.05, 0.02) for _ in range(len(bwd_payloads))]

        # CIC-IDS2017 FTP Brute Force Init_Win:
        # fwd: 227 (50%), 29200 (50%) — bimodal
        # bwd: 235 (70%), 0 (30%)
        init_win = random.choice([227, 29200])
        win_bwd = random.choices([235, 0], weights=[70, 30], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 21,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=init_win, init_win_bwd=win_bwd,
            include_handshake=False,
            include_fin=False,
            fwd_data_flag='PA',  # PSH 50% for FTP
            bwd_data_flag='PA',
        ))

    path = write_pcap(packets, "ftp_bruteforce_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# SSH Brute Force — Duration ~2.5s, Port 22, SSH banner + binary exchange
# ---------------------------------------------------------------------------

def simulate_ssh_bruteforce(duration=60):
    print(f"\n[SSH BruteForce] Generating pcap ({duration}s window)...")
    base = time.time()
    packets = generate_benign_background(base, duration)
    num_sessions = random.randint(20, 30)

    for i in range(num_sessions):
        src_ip = ATTACKER_IPS[i % 2]
        sport = rand_high_port()
        session_start = base + jitter(i * (duration / num_sessions), 0.3, 0)

        # SSH exchange: banner + ~10 encrypted packets (fwd ~48B each)
        n_kex = random.randint(8, 12)
        fwd_payloads = [b"SSH-2.0-OpenSSH_8.0\r\n"]
        for _ in range(n_kex):
            fwd_payloads.append(os.urandom(random.randint(40, 56)))

        # Server: banner + key exchange responses (~43B each)
        n_bwd = random.randint(14, 18)
        bwd_payloads = [b"SSH-2.0-OpenSSH_7.6p1\r\n"]
        for _ in range(n_bwd - 1):
            bwd_payloads.append(os.urandom(random.randint(35, 50)))

        # Fast IAT (~0.1-0.2s between packets)
        fwd_iats = [jitter(0.12, 0.05, 0.02) for _ in range(len(fwd_payloads))]
        bwd_delays = [jitter(0.03, 0.01, 0.01) for _ in range(len(bwd_payloads))]

        # CIC-IDS2017 SSH Brute Force Init_Win:
        # fwd: 29200 (60%), 8192 (20%), 0 (20%)
        # bwd: 235 (55%), 29200 (25%), 0 (20%)
        win_fwd = random.choices([29200, 8192, 0], weights=[60, 20, 20], k=1)[0]
        win_bwd = random.choices([235, 29200, 0], weights=[55, 25, 20], k=1)[0]

        packets.extend(build_tcp_session(
            src_ip, VICTIM_IP, sport, 22,
            fwd_payloads, bwd_payloads,
            session_start, fwd_iats, bwd_delays,
            init_win_fwd=win_fwd, init_win_bwd=win_bwd,
            include_handshake=False,
            include_fin=False,
            fwd_data_flag='PA',  # PSH 51% for SSH
            bwd_data_flag='PA',
        ))

    path = write_pcap(packets, "ssh_bruteforce_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets, {num_sessions} sessions)")
    return path


# ---------------------------------------------------------------------------
# Benign — Varied IPs, natural timing, standard HTTP
# ---------------------------------------------------------------------------

def simulate_normal_traffic(duration=30):
    print(f"\n[BENIGN] Generating normal traffic pcap ({duration}s window)...")
    base = time.time()
    # Pure benign — use the background generator with more sessions
    packets = generate_benign_background(base, duration, num_sessions=random.randint(150, 200))

    path = write_pcap(packets, "normal_sim")
    print(f"  Saved: {os.path.basename(path)} ({len(packets)} packets)")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(duration=60):
    """Run all attack simulations sequentially."""
    print("=" * 60)
    print("Generating CIC-IDS2017-realistic attack pcaps (Scapy offline)")
    print("=" * 60)

    simulate_normal_traffic(min(duration, 30))
    simulate_dos_hulk(duration)
    simulate_dos_goldeneye(duration)
    simulate_slowloris(min(duration, 120))
    simulate_ddos(duration)
    simulate_portscan(min(duration, 30))
    simulate_webattack(duration)
    simulate_ftp_bruteforce(duration)
    simulate_ssh_bruteforce(duration)

    print("\n" + "=" * 60)
    print("All pcaps generated. Check wireshark/ directory.")
    print("Start the dashboard with: streamlit run app/main.py")
    print("=" * 60)


def run_continuous(interval=60, duration=45):
    """Run attack simulations in a loop."""
    attacks = [
        ("dos-hulk", simulate_dos_hulk),
        ("dos-goldeneye", simulate_dos_goldeneye),
        ("slowloris", simulate_slowloris),
        ("ddos", simulate_ddos),
        ("portscan", simulate_portscan),
        ("webattack", simulate_webattack),
        ("ftp-brute", simulate_ftp_bruteforce),
        ("ssh-brute", simulate_ssh_bruteforce),
        ("normal", simulate_normal_traffic),
    ]

    print(f"Continuous mode: cycling attacks every {interval}s (Ctrl+C to stop)")
    i = 0
    try:
        while True:
            name, func = attacks[i % len(attacks)]
            print(f"\n--- Round {i+1}: {name} ---")
            func(duration)
            remaining = interval - 1  # pcap generation is near-instant
            if remaining > 0:
                print(f"  Waiting {remaining}s before next attack...")
                time.sleep(remaining)
            i += 1
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CIC-IDS2017-realistic attack pcaps using Scapy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack types (matching CIC-IDS2017 feature distributions):
  dos-hulk       Long HTTP floods (~85s flows, 4.8s IAT)
  dos-goldeneye  HTTP keep-alive abuse (~12s flows, 2.3s IAT)
  slowloris      Partial HTTP headers (~97s flows, extreme IAT variability)
  ddos           Short HTTP bursts (~2s flows, 0.5s IAT, multi-source)
  portscan       Rapid SYN probes (~47us per port, 500-2000 ports)
  webattack      Brute force + injection (~5.5s flows)
  ftp-brute      FTP credential stuffing (~4s flows, port 21)
  ssh-brute      SSH credential stuffing (~2.5s flows, port 22)
  normal         Benign HTTP browsing (varied IPs, natural timing)
  all            Generate all attack types

No sudo or network access required — uses Scapy offline packet construction.
""")
    parser.add_argument("--attack",
                        choices=["all", "normal", "dos-hulk", "dos-goldeneye",
                                 "slowloris", "ddos", "portscan", "webattack",
                                 "ftp-brute", "ssh-brute"],
                        default="all", help="Attack type to simulate")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration of traffic window in seconds (default: 60)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=60,
                        help="Interval between attacks in continuous mode")

    args = parser.parse_args()
    os.makedirs(WIRESHARK_DIR, exist_ok=True)

    dispatch = {
        "all": lambda: run_all(args.duration),
        "normal": lambda: simulate_normal_traffic(args.duration),
        "dos-hulk": lambda: simulate_dos_hulk(args.duration),
        "dos-goldeneye": lambda: simulate_dos_goldeneye(args.duration),
        "slowloris": lambda: simulate_slowloris(args.duration),
        "ddos": lambda: simulate_ddos(args.duration),
        "portscan": lambda: simulate_portscan(args.duration),
        "webattack": lambda: simulate_webattack(args.duration),
        "ftp-brute": lambda: simulate_ftp_bruteforce(args.duration),
        "ssh-brute": lambda: simulate_ssh_bruteforce(args.duration),
    }

    if args.continuous:
        run_continuous(args.interval, args.duration)
    else:
        dispatch[args.attack]()


if __name__ == "__main__":
    main()

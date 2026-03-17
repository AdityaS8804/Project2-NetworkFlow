#!/usr/bin/env python3
"""
Attack Simulation & Capture Script — CIC-IDS2017 Realistic Patterns

Simulates network attacks that produce flow features matching the CIC-IDS2017
dataset characteristics. Each attack type mimics the real tool behavior:

  - DoS Hulk:       Long-lived HTTP floods (~85s flows, small requests, large responses)
  - DoS GoldenEye:  HTTP keep-alive abuse (~12s flows, slow requests)
  - DoS Slowloris:  Partial HTTP headers to hold connections open (~97s flows)
  - DDoS:           Short HTTP floods from single source (~2s flows)
  - PortScan:       Rapid SYN probes across many ports (~47μs flows)
  - Web Attack:     Brute force login attempts (~5s flows)
  - FTP Brute Force: Credential stuffing on FTP (~4s flows)
  - SSH Brute Force: Credential stuffing on SSH (~2.5s flows)
  - Benign:         Normal HTTP/HTTPS browsing with varied timing

Usage:
    sudo python simulate_attacks.py --attack all
    sudo python simulate_attacks.py --attack dos-hulk
    sudo python simulate_attacks.py --attack slowloris
    sudo python simulate_attacks.py --attack portscan
    sudo python simulate_attacks.py --continuous --interval 60

Prerequisites:
    pip install scapy       # already installed
    # Requires sudo for raw packet capture and crafting
"""

import os
import sys
import time
import signal
import socket
import random
import argparse
import subprocess
import threading
from datetime import datetime

WIRESHARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wireshark")
CAPTURE_INTERFACE = "lo0"  # loopback on macOS; use "lo" on Linux

# CIC-IDS2017 uses these IPs — we use localhost equivalents
ATTACKER_IP = "127.0.0.1"
VICTIM_IP = "127.0.0.1"


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def start_capture(output_name, duration=60, interface=None):
    """Start packet capture in background using tcpdump."""
    if interface is None:
        interface = CAPTURE_INTERFACE
    output_path = os.path.join(WIRESHARK_DIR, f"{output_name}_{get_timestamp()}.pcapng")

    if subprocess.run(["which", "tshark"], capture_output=True).returncode == 0:
        cmd = ["tshark", "-i", interface, "-a", f"duration:{duration}",
               "-w", output_path, "-F", "pcapng"]
    else:
        cmd = ["sudo", "tcpdump", "-i", interface, "-G", str(duration),
               "-W", "1", "-w", output_path]

    print(f"  Capturing on {interface} -> {os.path.basename(output_path)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    return proc, output_path


def stop_capture(proc):
    """Stop capture process gracefully."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Target servers — simple TCP listeners that respond like real services
# ---------------------------------------------------------------------------

def _start_http_server(port):
    """Start a simple HTTP server that returns large responses (like real web servers)."""
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    return server


def _start_tcp_listener(port):
    """Start a raw TCP listener that accepts and holds connections."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1024)
    sock.settimeout(1.0)

    def _accept_loop():
        conns = []
        while not _stop_flag.is_set():
            try:
                conn, _ = sock.accept()
                conns.append(conn)
            except socket.timeout:
                pass
            except OSError:
                break
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        sock.close()

    _stop_flag = threading.Event()
    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()
    return sock, _stop_flag


# ---------------------------------------------------------------------------
# DoS Hulk — CIC-IDS2017 characteristics:
#   Duration: median 85s, Fwd pkts: 5, Bwd pkts: 4
#   Fwd len: 46B (small GET/POST), Bwd len: 1656B (large response)
#   Flow IAT: ~6.6s between packets, Port: 80
# ---------------------------------------------------------------------------

def simulate_dos_hulk(duration=90):
    """Simulate DoS Hulk: long-lived HTTP connections with periodic requests.

    Hulk sends randomized HTTP requests to bypass caching, keeping connections
    alive for ~85s with ~6.6s between requests.
    """
    print(f"\n[DoS Hulk] Simulating HTTP flood for {duration}s...")
    print("  Pattern: Long-lived connections, ~6.6s between requests, port 80")

    port = 18080
    server = _start_http_server(port)
    proc, path = start_capture("dos_hulk_sim", duration=duration + 10)

    # Create multiple persistent connections that send requests slowly
    num_connections = 50
    threads = []
    stop_event = threading.Event()

    def _hulk_worker(worker_id):
        """One persistent HTTP connection sending periodic requests."""
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                s.connect(("127.0.0.1", port))

                # Send ~5 requests per connection with ~6.6s gaps (like real Hulk)
                for _ in range(random.randint(3, 7)):
                    if stop_event.is_set():
                        break

                    # Randomized GET request (~46 bytes like CIC-IDS2017)
                    rand_param = random.randint(1000, 9999)
                    request = (
                        f"GET /?q={rand_param} HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Connection: keep-alive\r\n"
                        f"\r\n"
                    )
                    s.sendall(request.encode())

                    # Read response (server sends ~1656 bytes back)
                    try:
                        s.recv(4096)
                    except socket.timeout:
                        pass

                    # Wait ~6.6s between requests (matching CIC-IDS2017 IAT)
                    stop_event.wait(random.uniform(5.0, 8.0))

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                stop_event.wait(1)

    for i in range(num_connections):
        t = threading.Thread(target=_hulk_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.1)  # stagger connection starts

    # Let it run
    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    server.kill()
    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# DoS GoldenEye — CIC-IDS2017 characteristics:
#   Duration: median 11.6s, Fwd pkts: 7, Bwd pkts: 5
#   Fwd len: 53B, Bwd len: 1175B, Flow IAT: ~1.1s, Port: 80
# ---------------------------------------------------------------------------

def simulate_dos_goldeneye(duration=60):
    """Simulate DoS GoldenEye: HTTP keep-alive abuse with moderate timing.

    GoldenEye opens many connections and sends requests every ~1.1s,
    each connection lasting ~12s.
    """
    print(f"\n[DoS GoldenEye] Simulating for {duration}s...")
    print("  Pattern: Keep-alive abuse, ~1.1s between requests, ~12s connections")

    port = 18080
    server = _start_http_server(port)
    proc, path = start_capture("dos_goldeneye_sim", duration=duration + 10)

    num_connections = 80
    stop_event = threading.Event()
    threads = []

    def _goldeneye_worker(worker_id):
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect(("127.0.0.1", port))

                # ~7 fwd packets, ~5 bwd over ~12s connection
                for _ in range(random.randint(5, 9)):
                    if stop_event.is_set():
                        break

                    # Randomized request with varied User-Agent (~53 bytes payload)
                    agents = ["Mozilla/5.0", "Chrome/91.0", "Safari/537.36", "curl/7.68"]
                    request = (
                        f"GET /?{random.randint(1, 9999)} HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"User-Agent: {random.choice(agents)}\r\n"
                        f"Connection: keep-alive\r\n"
                        f"\r\n"
                    )
                    s.sendall(request.encode())
                    try:
                        s.recv(4096)
                    except socket.timeout:
                        pass

                    # ~1.1s IAT between requests
                    stop_event.wait(random.uniform(0.8, 1.5))

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                stop_event.wait(0.5)

    for i in range(num_connections):
        t = threading.Thread(target=_goldeneye_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.05)

    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    server.kill()
    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# DoS Slowloris — CIC-IDS2017 characteristics:
#   Duration: median 97s, Fwd pkts: 3, Bwd pkts: 2
#   Fwd len: 8B (tiny headers), Bwd len: 0B (server starved)
#   Extremely low throughput (~2.23 bytes/s), Port: 80
# ---------------------------------------------------------------------------

def simulate_slowloris(duration=120):
    """Simulate Slowloris: hold connections open with partial HTTP headers.

    Sends incomplete HTTP headers very slowly, keeping connections alive
    for ~97s while sending only ~8 bytes at a time.
    """
    print(f"\n[Slowloris] Simulating for {duration}s...")
    print("  Pattern: Partial headers, ~8 bytes/send, connections held ~97s")

    port = 18080
    server = _start_http_server(port)
    proc, path = start_capture("dos_slowloris_sim", duration=duration + 10)

    num_connections = 100
    stop_event = threading.Event()
    threads = []

    def _slowloris_worker(worker_id):
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10)
                s.connect(("127.0.0.1", port))

                # Send initial partial header
                s.sendall(b"GET / HTTP/1.1\r\n")
                s.sendall(f"Host: 127.0.0.1:{port}\r\n".encode())

                # Keep connection alive by sending partial headers (~8 bytes each)
                # for ~97s, with long pauses between sends
                for _ in range(random.randint(2, 5)):
                    if stop_event.is_set():
                        break

                    # Send tiny header fragment (~8 bytes, matching CIC-IDS2017)
                    header = f"X-a: {random.randint(1, 99)}\r\n"
                    s.sendall(header.encode())

                    # Long pause between sends (total ~97s / ~3 sends = ~30s each)
                    stop_event.wait(random.uniform(20.0, 40.0))

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                stop_event.wait(2)

    for i in range(num_connections):
        t = threading.Thread(target=_slowloris_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.2)

    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    server.kill()
    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# DDoS — CIC-IDS2017 characteristics:
#   Duration: median 1.9s, Fwd pkts: 4, Bwd pkts: 4
#   Fwd len: 7B (tiny), Bwd len: 1934B (large response)
#   Flow IAT: ~489ms, Port: 80
# ---------------------------------------------------------------------------

def simulate_ddos(duration=60):
    """Simulate DDoS: short burst HTTP floods.

    Many short-lived connections (~2s each) sending small requests
    that trigger large responses.
    """
    print(f"\n[DDoS] Simulating for {duration}s...")
    print("  Pattern: Short bursts ~2s, 4 pkts each, ~489ms IAT")

    port = 18080
    server = _start_http_server(port)
    proc, path = start_capture("ddos_sim", duration=duration + 10)

    stop_event = threading.Event()
    threads = []

    def _ddos_worker():
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)
                s.connect(("127.0.0.1", port))

                # ~4 requests over ~2s connection
                for _ in range(random.randint(3, 5)):
                    if stop_event.is_set():
                        break
                    request = (
                        f"GET / HTTP/1.1\r\n"
                        f"Host: 127.0.0.1\r\n"
                        f"\r\n"
                    )
                    s.sendall(request.encode())
                    try:
                        s.recv(4096)
                    except socket.timeout:
                        pass
                    # ~489ms IAT
                    stop_event.wait(random.uniform(0.3, 0.7))

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                pass
            # Brief pause before next connection
            stop_event.wait(random.uniform(0.1, 0.5))

    # Many concurrent workers to create high volume
    for _ in range(100):
        t = threading.Thread(target=_ddos_worker, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    server.kill()
    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# PortScan — CIC-IDS2017 characteristics:
#   Duration: median 47 microseconds (extremely fast)
#   Fwd pkts: 1, Bwd pkts: 1 (SYN/SYN-ACK)
#   Fwd len: 0B, Bwd len: 6B
#   Scans thousands of ports rapidly
# ---------------------------------------------------------------------------

def simulate_portscan(duration=30):
    """Simulate PortScan: rapid TCP connect probes across many ports.

    Each probe is ~47μs: send SYN, get RST or SYN-ACK, done.
    Scans port 1-1024+ sequentially.
    """
    print(f"\n[PortScan] Scanning ports for {duration}s...")
    print("  Pattern: Rapid SYN probes, ~47μs per port, 1 pkt each direction")

    proc, path = start_capture("portscan_sim", duration=duration + 5)

    # Use nmap if available (produces most realistic scan patterns)
    if subprocess.run(["which", "nmap"], capture_output=True).returncode == 0:
        print("  Using nmap SYN scan")
        # -T4 aggressive timing, scan many ports
        subprocess.run(
            ["nmap", "-sS", "-T4", "-p", "1-4096", "--max-retries", "1",
             "127.0.0.1"],
            capture_output=True, timeout=duration + 5,
        )
    else:
        # TCP connect scan with minimal delay
        print("  Using Python socket scan")
        end_time = time.time() + duration
        port = 1
        scanned = 0
        while time.time() < end_time and port <= 65535:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.05)  # Very short timeout like real scanner
                s.connect_ex(("127.0.0.1", port))
                s.close()
            except OSError:
                pass
            port += 1
            scanned += 1
        print(f"  Scanned {scanned} ports")

    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# Web Attack — CIC-IDS2017 characteristics:
#   Duration: median 5.5s, Fwd pkts: 3, Bwd pkts: 1
#   Fwd len: 0B, Bwd len: 0B (small payloads)
#   Flow IAT: ~1.8s, Fwd IAT: ~2.7s, Port: 80
# ---------------------------------------------------------------------------

def simulate_webattack(duration=60):
    """Simulate Web Attacks: brute force login + injection attempts.

    Each connection lasts ~5.5s with ~3 requests and ~2.7s between them.
    """
    print(f"\n[WebAttack] Simulating for {duration}s...")
    print("  Pattern: Login brute force, ~5.5s connections, ~2.7s between requests")

    port = 18083
    server = _start_http_server(port)
    proc, path = start_capture("webattack_sim", duration=duration + 10)

    stop_event = threading.Event()
    threads = []

    # Credential lists (like real brute force tools)
    usernames = ["admin", "root", "user", "test", "administrator", "guest"]
    passwords = ["password", "123456", "admin", "root", "pass123", "letmein",
                 "qwerty", "abc123", "monkey", "master", "dragon", "login"]

    # Attack payload templates
    sqli_payloads = [
        "/login?user=admin' OR '1'='1&pass=x",
        "/search?q=' UNION SELECT * FROM users--",
        "/api?id=1; DROP TABLE users--",
    ]
    xss_payloads = [
        "/search?q=<script>alert(1)</script>",
        "/comment?t=<img src=x onerror=alert(1)>",
    ]

    def _webattack_worker():
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(8)
                s.connect(("127.0.0.1", port))

                # ~3 requests per connection over ~5.5s
                attack_type = random.choice(["brute", "sqli", "xss"])
                for _ in range(random.randint(2, 4)):
                    if stop_event.is_set():
                        break

                    if attack_type == "brute":
                        user = random.choice(usernames)
                        pwd = random.choice(passwords)
                        path_str = f"/login?user={user}&pass={pwd}"
                    elif attack_type == "sqli":
                        path_str = random.choice(sqli_payloads)
                    else:
                        path_str = random.choice(xss_payloads)

                    request = (
                        f"GET {path_str} HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Connection: keep-alive\r\n"
                        f"\r\n"
                    )
                    s.sendall(request.encode())
                    try:
                        s.recv(4096)
                    except socket.timeout:
                        pass

                    # ~2.7s between requests (matching CIC-IDS2017 Fwd IAT)
                    stop_event.wait(random.uniform(2.0, 3.5))

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                pass
            stop_event.wait(random.uniform(0.5, 1.5))

    for _ in range(30):
        t = threading.Thread(target=_webattack_worker, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(duration)
    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    server.kill()
    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# FTP Brute Force — CIC-IDS2017 characteristics:
#   Duration: median 4.0s, Fwd pkts: 6, Bwd pkts: 6
#   Fwd len: ~10B (credentials), Bwd len: ~13B (server response)
#   Port: 21
# ---------------------------------------------------------------------------

def simulate_ftp_bruteforce(duration=60):
    """Simulate FTP Brute Force: credential stuffing against FTP.

    Each connection: ~6 exchanges over ~4s. Sends USER/PASS commands.
    """
    print(f"\n[FTP BruteForce] Simulating for {duration}s...")
    print("  Pattern: FTP login attempts, ~4s connections, 6 pkts each, port 21")

    # Start a simple FTP-like listener on port 2121 (avoids needing real FTP)
    ftp_port = 2121
    sock, stop_flag = _start_tcp_listener(ftp_port)
    proc, path = start_capture("ftp_bruteforce_sim", duration=duration + 10)

    stop_event = threading.Event()
    threads = []

    usernames = ["admin", "root", "ftp", "anonymous", "user", "test"]
    passwords = ["password", "123456", "admin", "ftp", "pass", "letmein"]

    def _ftp_worker():
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect(("127.0.0.1", ftp_port))

                # Simulate FTP handshake (~6 exchanges)
                # Banner
                try:
                    s.recv(1024)
                except socket.timeout:
                    pass

                user = random.choice(usernames)
                pwd = random.choice(passwords)

                # USER command (~10 bytes)
                s.sendall(f"USER {user}\r\n".encode())
                stop_event.wait(random.uniform(0.3, 0.8))
                try:
                    s.recv(1024)
                except socket.timeout:
                    pass

                # PASS command
                s.sendall(f"PASS {pwd}\r\n".encode())
                stop_event.wait(random.uniform(0.3, 0.8))
                try:
                    s.recv(1024)
                except socket.timeout:
                    pass

                # QUIT
                s.sendall(b"QUIT\r\n")
                stop_event.wait(random.uniform(0.5, 1.0))

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                pass
            stop_event.wait(random.uniform(0.2, 0.5))

    for _ in range(20):
        t = threading.Thread(target=_ftp_worker, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(duration)
    stop_event.set()
    stop_flag.set()
    for t in threads:
        t.join(timeout=3)

    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# SSH Brute Force — CIC-IDS2017 characteristics:
#   Duration: median 2.5s, Fwd pkts: 14, Bwd pkts: 12
#   Fwd len: ~73B (encrypted auth), Bwd len: ~77B
#   Port: 22
# ---------------------------------------------------------------------------

def simulate_ssh_bruteforce(duration=60):
    """Simulate SSH Brute Force: rapid connection attempts to SSH.

    Each connection: ~14 fwd/12 bwd packets over ~2.5s (SSH handshake + auth).
    """
    print(f"\n[SSH BruteForce] Simulating for {duration}s...")
    print("  Pattern: SSH auth attempts, ~2.5s connections, 14/12 pkts, port 22")

    # Use a high port to avoid needing real SSH
    ssh_port = 2222
    sock, stop_flag = _start_tcp_listener(ssh_port)
    proc, path = start_capture("ssh_bruteforce_sim", duration=duration + 10)

    stop_event = threading.Event()
    threads = []

    def _ssh_worker():
        while not stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)
                s.connect(("127.0.0.1", ssh_port))

                # Simulate SSH handshake + auth (~14 fwd packets, ~73B each)
                # SSH banner exchange
                s.sendall(b"SSH-2.0-OpenSSH_8.0\r\n")
                try:
                    s.recv(1024)
                except socket.timeout:
                    pass

                # Simulate key exchange and auth packets
                for _ in range(random.randint(10, 16)):
                    if stop_event.is_set():
                        break
                    # ~73 byte encrypted packets
                    payload = os.urandom(random.randint(60, 90))
                    s.sendall(payload)
                    stop_event.wait(random.uniform(0.05, 0.2))
                    try:
                        s.recv(1024)
                    except socket.timeout:
                        pass

                s.close()
            except (ConnectionRefusedError, socket.timeout, OSError, BrokenPipeError):
                pass
            stop_event.wait(random.uniform(0.1, 0.3))

    for _ in range(30):
        t = threading.Thread(target=_ssh_worker, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(duration)
    stop_event.set()
    stop_flag.set()
    for t in threads:
        t.join(timeout=3)

    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# Benign traffic — CIC-IDS2017 characteristics:
#   Duration: median 148ms (variable), Fwd pkts: 2, Bwd pkts: 2
#   Varied ports (443, 53, 80), diverse IPs, TCP+UDP mix
# ---------------------------------------------------------------------------

def simulate_normal_traffic(duration=30):
    """Generate normal browsing-like traffic with varied timing."""
    print(f"\n[BENIGN] Generating normal traffic for {duration}s...")
    print("  Pattern: Varied HTTP requests, natural timing, port 80")

    port = 18080
    server = _start_http_server(port)
    proc, path = start_capture("normal_sim", duration=duration + 5)

    end_time = time.time() + duration
    while time.time() < end_time:
        subprocess.run(
            ["curl", "-s", "-o", "/dev/null", f"http://127.0.0.1:{port}/"],
            capture_output=True, timeout=5,
        )
        # Natural browsing: variable 0.5-3s between requests
        time.sleep(random.uniform(0.5, 3.0))

    server.kill()
    stop_capture(proc)
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(duration=60):
    """Run all attack simulations sequentially."""
    print("=" * 60)
    print("Running CIC-IDS2017-realistic attack simulations")
    print("=" * 60)

    simulate_normal_traffic(min(duration, 30))
    time.sleep(3)
    simulate_dos_hulk(duration)
    time.sleep(3)
    simulate_dos_goldeneye(duration)
    time.sleep(3)
    simulate_slowloris(min(duration, 120))
    time.sleep(3)
    simulate_ddos(duration)
    time.sleep(3)
    simulate_portscan(min(duration, 30))
    time.sleep(3)
    simulate_webattack(duration)
    time.sleep(3)
    simulate_ftp_bruteforce(duration)
    time.sleep(3)
    simulate_ssh_bruteforce(duration)

    print("\n" + "=" * 60)
    print("All simulations complete. Check the Streamlit dashboard.")
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
            remaining = interval - duration
            if remaining > 0:
                print(f"  Waiting {remaining}s...")
                time.sleep(remaining)
            i += 1
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    global CAPTURE_INTERFACE
    parser = argparse.ArgumentParser(
        description="Simulate CIC-IDS2017-realistic network attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack types (matching CIC-IDS2017 patterns):
  dos-hulk       Long HTTP floods (~85s flows, 6.6s IAT)
  dos-goldeneye  HTTP keep-alive abuse (~12s flows, 1.1s IAT)
  slowloris      Partial HTTP headers (~97s flows, tiny payloads)
  ddos           Short HTTP bursts (~2s flows, 489ms IAT)
  portscan       Rapid SYN probes (~47μs per port)
  webattack      Brute force + injection (~5.5s flows)
  ftp-brute      FTP credential stuffing (~4s flows)
  ssh-brute      SSH credential stuffing (~2.5s flows)
  normal         Benign HTTP browsing
  all            Run all attacks sequentially
""")
    parser.add_argument("--attack",
                        choices=["all", "normal", "dos-hulk", "dos-goldeneye",
                                 "slowloris", "ddos", "portscan", "webattack",
                                 "ftp-brute", "ssh-brute"],
                        default="all", help="Attack type to simulate")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration per attack in seconds (default: 60)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=60,
                        help="Interval between attacks in continuous mode")
    parser.add_argument("--interface", default=CAPTURE_INTERFACE,
                        help="Capture interface (default: lo0)")

    args = parser.parse_args()
    CAPTURE_INTERFACE = args.interface

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

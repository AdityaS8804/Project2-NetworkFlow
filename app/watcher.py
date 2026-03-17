import os
import glob
import time
import threading
import traceback

from .config import WATCH_DIR, POLL_INTERVAL_SEC
from .pcap_processor import process_pcap

# Minimum age (seconds) before processing a file — avoids reading while tcpdump is still writing
FILE_SETTLE_SEC = 3


class PcapWatcher(threading.Thread):
    """Watches wireshark/ directory for new .pcapng files and processes them."""

    def __init__(self, watch_dir, state, pipeline, poll_interval=POLL_INTERVAL_SEC):
        super().__init__(daemon=True)
        self.watch_dir = watch_dir
        self.state = state
        self.pipeline = pipeline
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # when set, watcher is paused
        self._pending = {}  # filename -> first_seen_time (for settle check)

    def run(self):
        """Poll directory for new .pcapng files."""
        self.state.set_status("Watching for PCAPs...")
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                try:
                    self._check_for_new_files()
                except Exception as e:
                    self.state.add_error(f"Watcher error: {e}")
                    traceback.print_exc()
            self._stop_event.wait(self.poll_interval)

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def _is_file_ready(self, pcap_path):
        """Check file is not still being written (stable size + age check)."""
        try:
            mtime = os.path.getmtime(pcap_path)
            age = time.time() - mtime
            if age < FILE_SETTLE_SEC:
                return False
            # Also check file size > 0
            if os.path.getsize(pcap_path) < 100:
                return False
            return True
        except OSError:
            return False

    def _check_for_new_files(self):
        """Scan for unprocessed .pcapng/.pcap/.csv files."""
        # PCAP files (processed through cicflowmeter)
        pcap_patterns = [
            os.path.join(self.watch_dir, "*.pcapng"),
            os.path.join(self.watch_dir, "*.pcap"),
        ]
        for pattern in pcap_patterns:
            for pcap_path in sorted(glob.glob(pattern)):
                filename = os.path.basename(pcap_path)
                if self.state.is_file_processed(filename):
                    continue

                if not self._is_file_ready(pcap_path):
                    continue

                self.state.set_status(f"Processing {filename}...")
                try:
                    df = process_pcap(pcap_path)
                    if df.empty:
                        self.state.add_error(f"No flows extracted from {filename}")
                    else:
                        self.pipeline.process_new_flows(df, filename)
                    self.state.mark_file_processed(filename)
                    self.state.set_status(f"Processed {filename} ({len(df)} flows)")
                except Exception as e:
                    self.state.add_error(f"Failed to process {filename}: {e}")
                    traceback.print_exc()
                    self.state.mark_file_processed(filename)

        # CSV files (CICFlowMeter format - loaded directly, bypasses PCAP processing)
        csv_pattern = os.path.join(self.watch_dir, "*.csv")
        for csv_path in sorted(glob.glob(csv_pattern)):
            filename = os.path.basename(csv_path)
            if self.state.is_file_processed(filename):
                continue

            if not self._is_file_ready(csv_path):
                continue

            self.state.set_status(f"Processing CSV {filename}...")
            try:
                self.pipeline.load_csv_data(csv_path)
                self.state.mark_file_processed(filename)
            except Exception as e:
                self.state.add_error(f"Failed to process CSV {filename}: {e}")
                traceback.print_exc()
                self.state.mark_file_processed(filename)

        self.state.set_status(f"Watching... ({self.state.get_record_count()} graphs)")

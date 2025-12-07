"""Simple filesystem watcher that runs the pipeline when new images appear.

Designed for running inside a container where host folders are mounted.
It polls the input directory every `POLL_INTERVAL` seconds (default 5).
"""
import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
RUN_PIPELINE = ROOT / "src" / "run_pipeline.py"

# Configure via environment variables (use absolute paths when mounting volumes)
INPUT_DIR = Path(os.environ.get("PHOTO_INPUT", ROOT / "photo_input"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))

# file extensions to consider
EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".heic"]

stop_requested = False


def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


def has_images(directory: Path) -> bool:
    if not directory.exists():
        return False
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            return True
    return False


def main():
    print(f"Watcher starting. Monitoring: {INPUT_DIR}")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    while not stop_requested:
        try:
            if has_images(INPUT_DIR):
                print("New images detected — running pipeline...")
                # run pipeline with same interpreter
                proc = subprocess.run([sys.executable, str(RUN_PIPELINE)], cwd=str(ROOT))
                if proc.returncode != 0:
                    print(f"Pipeline exited with code {proc.returncode}")
                else:
                    print("Pipeline finished successfully.")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"Watcher error: {e}")
            time.sleep(POLL_INTERVAL)

    print("Stop requested — watcher exiting")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the detection then anonymization pipeline.

This script invokes `detector.py` then `anonymizer.py` using the current
Python interpreter (so it will run inside the active virtual environment).
"""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parent.parent
DETECTOR = ROOT / "src" / "detector.py"
ANONYMIZER = ROOT / "src" / "anonymizer.py"


def run_script(path: Path) -> None:
    print(f"\n--- Running {path.name} ---")
    proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"{path.name} exited with code {proc.returncode}")


def main() -> None:
    if not DETECTOR.exists():
        raise SystemExit(f"Detector script not found: {DETECTOR}")
    if not ANONYMIZER.exists():
        raise SystemExit(f"Anonymizer script not found: {ANONYMIZER}")

    run_script(DETECTOR)
    run_script(ANONYMIZER)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

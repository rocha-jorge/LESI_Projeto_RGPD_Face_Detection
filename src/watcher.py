"""Continuous watcher that processes images as they appear using a single YOLO model.

Initializes the YOLO model once, then loops:
- Detects faces on new images
- Saves EXIF coordinates
- Anonymizes detected faces into photo_output

Optional: generate IDs/backups via generate_photo_id.py (disabled by default).
"""
import os
import sys
import time
import signal
import subprocess
from ultralytics import YOLO
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
DETECT_OUTPUT = ROOT / "photo_detection_output"
PHOTO_OUTPUT = ROOT / "photo_output"
ID_SCRIPT = ROOT / "src" / "generate_photo_id.py"

from detector import detector
from anonymizer import face_blur

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


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]


def main():
    print(f"Watcher starting. Monitoring: {INPUT_DIR}")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DETECT_OUTPUT.mkdir(parents=True, exist_ok=True)
    PHOTO_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Initialize YOLO once
    model_path = ROOT / "models" / "yolov8n-face.pt"
    weights_cache = ROOT / "weights" / model_path.name
    if not model_path.exists() and weights_cache.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        weights_cache.replace(model_path)
        print(f"Moved cached weights from {weights_cache} to {model_path}")
    if not model_path.exists():
        print("Downloading YOLOv8-Face model...")
        model = YOLO("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    else:
        model = YOLO(str(model_path))

    while not stop_requested:
        try:
            images = list_images(INPUT_DIR)
            if images:
                print(f"Found {len(images)} image(s). Processing...")
                for img in images:
                    try:
                        # 1) Generate ID/backup per image (must happen first)
                        subprocess.run([sys.executable, str(ID_SCRIPT), str(img)], cwd=str(ROOT))

                        # 2) Copy original to output for traceability (after ID assignment)
                        try:
                            import shutil
                            original_copy = PHOTO_OUTPUT / f"original_{img.name}"
                            if not original_copy.exists():
                                shutil.copy2(str(img), str(original_copy))
                        except Exception as e:
                            print(f"Failed to copy original for {img.name}: {e}")

                        # 3) Detect; blur faces only if any were detected
                        faces = detector(img, model)
                        if faces:
                            detected_path = DETECT_OUTPUT / img.name
                            if detected_path.exists():
                                face_blur(detected_path)
                    except Exception as e:
                        print(f"Error processing {img.name}: {e}")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"Watcher error: {e}")
            time.sleep(POLL_INTERVAL)

    print("Stop requested — watcher exiting")


if __name__ == "__main__":
    main()

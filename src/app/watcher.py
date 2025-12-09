"""Continuous watcher that processes images as they appear using a single YOLO model.

Initializes the YOLO model once, then continuously:
- Generates a per-image ID (timestamp)
- Copies the original to image_output/original_<filename> for traceability
- Detects faces on the image and saves EXIF coordinates
- Blurs faces (face_blur) only when detection finds faces, writing to image_output

Notes:
- ID generation is always performed before copying/detection.
- The model is kept in memory for efficiency (single initialization).
"""



# ADICIONAR 

# Configurable thresholds: Add env/config for detection confidence, IoU, min face size, and blur strength. Makes results tunable without code changes.
# evitar processar ficheiros que ainda não acabaram de ser escritos na pasta de input // Ready file detection: Avoid processing partially copied files by verifying file size is stable (e.g., unchanged across N checks) before processing.

# Non-JPEG metadata: EXIF on PNG/WebP isn’t standard. Keep EXIF for JPEG/TIFF, but add XMP embedding for PNG/WebP to preserve face boxes cross-format. Sidecar JSON as a fallback if XMP lib isn’t available.
# HEIC/HEIF support: You list “.heic” as acceptable — add pillow-heif to read and convert to JPEG/PNG for processing and metadata.

# GPU/CPU selection: Env flag to force CPU if CUDA is missing; log the selected device for clarity.

# Unit tests: For generate_timestamp_name, rename_image, ensure_processable_image, EXIF round-trip (JPEG), and XMP embedding (PNG).
# E2E harness: A small scripted test that drops 3–5 images (no faces + faces of different sizes) and verifies expected outputs in image_output with timings.

# EXIF hygiene: After you write face-location metadata, consider stripping other PII EXIF tags (GPS, camera serial) unless explicitly needed.

# Filesystem events: Replace polling with watchdog for instant reaction and less CPU (still keep a fallback polling mode).

import os
import time
import signal
import logging
import sys
from pathlib import Path

# Ensure 'src' is on the Python path for absolute imports like 'utils.*'
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.logging_setup import init_logging
from utils.paths import *
from utils.setup_model import setup_model
from image_processing.list_images import list_images
from app.image_processing_pipeline import process_image
from input_output.handle_unsupported_files import handle_unsupported_file
from utils.system_metrics import get_process_usage, get_system_usage

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))

# Flag used by the main loop to know when to exit.
# When a shutdown signal is received (CTRL+C or service stop),
# we set this to True and let the loop break cleanly.
stop_requested = False

# Signal handler: flips the flag when termination/interruption is requested.
def handle_sigterm(_signum, _frame):
    global stop_requested
    stop_requested = True

# Bind both SIGTERM (service stop) and SIGINT (CTRL+C) to the handler
# so the watcher can shut down gracefully.
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# setup_environment_and_model now lives in src/utils/setup_environment.py

def _init_logging() -> None:
    init_logging(LOG_DIR, LOG_FILE, logging.INFO)

def main():

    # Initialize logging
    _init_logging()
    logging.info("Watcher starting up")

    # Ensure required directories exist before starting
    ensure_dirs()

    # Setup environment and initialize YOLO once (kept in memory for the life of the process)
    model = setup_model(IMAGE_INPUT)

    while not stop_requested:
        try:
            # Poll the input directory: scan all files once
            all_files = [p for p in IMAGE_INPUT.iterdir() if p.is_file()]
            images = [p for p in all_files if p in set(list_images(IMAGE_INPUT))]
            unsupported_files = [p for p in all_files if p not in images]
            
            # Process each supported image file
            if images:
                logging.info(f"Found {len(images)} image(s). Processing...")
                for img in images:
                    process_image(img, model)
                    
            # After processing supported images, handle unsupported files
            if unsupported_files:
                for unsupported in unsupported_files:
                    handle_unsupported_file(unsupported)

            # Sleep before the next polling iteration
            p_cpu, p_mem = get_process_usage(0.1)
            s_cpu, s_mem = get_system_usage(0.1)
            logging.info(f"Loop resources | Proc CPU: {p_cpu:.1f}% | Proc RAM: {p_mem:.1f} MB | Sys CPU: {s_cpu:.1f}% | Sys RAM: {s_mem:.0f} MB")
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            logging.error("Watcher error", exc_info=True)
            time.sleep(POLL_INTERVAL)             # Even on unexpected errors, keep polling after a short delay

    logging.info("Stop requested — watcher exiting")

if __name__ == "__main__":
    main()

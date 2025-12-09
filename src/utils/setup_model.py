from pathlib import Path
import logging
from ultralytics import YOLO
from utils.paths import IMAGE_INPUT, PROJECT_ROOT
import torch

def log_cuda_status() -> None:
    """Log one-time CUDA/GPU availability and GPU name if present."""
    try:
        if torch.cuda.is_available():
            gpu_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            logging.info(f"CUDA available: True | GPU: {gpu_name} (index {gpu_idx})")
        else:
            logging.info("CUDA available: False | Using CPU")
    except Exception:
        logging.warning("Unable to query CUDA/GPU status.", exc_info=True)


def setup_model(input_dir: Path = IMAGE_INPUT) -> YOLO:
    """Prepare model weights and return an initialized YOLO model.

    Assumes required directories are created by the caller (e.g., watcher).
    If model weights are not present, downloads YOLOv8-Face and saves them.
    """
    logging.info(f"Watcher initializing. Monitoring: {input_dir}")
    # Log one-time CUDA/GPU status at model setup
    log_cuda_status()

    model_path = PROJECT_ROOT / "models" / "yolov8n-face.pt"
    weights_cache = PROJECT_ROOT / "weights" / model_path.name
    if not model_path.exists() and weights_cache.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        weights_cache.replace(model_path)
        logging.info(f"Moved cached weights from {weights_cache} to {model_path}")
    if not model_path.exists():
        logging.info("Downloading YOLOv8-Face model...")
        model = YOLO("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    else:
        model = YOLO(str(model_path))

    # Prefer GPU when available; otherwise CPU
    try:
        if torch.cuda.is_available():
            try:
                model.to('cuda')
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                logging.info(f"Model YOLOv8-Face moved to cuda:0 | GPU: {gpu_name}")
            except Exception:
                logging.warning("Failed to move model to CUDA; staying on CPU.", exc_info=True)
        else:
            logging.info("CUDA not available; using CPU.")

        # Log resolved device string
        device_str = str(getattr(model, 'device', 'cpu'))
        logging.info(f"Model device in use: {device_str}")
    except Exception:
        logging.warning("Unable to determine or set model device.", exc_info=True)
    return model

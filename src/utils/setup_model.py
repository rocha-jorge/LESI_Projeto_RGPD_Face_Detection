from dataclasses import dataclass
from typing import Optional

from pathlib import Path
import logging
from ultralytics import YOLO
from utils.paths import IMAGE_INPUT, PROJECT_ROOT
import torch

def log_cuda_status() -> None:
    """Log one-time CUDA/GPU availability and GPU name if present."""
FACE_URL = (
    "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
)
LICENSE_PLATE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-license-plate.pt"
)

DEFAULT_MODEL_KEY = "face"

# Simple in-process cache to avoid reloading the same model repeatedly
_MODEL_CACHE: dict[str, YOLO] = {}

@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    weights_path: Path
    download_url: Optional[str]

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "face": ModelConfig(
        key="face",
        display_name="YOLOv8-Face",
        weights_path=PROJECT_ROOT / "models" / "yolov8n-face.pt",
        download_url=FACE_URL,
    ),
    "license_plate": ModelConfig(
        key="license_plate",
        display_name="YOLOv8-LicensePlate",
        weights_path=PROJECT_ROOT / "models" / "yolov8n-license-plate.pt",
        download_url=LICENSE_PLATE_URL,
    ),
}

def log_cuda_status() -> None:
    """Log one-time CUDA/GPU availability and GPU name when accessible."""
    try:
        if torch.cuda.is_available():
            gpu_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            logging.info(f"CUDA available: True | GPU: {gpu_name} (index {gpu_idx})")
        else:
            logging.info("CUDA available: False | Using CPU")
    except Exception:  # pragma: no cover - best-effort logging only
        logging.warning("Unable to query CUDA/GPU status.", exc_info=True)

def _get_model_config(model_key: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {model_key}") from exc

def _prepare_weights(config: ModelConfig) -> YOLO:
    model_path = config.weights_path
    cache_path = PROJECT_ROOT / "weights" / model_path.name
    if not model_path.exists() and cache_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.replace(model_path)
        logging.info(f"Moved cached weights from {cache_path} to {model_path}")

    if not model_path.exists():
        if not config.download_url:
            raise FileNotFoundError(
                f"Weights for {config.display_name} not found and no download URL provided."
            )
        logging.info(f"Downloading {config.display_name} model...")
        model = YOLO(config.download_url)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    else:
        model = YOLO(str(model_path))
    return model

def _move_model_to_device(model: YOLO) -> None:
    try:
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                logging.info(
                    f"Model moved to cuda:0 | GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}"
                )
            except Exception:
                logging.warning("Failed to move model to CUDA; staying on CPU.", exc_info=True)
        else:
            logging.info("CUDA not available; using CPU.")
    except Exception:  # pragma: no cover - best-effort logging only
        logging.warning("Unable to determine or set model device.", exc_info=True)

def setup_model(model_key: str = DEFAULT_MODEL_KEY, input_dir: Path = IMAGE_INPUT) -> YOLO:
    """Initialize and return a YOLO model for the requested key."""
    config = _get_model_config(model_key)
    # Return cached model if available (still log weights for visibility)
    cached = _MODEL_CACHE.get(model_key)
    if cached is not None:
        try:
            logging.info(f"INFO | {config.display_name} weights: {config.weights_path.name}")
        except Exception:
            pass
        return cached
    logging.info(f"{config.display_name} initializing. Monitoring: {input_dir}")
    log_cuda_status()
    model = _prepare_weights(config)
    logging.info(f"INFO | {config.display_name} weights: {config.weights_path.name}")
    _move_model_to_device(model)
    device_str = str(getattr(model, "device", "cpu"))
    logging.info(f"Model device in use: {device_str}")
    _MODEL_CACHE[model_key] = model
    return model
    # If model weights are not present, downloads YOLOv8-Face and saves them.




    # logging.info(f"Watcher initializing. Monitoring: {input_dir}")
    # # Log one-time CUDA/GPU status at model setup
    # log_cuda_status()

    # model_path = PROJECT_ROOT / "models" / "yolov8n-face.pt"
    # weights_cache = PROJECT_ROOT / "weights" / model_path.name
    # if not model_path.exists() and weights_cache.exists():
    #     model_path.parent.mkdir(parents=True, exist_ok=True)
    #     weights_cache.replace(model_path)
    #     logging.info(f"Moved cached weights from {weights_cache} to {model_path}")
    # if not model_path.exists():
    #     logging.info("Downloading YOLOv8-Face model...")
    #     model = YOLO("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
    #     model_path.parent.mkdir(parents=True, exist_ok=True)
    #     model.save(model_path)
    # else:
    #     model = YOLO(str(model_path))

    # # Prefer GPU when available; otherwise CPU
    # try:
    #     if torch.cuda.is_available():
    #         try:
    #             model.to('cuda')
    #             gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    #             logging.info(f"Model YOLOv8-Face moved to cuda:0 | GPU: {gpu_name}")
    #         except Exception:
    #             logging.warning("Failed to move model to CUDA; staying on CPU.", exc_info=True)
    #     else:
    #         logging.info("CUDA not available; using CPU.")

    #     # Log resolved device string
    #     device_str = str(getattr(model, 'device', 'cpu'))
    #     logging.info(f"Model device in use: {device_str}")
    # except Exception:
    #     logging.warning("Unable to determine or set model device.", exc_info=True)
    # return model

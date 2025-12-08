from pathlib import Path
from ultralytics import YOLO
from utils.paths import IMAGE_INPUT, PROJECT_ROOT


def setup_environment_and_model(input_dir: Path = IMAGE_INPUT) -> YOLO:
    """Prepare model weights and return an initialized YOLO model.

    Assumes required directories are created by the caller (e.g., watcher).
    If model weights are not present, downloads YOLOv8-Face and saves them.
    """
    print(f"Watcher starting. Monitoring: {input_dir}")

    model_path = PROJECT_ROOT / "models" / "yolov8n-face.pt"
    weights_cache = PROJECT_ROOT / "weights" / model_path.name
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
    print("Model YOLOv8-Face initialized and ready.")
    return model

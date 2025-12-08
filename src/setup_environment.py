from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).parent.parent
PHOTO_OUTPUT = ROOT / "photo_output"
PHOTO_ERROR = ROOT / "photo_error"
PHOTO_INPUT = ROOT / "photo_input"


def setup_environment_and_model(input_dir: Path = PHOTO_INPUT) -> YOLO:
    """Ensure folders exist, prepare model weights, and return initialized YOLO model.

    Directories ensured: photo_input, photo_output, photo_error.
    If model weights are not present, download yolov8n-face.
    """
    print(f"Watcher starting. Monitoring: {input_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    PHOTO_OUTPUT.mkdir(parents=True, exist_ok=True)
    PHOTO_ERROR.mkdir(parents=True, exist_ok=True)

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
    print("Model YOLOv8-Face initialized and ready.")
    return model

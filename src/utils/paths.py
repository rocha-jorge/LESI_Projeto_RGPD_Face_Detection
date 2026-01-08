import os
from pathlib import Path

# Project root (repo root) is two levels up from this file: src/utils/ -> repo
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard directories for image processing

#if ficheiro externo, usar externo, senao usar estes por degfaul
IMAGE_INPUT = Path(os.environ.get("IMAGE_INPUT") or (PROJECT_ROOT / "image_input"))
IMAGE_OUTPUT = Path(os.environ.get("IMAGE_OUTPUT") or (PROJECT_ROOT / "image_output"))
IMAGE_ERROR = Path(os.environ.get("IMAGE_ERROR") or (PROJECT_ROOT / "image_error"))

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "log_registry.log"

# Ensure standard directories exist
def ensure_dirs() -> None:
    """Create standard directories if they don't exist."""
    IMAGE_INPUT.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
    IMAGE_ERROR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

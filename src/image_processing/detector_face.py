from pathlib import Path
import os
import logging
import cv2
from ultralytics import YOLO
from input_output.move_to_error import move_to_error


def _read_threshold(var_name: str, default: float) -> float:
    try:
        val = float(os.environ.get(var_name, str(default)))
        if 0.0 <= val <= 1.0:
            return val
        logging.warning(f"Invalid {var_name} value '{val}', using default {default}")
        return default
    except Exception:
        logging.warning(f"Could not parse {var_name}, using default {default}")
        return default


def detector_face(img_file: Path, model: YOLO) -> list[tuple[int, int, int, int, float]]:
    """
    Minimal face detector: returns a list of (x, y, w, h, conf).
    """
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        logging.error(f"Unsupported file format for detection: {img_file.name}")
        return []

    img = cv2.imread(str(img_file))
    if img is None:
        logging.error(f"Could not read {img_file.name} for detection")
        return []

    conf_thr = _read_threshold("FACE_CONF_THRESHOLD", 0.25)
    logging.debug(f"Confidence threshold (face): {conf_thr}")
    results = model(img, conf=conf_thr, verbose=False)

    # Log Ultralytics per-stage speed in our format
    for res in results:
        spd = getattr(res, "speed", None)
        if isinstance(spd, dict):
            pp = spd.get("preprocess")
            inf = spd.get("inference")
            post = spd.get("postprocess")
            if pp is not None and inf is not None and post is not None:
                logging.debug(
                    f"Ultralytics inference stats: preprocess={pp:.1f}ms inference={inf:.1f}ms postprocess={post:.1f}ms"
                )
    faces_coords: list[tuple[int, int, int, int, float]] = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = None
        try:
            confs = result.boxes.conf.cpu().numpy().reshape(-1)
        except Exception:
            confs = None
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            w, h = x2 - x1, y2 - y1
            conf = float(confs[idx]) if confs is not None and idx < len(confs) else 0.0
            faces_coords.append((int(x1), int(y1), int(w), int(h), conf))
            logging.info(
                f"Detection | face | bbox=({int(x1)},{int(y1)},{int(w)},{int(h)}) | conf={conf:.3f}"
            )

    logging.info(f"Detected {len(faces_coords)} face(s) in {img_file.name}.")
    return faces_coords


def detect_faces(img: Path, model: YOLO) -> tuple[bool, list | None]:
    """Wrapper for detection: returns (ok, faces_or_none). On failure, move to error."""
    try:
        faces = detector_face(img, model)
        return True, faces
    except Exception:
        logging.error(f"Could not apply face detection for {img.name}", exc_info=True)
        move_to_error(img)
        return False, None
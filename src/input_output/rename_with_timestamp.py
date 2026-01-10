from datetime import datetime
from pathlib import Path
import logging
import hashlib

MAX_FULL_PATH = 240  # conservative cap to avoid Windows MAX_PATH issues

def _generate_timestamp_name(path: Path) -> str:
    """Generate a timestamped filename based on the original name.

    Format: YYYYMMDD_HHMMSS_mmm_originalname.ext
    """
    now = datetime.now()
    base = path.name
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
    return f"{ts}_{base}"


def _shorten_if_needed(parent: Path, candidate: str) -> str:
    """Ensure resulting full path stays within a safe length.

    If too long, shorten the stem and add an 8-char hash suffix to preserve uniqueness.
    """
    full = str(parent / candidate)
    if len(full) <= MAX_FULL_PATH:
        return candidate

    p = Path(candidate)
    stem, ext = p.stem, p.suffix
    # Prepare a hash of the original stem for uniqueness
    digest = hashlib.md5(stem.encode("utf-8", errors="ignore")).hexdigest()[:8]

    # Iteratively trim the stem until it fits, keeping at least some chars
    # Reserve space for "_" + hash + ext
    reserve = 1 + len(digest) + len(ext)
    # Worst case: if parent path is already very long, fall back to just hash
    max_name_len = max(8 + reserve, MAX_FULL_PATH - len(str(parent)) - 1)

    # If even minimal name won't fit, return a minimal name with just timestamp+hash
    if max_name_len <= reserve + 10:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        minimal = f"{ts}_{digest}{ext}"
        logging.debug(
            f"Filename shortened (hard fallback) to fit path limit {MAX_FULL_PATH} chars | new='{minimal}'"
        )
        return minimal

    # Base pattern: trimmed_stem + '_' + hash + ext
    # Compute how many stem chars we can keep
    keep = max_name_len - reserve
    trimmed = stem[:keep]
    safe_name = f"{trimmed}_{digest}{ext}"

    # Final guard: if still too long due to parent path, hard fallback
    if len(str(parent / safe_name)) > MAX_FULL_PATH:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_name = f"{ts}_{digest}{ext}"
        logging.debug(
            f"Filename shortened (post-guard) to fit path limit {MAX_FULL_PATH} chars | new='{safe_name}'"
        )
    else:
        logging.debug(
            f"Filename shortened to fit path limit {MAX_FULL_PATH} chars | new='{safe_name}'"
        )
    return safe_name

def _rename_image(path: Path, new_name: str) -> Path:
    """Rename the image in-place (same directory) to new_name and return the new Path."""
    # Ensure name won't exceed safe full path limits
    safe_name = _shorten_if_needed(path.parent, new_name)
    new_path = path.parent / safe_name
    path.rename(new_path)
    return new_path

def rename_with_timestamp(img: Path) -> Path | None:
    """Rename an input file with a timestamp in-place. On failure, return None."""
    try:
        new_name = _generate_timestamp_name(img)
        renamed = _rename_image(img, new_name)
        logging.info(f"Renamed: {img} to {new_name}")
        return renamed
    except Exception:
        logging.error(f"Failed generating/including timestamp ID for {img.name}", exc_info=True)
        return None

import os
import tempfile
from pathlib import Path
import numpy as np
import cv2

DATA_FOLDER = Path.cwd() / "face_guard_data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

def sanitize_name(n: str) -> str:
    out = ''.join([c for c in n if c.isalnum() or c in ('_', '-')])
    return out or 'user'

def center_crop_square(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    side = min(h, w)
    cy = h // 2
    cx = w // 2
    y1 = max(0, cy - side // 2)
    x1 = max(0, cx - side // 2)
    crop = img[y1:y1+side, x1:x1+side]
    resized = cv2.resize(crop, (size, size))
    return resized

def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass

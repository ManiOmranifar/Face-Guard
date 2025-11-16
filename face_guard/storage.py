import json
import shutil
from pathlib import Path
import numpy as np
import io
from .crypto import save_encrypted_file, load_encrypted_file
from .utils import DATA_FOLDER, sanitize_name
import xml.etree.ElementTree as ET
import time

class ProfileStore:
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    def list_admins(self):
        return [p.name for p in DATA_FOLDER.iterdir() if p.is_dir()]

    def get_admin_thumbnail(self, username: str, size=64):
        name = sanitize_name(username)
        folder = DATA_FOLDER / name
        imgs = folder / 'images'
        if not imgs.exists():
            return None
        for f in imgs.iterdir():
            if f.is_file():
                try:
                    raw = load_encrypted_file(f, self.master_key)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    import cv2
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        from .utils import center_crop_square
                        thumb = center_crop_square(img, size)
                        return thumb
                except Exception:
                    continue
        return None

    def create_admin(self, username: str, images: list, encodings: list):
        name = sanitize_name(username)
        folder = DATA_FOLDER / name
        folder.mkdir(parents=True, exist_ok=True)
        imgs_dir = folder / 'images'
        imgs_dir.mkdir(exist_ok=True)
        saved = []
        import cv2, numpy as np
        for i, img in enumerate(images):
            fname = f'img_{int(time.time())}_{i}.png'
            path = imgs_dir / fname
            ok, buf = cv2.imencode('.png', img)
            if not ok:
                continue
            save_encrypted_file(path, self.master_key, buf.tobytes())
            saved.append(fname)
        # save encodings as compressed numpy archive into encrypted file
        enc_path = folder / 'encodings.npz'
        bio = io.BytesIO()
        np.savez_compressed(bio, *[np.array(e, dtype=np.float32) for e in encodings])
        save_encrypted_file(enc_path, self.master_key, bio.getvalue())
        # save xml metadata
        xml_path = folder / f'{name}.xml'
        root = ET.Element('admin')
        ET.SubElement(root, 'username').text = username
        ET.SubElement(root, 'created_at').text = json.dumps({'utc': True})
        imgs_node = ET.SubElement(root, 'images')
        for s in saved:
            ET.SubElement(imgs_node, 'image').text = s
        save_encrypted_file(xml_path, self.master_key, ET.tostring(root, encoding='utf-8'))
        return True

    def delete_admin(self, username: str):
        name = sanitize_name(username)
        folder = DATA_FOLDER / name
        if not folder.exists():
            return False
        try:
            shutil.rmtree(folder)
            return True
        except Exception:
            return False

    def load_encodings_all(self):
        out = {}
        for admin in self.list_admins():
            folder = DATA_FOLDER / admin
            enc_path = folder / 'encodings.npz'
            if not enc_path.exists():
                continue
            try:
                data = load_encrypted_file(enc_path, self.master_key)
                bio = io.BytesIO(data)
                arrs = np.load(bio, allow_pickle=False)
                encs = [arrs[f] for f in arrs.files]
                # ensure all encodings are numpy arrays of dtype float32
                encs = [np.asarray(e, dtype=np.float32) for e in encs]
                out[admin] = encs
            except Exception as e:
                # skip corrupted entry
                print(f'Warning: failed to load encodings for {admin}: {e}')
                continue
        return out

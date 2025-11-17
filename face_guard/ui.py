# face_guard_ui_final.py
"""
Face Guard — Final UX fixes
- Fix: immediate exit if master password wrong at startup (when settings exist)
- Fix: QLineEdit / SpinBox editor styling so typed text and placeholders are visible
- Fix: admin thumbnail cropping to square (center-crop) to avoid stretched icons
- Replace QInputDialog password with a styled custom PasswordDialog for Admin Panel
- Preserve: encrypted storage, xml, enrollment, overlay only after Start, similarity threshold
"""

import sys, os, json, time, base64, threading, traceback
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import face_recognition

from PyQt5 import QtCore, QtGui, QtWidgets

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

# ---------------- CONFIG ----------------
DATA_FOLDER = Path.cwd() / "face_guard_data"
SETTINGS_FILE = DATA_FOLDER / "settings.json"
MASTER_SALT_SIZE = 16
KDF_ITERATIONS = 390000

SIMILARITY_THRESHOLD = 0.98   # default 98%
DEFAULT_CAPTURE_COUNT = 8
CAPTURE_TIMEOUT = 60  # seconds

WINDOW_TITLE = "Face Guard — Final"

# ---------- Utilities (FS + Crypto) ----------
def ensure_data_folder():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

def gen_salt(n=MASTER_SALT_SIZE):
    return os.urandom(n)

def derive_key_from_password(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=KDF_ITERATIONS,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_bytes(key: bytes, data: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_bytes(key: bytes, token: bytes) -> bytes:
    f = Fernet(key)
    return f.decrypt(token)

def save_encrypted_file(path: Path, key: bytes, data: bytes):
    enc = encrypt_bytes(key, data)
    with open(path, "wb") as f:
        f.write(enc)

def load_encrypted_file(path: Path, key: bytes) -> bytes:
    with open(path, "rb") as f:
        enc = f.read()
    return decrypt_bytes(key, enc)

def sanitize_name(n: str) -> str:
    out = "".join([c for c in n if c.isalnum() or c in ("_", "-")])
    return out or "user"

def center_crop_square(img: np.ndarray, size: int):
    # center crop image to square then resize to (size,size)
    h, w = img.shape[:2]
    side = min(h, w)
    cy = h // 2
    cx = w // 2
    y1 = max(0, cy - side // 2)
    x1 = max(0, cx - side // 2)
    crop = img[y1:y1+side, x1:x1+side]
    resized = cv2.resize(crop, (size, size))
    return resized

# ---------- Settings / Master Key ----------
def init_or_load_settings():
    """
    If SETTINGS_FILE absent -> create master password (console).
    If present -> prompt console for master password, derive key and validate:
      - If there are admin folders with encodings, attempt to decrypt at least one encodings.bin.
      - If decryption fails for all, treat password as wrong and exit immediately.
      - If no admin folders exist, accept the derived key.
    """
    ensure_data_folder()
    if not SETTINGS_FILE.exists():
        print("First run: set a master password (console). It will be required to open Admin Panel.")
        while True:
            pw1 = input("Enter new master password: ").strip()
            pw2 = input("Confirm master password: ").strip()
            if len(pw1) < 4:
                print("Password too short (min 4).")
                continue
            if pw1 != pw2:
                print("Passwords don't match.")
                continue
            break
        salt = gen_salt()
        settings = {"master_salt": base64.b64encode(salt).decode("utf-8"), "created_at": datetime.utcnow().isoformat()}
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        key = derive_key_from_password(pw1, salt)
        print("Master password set. Keep it safe.")
        return key, pw1
    else:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        salt = base64.b64decode(settings["master_salt"].encode("utf-8"))
        pw = input("Enter master password to unlock Face Guard (console): ").strip()
        key = derive_key_from_password(pw, salt)

        # validate: if there exist admin encodings, try decrypting at least one
        admins = [p.name for p in DATA_FOLDER.iterdir() if p.is_dir()]
        if not admins:
            # no admins yet, accept key
            return key, pw

        valid = False
        for admin in admins:
            enc_path = DATA_FOLDER / admin / "encodings.bin"
            if enc_path.exists():
                try:
                    _ = load_encrypted_file(enc_path, key)  # if this succeeds, password ok
                    valid = True
                    break
                except Exception:
                    # couldn't decrypt this admin; try next
                    continue
        if not valid:
            print("Master password incorrect — exiting.")
            sys.exit(1)
        return key, pw

# ---------- Profile Store ----------
class ProfileStore:
    def __init__(self, master_key: bytes):
        ensure_data_folder()
        self.master_key = master_key

    def list_admins(self):
        admins = []
        for p in DATA_FOLDER.iterdir():
            if p.is_dir():
                admins.append(p.name)
        return admins

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
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        thumb = center_crop_square(img, size)
                        return thumb
                except Exception:
                    continue
        return None

    def create_admin(self, username: str, images: list, encodings: list):
        name = sanitize_name(username)
        folder = DATA_FOLDER / name
        folder.mkdir(parents=True, exist_ok=True)
        imgs_dir = folder / "images"
        imgs_dir.mkdir(exist_ok=True)
        saved = []
        for i, img in enumerate(images):
            fname = f"img_{int(time.time())}_{i}.png"
            path = imgs_dir / fname
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                continue
            save_encrypted_file(path, self.master_key, buf.tobytes())
            saved.append(fname)
        enc_path = folder / "encodings.bin"
        enc_json = json.dumps({"encodings": encodings})
        save_encrypted_file(enc_path, self.master_key, enc_json.encode("utf-8"))
        xml_path = folder / f"{name}.xml"
        root = ET.Element("admin")
        ET.SubElement(root, "username").text = username
        ET.SubElement(root, "created_at").text = datetime.utcnow().isoformat()
        imgs_node = ET.SubElement(root, "images")
        for s in saved:
            ET.SubElement(imgs_node, "image").text = s
        save_encrypted_file(xml_path, self.master_key, ET.tostring(root, encoding="utf-8"))
        return True

    def delete_admin(self, username: str):
        name = sanitize_name(username)
        folder = DATA_FOLDER / name
        if not folder.exists():
            return False
        try:
            for child in folder.rglob("*"):
                try:
                    if child.is_file(): child.unlink()
                except: pass
            for child in folder.iterdir():
                try:
                    if child.is_dir():
                        for f in child.iterdir():
                            try: f.unlink()
                            except: pass
                        child.rmdir()
                except: pass
            folder.rmdir()
        except Exception:
            pass
        return True

    def load_encodings_all(self):
        out = {}
        for admin in self.list_admins():
            folder = DATA_FOLDER / admin
            enc_path = folder / "encodings.bin"
            if not enc_path.exists():
                continue
            try:
                data = load_encrypted_file(enc_path, self.master_key)
                parsed = json.loads(data.decode("utf-8"))
                encs = parsed.get("encodings", [])
                encs_np = [np.array(e, dtype=np.float64) for e in encs]
                out[admin] = encs_np
            except Exception as e:
                # if decryption fails for this one, skip it (but given validation earlier, this should be rare)
                print(f"Warning: failed to load encodings for {admin}: {e}")
                continue
        return out

# ---------- Similarity ----------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------- Camera Worker ----------
class CameraWorker(QtCore.QObject):
    recognized = QtCore.pyqtSignal(str)
    not_recognized = QtCore.pyqtSignal()
    no_face = QtCore.pyqtSignal()
    frame_ready = QtCore.pyqtSignal(object)

    def __init__(self, encodings_by_user: dict, similarity_threshold: float = SIMILARITY_THRESHOLD):
        super().__init__()
        self._running = False
        self.cap = None
        self.encodings_by_user = encodings_by_user
        self.similarity_threshold = similarity_threshold
        self._lock = threading.Lock()

    def update_encodings(self, encodings_by_user: dict):
        with self._lock:
            self.encodings_by_user = encodings_by_user

    def start(self):
        if self._running:
            return
        self._running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False
        try:
            if self.cap is not None: self.cap.release()
        except: pass

    def _run(self):
        while self._running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1); continue
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb_small, model='hog')
                if len(face_locs) == 0:
                    self.no_face.emit(); self.frame_ready.emit(frame); time.sleep(0.08); continue
                face_encs = face_recognition.face_encodings(rgb_small, face_locs)
                recognized_any = False; recognized_user = None
                with self._lock:
                    for enc in face_encs:
                        best_user = None; best_sim = -1.0
                        for user, known_list in self.encodings_by_user.items():
                            for known in known_list:
                                sim = cosine_similarity(enc, known)
                                if sim > best_sim:
                                    best_sim = sim; best_user = user
                        if best_sim >= self.similarity_threshold:
                            recognized_any = True; recognized_user = best_user; break
                if recognized_any:
                    self.recognized.emit(recognized_user)
                else:
                    self.not_recognized.emit()
                self.frame_ready.emit(frame)
                time.sleep(0.05)
            except Exception as e:
                print("CameraWorker error:", e)
                traceback.print_exc()
                time.sleep(0.5)

# ---------- Custom Password Dialog (styled) ----------
class PasswordDialog(QtWidgets.QDialog):
    def __init__(self, prompt="Enter master password:", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Master Password")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(360, 140)

        layout = QtWidgets.QVBoxLayout()

        # فقط رنگ label سیاه شد
        label = QtWidgets.QLabel(prompt)
        label.setStyleSheet("color: black;")
        layout.addWidget(label)

        self.pw_edit = QtWidgets.QLineEdit()
        self.pw_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pw_edit.setPlaceholderText("Enter master password:")
        layout.addWidget(self.pw_edit)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)

        self.setLayout(layout)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def get_password(self):
        return self.pw_edit.text().strip()

# ---------- GUI Components ----------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # ensure this is a top-level window (no parent) so it covers all screens
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        # Allow translucent background so RGBA colors work
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        # Workaround: set an explicit stylesheet with RGBA black; many platforms honor this when WA_TranslucentBackground is True
        self.setStyleSheet('background-color: rgba(0, 0, 0, 220);')
        # Ensure we're initially hidden
        self.hide()

    def paintEvent(self, event):
        # Defensive paint: if the stylesheet isn't honored on some platform, still paint a solid/semi-transparent black rect
        p = QtGui.QPainter(self)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        color = QtGui.QColor(0, 0, 0, 220)
        p.fillRect(self.rect(), color)
        p.end()

    def lock(self):
        # Show full-screen overlay; try to force it above everything
        try:
            # On multi-screen setups, showFullScreen typically covers the screen where the window is.
            # Move to primary screen geometry to be safe.
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                geom = screen.geometry()
                self.setGeometry(geom)
            # Make sure window is visible on top
            self.setWindowOpacity(1.0)
            self.showFullScreen()
            self.raise_()
            QtWidgets.QApplication.processEvents()
        except Exception:
            try:
                self.showFullScreen(); self.raise_()
            except:
                pass

    def unlock(self):
        try:
            self.hide()
        except:
            pass

class EnrollDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enroll Admin — Capture")
        self.resize(760, 540)
        self.captured_images = []
        self.captured_encodings = []
        self._stop_flag = False
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.username = QtWidgets.QLineEdit()
        self.username.setPlaceholderText("username")
        # متن Username سیاه
        label_username = QtWidgets.QLabel("Username:")
        label_username.setStyleSheet("color: black;")
        form.addRow(label_username, self.username)
        layout.addLayout(form)
        h = QtWidgets.QHBoxLayout()
        self.capture_btn = QtWidgets.QPushButton("Start Capture")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.count_spin = QtWidgets.QSpinBox(); self.count_spin.setRange(3, 60); self.count_spin.setValue(DEFAULT_CAPTURE_COUNT)
        h.addWidget(self.capture_btn); h.addWidget(self.stop_btn); h.addStretch()
        # متن Images سیاه
        images_label = QtWidgets.QLabel("Images:")
        images_label.setStyleSheet("color: black;")
        h.addWidget(images_label); h.addWidget(self.count_spin)
        layout.addLayout(h)
        self.preview = QtWidgets.QLabel(); self.preview.setFixedSize(720, 405); self.preview.setStyleSheet("background:#111; border-radius:8px;")
        layout.addWidget(self.preview, alignment=QtCore.Qt.AlignCenter)
        # thumbnail strip
        self.thumb_strip = QtWidgets.QHBoxLayout()
        layout.addLayout(self.thumb_strip)
        # progress
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        layout.addWidget(self.progress)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        self.setLayout(layout)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.capture_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)

    def start_capture(self):
        target = int(self.count_spin.value())
        self._stop_flag = False
        self.capture_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Camera", "Cannot open camera."); self.capture_btn.setEnabled(True); self.stop_btn.setEnabled(False); return
        captured = 0; imgs = []; encs = []
        start = time.time()
        while captured < target and (time.time() - start) < CAPTURE_TIMEOUT and not self._stop_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05); QtWidgets.QApplication.processEvents(); continue
            disp = cv2.resize(frame, (720, 405))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            qtimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qtimg)
            self.preview.setPixmap(pix)
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb_small, model='hog')
            if locs:
                enc_small = face_recognition.face_encodings(rgb_small, locs)
                if enc_small:
                    full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    full_locs = face_recognition.face_locations(full_rgb, model='hog')
                    full_encs = face_recognition.face_encodings(full_rgb, full_locs)
                    if full_encs:
                        imgs.append(frame.copy())
                        encs.append(full_encs[0].tolist())
                        captured += 1
                        # update progress and thumbs (center-crop square)
                        pval = int((captured/target)*100)
                        self.progress.setValue(pval)
                        thumb = center_crop_square(frame, 64)
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        q = QtGui.QImage(thumb.data, thumb.shape[1], thumb.shape[0], thumb.strides[0], QtGui.QImage.Format_RGB888)
                        lbl = QtWidgets.QLabel(); lbl.setPixmap(QtGui.QPixmap.fromImage(q)); lbl.setFixedSize(68,68)
                        self.thumb_strip.addWidget(lbl)
                        time.sleep(0.25)
            QtWidgets.QApplication.processEvents()
        cap.release()
        self.capture_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if not imgs:
            QtWidgets.QMessageBox.warning(self, "Failed", "No face images captured. Improve lighting and face the camera.")
            return
        self.captured_images = imgs; self.captured_encodings = encs
        QtWidgets.QMessageBox.information(self, "Captured", f"Captured {len(imgs)} images.")

    def stop_capture(self):
        self._stop_flag = True

class ManagementDialog(QtWidgets.QDialog):
    def __init__(self, profile_store: ProfileStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Admin Management")
        self.resize(750, 520)
        self.store = profile_store
        self.init_ui()
        self.refresh_list()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText('Search admins...')
        self.btn_create = QtWidgets.QPushButton("Create Admin")
        top.addWidget(self.search); top.addWidget(self.btn_create)
        layout.addLayout(top)
        self.admin_list = QtWidgets.QListWidget(); self.admin_list.setIconSize(QtCore.QSize(64,64))
        layout.addWidget(self.admin_list)
        h = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete Selected")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        h.addWidget(self.btn_delete); h.addWidget(self.btn_refresh)
        layout.addLayout(h)
        self.setLayout(layout)
        self.btn_create.clicked.connect(self.on_create)
        self.btn_delete.clicked.connect(self.on_delete)
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.search.textChanged.connect(self.on_search)

    def refresh_list(self):
        self.admin_list.clear()
        for a in self.store.list_admins():
            thumb = self.store.get_admin_thumbnail(a, size=64)
            item = QtWidgets.QListWidgetItem(a)
            if thumb is not None:
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                q = QtGui.QImage(thumb_rgb.data, thumb_rgb.shape[1], thumb_rgb.shape[0], thumb_rgb.strides[0], QtGui.QImage.Format_RGB888)
                item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(q)))
            self.admin_list.addItem(item)

    def on_search(self, txt):
        for i in range(self.admin_list.count()):
            it = self.admin_list.item(i)
            it.setHidden(txt.lower() not in it.text().lower())

    def on_create(self):
        dlg = EnrollDialog(self)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            username = dlg.username.text().strip()
            
            if not username:
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setWindowTitle("Invalid")
                msg.setText("Enter username.")
                msg.setStyleSheet("QLabel { color: black; }")
                msg.exec_()
                return
            
            imgs = dlg.captured_images
            encs = dlg.captured_encodings
            
            if not encs:
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setWindowTitle("Error")
                msg.setText("No encodings captured.")
                msg.setStyleSheet("QLabel { color: black; }")
                msg.exec_()
                return
            
            self.store.create_admin(username, imgs, encs)
            
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Done")
            msg.setText(f"Admin '{username}' created.")
            msg.setStyleSheet("QLabel { color: black; }")
            msg.exec_()
            
            self.refresh_list()

    def on_delete(self):
        item = self.admin_list.currentItem()
        if not item:
            return
        u = item.text()
        
        # Confirm dialog
        confirm = QtWidgets.QMessageBox(self)
        confirm.setIcon(QtWidgets.QMessageBox.Question)
        confirm.setWindowTitle("Confirm")
        confirm.setText(f"Delete admin '{u}'?")
        confirm.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        confirm.setStyleSheet("QLabel{ color: black; }")
        ok = confirm.exec_()
        
        if ok == QtWidgets.QMessageBox.Yes:
            self.store.delete_admin(u)
            # Deleted information dialog
            info = QtWidgets.QMessageBox(self)
            info.setIcon(QtWidgets.QMessageBox.Information)
            info.setWindowTitle("Deleted")
            info.setText(f"Admin '{u}' removed.")
            info.setStyleSheet("QLabel{ color: black; }")
            info.exec_()
            self.refresh_list()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, master_key, raw_pw):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1100, 720)
        self.master_key = master_key; self.raw_pw = raw_pw
        self.store = ProfileStore(master_key)
        self.encodings = self.store.load_encodings_all()
        self.camera_worker = CameraWorker(self.encodings, similarity_threshold=SIMILARITY_THRESHOLD)
        self.overlay = OverlayWindow()
        self._monitoring = False
        self.init_ui()
        self.setup_signals()
        self.apply_styles()

    def init_ui(self):
        central = QtWidgets.QWidget(); main_layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        card = QtWidgets.QFrame(); card.setObjectName("card"); card_layout = QtWidgets.QVBoxLayout()
        # controls
        controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Monitoring")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.admin_panel_btn = QtWidgets.QPushButton("Admin Panel")
        controls.addWidget(self.start_btn); controls.addWidget(self.stop_btn); controls.addStretch(); controls.addWidget(self.admin_panel_btn)
        card_layout.addLayout(controls)
        # camera preview
        self.video_label = QtWidgets.QLabel(); self.video_label.setFixedSize(720, 480); self.video_label.setStyleSheet("background:#0f0f0f; border-radius:12px;")
        card_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignCenter)
        # status row
        info = QtWidgets.QHBoxLayout()
        self.status_chip = QtWidgets.QLabel(); self.status_chip.setFixedHeight(28); self.status_chip.setAlignment(QtCore.Qt.AlignCenter)
        self.status_chip.setText("Idle — Monitoring stopped")
        self.threshold_label = QtWidgets.QLabel(f"Threshold: {SIMILARITY_THRESHOLD:.2f}")
        info.addWidget(self.status_chip); info.addStretch(); info.addWidget(self.threshold_label)
        card_layout.addLayout(info)
        card.setLayout(card_layout)
        left.addWidget(card)
        # right sidebar
        right = QtWidgets.QVBoxLayout(); rcard = QtWidgets.QFrame(); rcard.setObjectName("card"); rlay = QtWidgets.QVBoxLayout()
        rlay.addWidget(QtWidgets.QLabel("Profiles"))
        self.admin_scroll = QtWidgets.QListWidget(); self.admin_scroll.setIconSize(QtCore.QSize(48,48)); self.admin_scroll.setFixedHeight(200)
        rlay.addWidget(self.admin_scroll)
        # settings
        rlay.addSpacing(6)
        rlay.addWidget(QtWidgets.QLabel("Settings"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox(); self.threshold_spin.setRange(0.80, 0.999); self.threshold_spin.setSingleStep(0.01); self.threshold_spin.setValue(SIMILARITY_THRESHOLD)
        rlay.addWidget(QtWidgets.QLabel("Similarity threshold:")); rlay.addWidget(self.threshold_spin)
        self.capture_count_spin = QtWidgets.QSpinBox(); self.capture_count_spin.setRange(3, 60); self.capture_count_spin.setValue(DEFAULT_CAPTURE_COUNT)
        rlay.addWidget(QtWidgets.QLabel("Default capture images:")); rlay.addWidget(self.capture_count_spin)
        rlay.addSpacing(8)
        self.reload_btn = QtWidgets.QPushButton("Reload Profiles")
        self.test_btn = QtWidgets.QPushButton("Quick Test")
        rlay.addWidget(self.reload_btn); rlay.addWidget(self.test_btn)
        rlay.addStretch()
        rcard.setLayout(rlay); right.addWidget(rcard)
        main_layout.addLayout(left, stretch=3); main_layout.addLayout(right, stretch=1)
        central.setLayout(main_layout); self.setCentralWidget(central)
        # connect
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.admin_panel_btn.clicked.connect(self.on_admin_panel)
        self.reload_btn.clicked.connect(self.on_reload)
        self.test_btn.clicked.connect(self.on_test)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
        # populate admin list
        self.refresh_admin_list()

    def apply_styles(self):
        qss = """
        * { color: #e9eef8; font-family: 'Segoe UI', Roboto, Arial; }
        #card { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #0b0b0b, stop:1 #151515); border-radius:12px; padding:12px; }
        QLabel { color:#dbe6ff; font-size:13px; }
        QPushButton { background:#2d89ef; color:white; padding:8px; border-radius:8px; }
        QPushButton:disabled { background:#444; color:#bbb; }
        QListWidget { background:#0f1724; border-radius:8px; }
        QProgressBar { height:10px; border-radius:6px; }
        QDoubleSpinBox, QSpinBox { background:#0b1220; color:#e7f0ff; border-radius:6px; padding:4px; }
        QLineEdit {
            background: #06121a;
            color: #e9eef8;
            border: 1px solid #1e2830;
            padding: 6px;
            border-radius: 6px;
        }
        QLineEdit:focus {
            border: 1px solid #2d89ef;
            background: #071826;
        }
        QLineEdit::placeholder {
            color: #9fb3d6;
        }
        QSpinBox QLineEdit, QDoubleSpinBox QLineEdit {
            background: #071226;
            color: #e9eef8;
        }
        """
        self.setStyleSheet(qss)
        self.status_chip.setStyleSheet("background:#223344; padding:4px 12px; border-radius:14px;")

    def setup_signals(self):
        self.camera_worker.frame_ready.connect(self.on_frame)
        self.camera_worker.recognized.connect(self.on_recognized)
        self.camera_worker.not_recognized.connect(self.on_not_recognized)
        self.camera_worker.no_face.connect(self.on_no_face)

    def refresh_admin_list(self):
        self.admin_scroll.clear()
        for a in self.store.list_admins():
            thumb = self.store.get_admin_thumbnail(a, size=48)
            item = QtWidgets.QListWidgetItem(a)
            if thumb is not None:
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                q = QtGui.QImage(thumb_rgb.data, thumb_rgb.shape[1], thumb_rgb.shape[0], thumb_rgb.strides[0], QtGui.QImage.Format_RGB888)
                item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(q)))
            self.admin_scroll.addItem(item)

    def on_start(self):
        if self._monitoring: return
        self.status_chip.setText('Starting camera...')
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._monitoring = True
        self.encodings = self.store.load_encodings_all()
        self.camera_worker.update_encodings(self.encodings)
        self.camera_worker.similarity_threshold = float(self.threshold_spin.value())
        self.camera_worker.start()
        self.status_chip.setText('Monitoring — looking for admins')
        QtCore.QTimer.singleShot(2000, lambda: None)

    def on_stop(self):
        if not self._monitoring: return
        self._monitoring = False
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.status_chip.setText('Stopped')
        try: self.camera_worker.stop()
        except: pass
        self.overlay.unlock()

    def on_admin_panel(self):
        dlg = PasswordDialog(parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        pw = dlg.get_password()
        if not pw:
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Denied")
            msg.setText("Empty password.")
            msg.setStyleSheet("QLabel{ color: black; }")  # متن سیاه
            msg.exec_()
            return
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        salt = base64.b64decode(settings["master_salt"].encode("utf-8"))
        derived = derive_key_from_password(pw, salt)
        if derived != self.master_key:
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Denied")
            msg.setText("Wrong master password.")
            msg.setStyleSheet("QLabel{ color: black; }")  # متن سیاه
            msg.exec_()
            return
        dlg2 = ManagementDialog(self.store, self)
        dlg2.exec_()
        self.refresh_admin_list()
        self.encodings = self.store.load_encodings_all()
        self.camera_worker.update_encodings(self.encodings)

    def on_reload(self):
        self.encodings = self.store.load_encodings_all()
        self.camera_worker.update_encodings(self.encodings)

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Reloaded")
        msg.setText("Profiles reloaded.")
        msg.setIcon(QtWidgets.QMessageBox.Information)

        # فقط رنگ متن پیام سیاه شود
        msg.setStyleSheet("QLabel{ color: black; }")

        msg.exec_()

    def on_test(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Camera")
            msg.setText("Cannot open camera.")
            msg.setStyleSheet("QLabel{ color: black; }")
            msg.exec_()
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Camera")
            msg.setText("Failed to capture.")
            msg.setStyleSheet("QLabel{ color: black; }")
            msg.exec_()
            return
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small, model='hog')
        if not locs:
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("No face")
            msg.setText("No face detected.")
            msg.setStyleSheet("QLabel{ color: black; }")
            msg.exec_()
            return
        encs = face_recognition.face_encodings(rgb_small, locs)
        if not encs:
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("No encoding")
            msg.setText("Failed to compute encoding.")
            msg.setStyleSheet("QLabel{ color: black; }")
            msg.exec_()
            return
        enc = encs[0]
        best_user, best_sim = None, -1.0
        for user, knowns in self.encodings.items():
            for known in knowns:
                sim = cosine_similarity(enc, known)
                if sim > best_sim:
                    best_sim = sim
                    best_user = user
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("Test Result")
        msg.setText(f"Best: {best_user} (sim={best_sim:.4f})
Threshold={self.threshold_spin.value():.3f}")
        msg.setStyleSheet("QLabel{ color: black; }")
        msg.exec_()

    def on_threshold_changed(self, v):
        self.threshold_label.setText(f'Threshold: {v:.3f}')

    def on_frame(self, frame):
        disp = cv2.resize(frame, (720, 480))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qtimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qtimg)
        self.video_label.setPixmap(pix)

    def on_recognized(self, username):
        self.status_chip.setText(f'Recognized: {username}')
        self.overlay.unlock()

    def on_not_recognized(self):
        self.status_chip.setText('Unknown person — overlay active')
        if self._monitoring:
            self.overlay.lock()

    def on_no_face(self):
        self.status_chip.setText('No face — overlay active')
        if self._monitoring:
            self.overlay.lock()

    def closeEvent(self, event):
        try: self.camera_worker.stop()
        except: pass
        event.accept()

# ---------- Entry ----------
def main():
    key, raw_pw = init_or_load_settings()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(key, raw_pw)
    win.show()
    # do NOT lock overlay at startup
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

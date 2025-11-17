# face_guard/ui.py
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import numpy as np
import time
import json
import base64

import face_recognition

from .storage import ProfileStore
from .dialogs import PasswordDialog, CreateMasterDialog
from .camera_worker import CameraWorker
from .utils import center_crop_square, DATA_FOLDER
from .crypto import derive_key_from_password
from pathlib import Path

SETTINGS_FILE = DATA_FOLDER / 'settings.json'
SIMILARITY_THRESHOLD = 0.98
DEFAULT_CAPTURE_COUNT = 8
CAPTURE_TIMEOUT = 60


# Helper to show styled message boxes so text is visible on dark/light themes
def show_message(parent, title: str, text: str, icon=QtWidgets.QMessageBox.Information):
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(title)
    msg.setText(text)
    # Force label text color to black so it's visible on light backgrounds (and consistent)
    # Keep only styling for QLabel so other controls aren't impacted.
    msg.setStyleSheet("QLabel{ color: black; }")
    msg.exec_()


class EnrollDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Enroll Admin — Capture')
        self.resize(760, 540)
        self.captured_images = []
        self.captured_encodings = []
        self._stop_flag = False
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.username = QtWidgets.QLineEdit()
        self.username.setPlaceholderText('username')
        label_username = QtWidgets.QLabel('Username:')
        label_username.setStyleSheet('color: black;')
        form.addRow(label_username, self.username)
        layout.addLayout(form)

        h = QtWidgets.QHBoxLayout()
        self.capture_btn = QtWidgets.QPushButton('Start Capture')
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(3, 60)
        self.count_spin.setValue(DEFAULT_CAPTURE_COUNT)
        h.addWidget(self.capture_btn)
        h.addWidget(self.stop_btn)
        h.addStretch()
        images_label = QtWidgets.QLabel('Images:')
        images_label.setStyleSheet('color: black;')
        h.addWidget(images_label)
        h.addWidget(self.count_spin)
        layout.addLayout(h)

        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(720, 405)
        self.preview.setStyleSheet('background:#111; border-radius:8px;')
        layout.addWidget(self.preview, alignment=QtCore.Qt.AlignCenter)

        self.thumb_strip = QtWidgets.QHBoxLayout()
        layout.addLayout(self.thumb_strip)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
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
        self.capture_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            show_message(self, 'Camera', 'Cannot open camera.', QtWidgets.QMessageBox.Warning)
            self.capture_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return

        captured = 0
        imgs = []
        encs = []
        begin = time.time()

        try:
            while captured < target and (time.time() - begin) < CAPTURE_TIMEOUT and not self._stop_flag:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    QtWidgets.QApplication.processEvents()
                    continue

                disp = cv2.resize(frame, (720, 405))
                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                qtimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(qtimg)
                self.preview.setPixmap(pix)

                # operate on a smaller frame for detection/encoding
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                # face_recognition is imported at module top
                locs = face_recognition.face_locations(rgb_small, model='hog')
                if locs:
                    enc_small = face_recognition.face_encodings(rgb_small, locs)
                    if enc_small:
                        # compute encoding on a larger frame for better accuracy
                        full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        full_locs = face_recognition.face_locations(full_rgb, model='hog')
                        full_encs = face_recognition.face_encodings(full_rgb, full_locs)
                        if full_encs:
                            imgs.append(frame.copy())
                            encs.append(full_encs[0].tolist())
                            captured += 1
                            pval = int((captured / target) * 100)
                            self.progress.setValue(pval)

                            thumb = center_crop_square(frame, 64)
                            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                            q = QtGui.QImage(thumb.data, thumb.shape[1], thumb.shape[0], thumb.strides[0], QtGui.QImage.Format_RGB888)
                            lbl = QtWidgets.QLabel()
                            lbl.setPixmap(QtGui.QPixmap.fromImage(q))
                            lbl.setFixedSize(68, 68)
                            self.thumb_strip.addWidget(lbl)
                            time.sleep(0.25)

                QtWidgets.QApplication.processEvents()
        finally:
            try:
                cap.release()
            except:
                pass

        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if not imgs:
            show_message(self, 'Failed', 'No face images captured. Improve lighting and face the camera.', QtWidgets.QMessageBox.Warning)
            return

        self.captured_images = imgs
        self.captured_encodings = encs
        show_message(self, 'Captured', f'Captured {len(imgs)} images.', QtWidgets.QMessageBox.Information)

    def stop_capture(self):
        self._stop_flag = True


class ManagementDialog(QtWidgets.QDialog):
    def __init__(self, profile_store, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Admin Management')
        self.resize(750, 520)
        self.store = profile_store
        self.init_ui()
        self.refresh_list()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText('Search admins...')
        self.btn_create = QtWidgets.QPushButton('Create Admin')
        top.addWidget(self.search)
        top.addWidget(self.btn_create)
        layout.addLayout(top)

        self.admin_list = QtWidgets.QListWidget()
        self.admin_list.setIconSize(QtCore.QSize(64, 64))
        layout.addWidget(self.admin_list)

        h = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton('Delete Selected')
        self.btn_refresh = QtWidgets.QPushButton('Refresh')
        h.addWidget(self.btn_delete)
        h.addWidget(self.btn_refresh)
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
                show_message(self, 'Invalid', 'Enter username.', QtWidgets.QMessageBox.Warning)
                return
            imgs = dlg.captured_images
            encs = dlg.captured_encodings
            if not encs:
                show_message(self, 'Error', 'No encodings captured.', QtWidgets.QMessageBox.Warning)
                return
            self.store.create_admin(username, imgs, encs)
            show_message(self, 'Done', f"Admin '{username}' created.", QtWidgets.QMessageBox.Information)
            self.refresh_list()

    def on_delete(self):
        item = self.admin_list.currentItem()
        if not item:
            return
        u = item.text()
        confirm = QtWidgets.QMessageBox(self)
        confirm.setIcon(QtWidgets.QMessageBox.Question)
        confirm.setWindowTitle('Confirm')
        confirm.setText(f"Delete admin '{u}'?")
        confirm.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        confirm.setStyleSheet("QLabel{ color: black; }")
        ok = confirm.exec_()
        if ok == QtWidgets.QMessageBox.Yes:
            self.store.delete_admin(u)
            show_message(self, 'Deleted', f"Admin '{u}' removed.", QtWidgets.QMessageBox.Information)
            self.refresh_list()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, master_key):
        super().__init__()
        self.setWindowTitle('Face Guard — Final')
        self.resize(1100, 720)
        self.master_key = master_key
        self.store = ProfileStore(master_key)
        self.encodings = self.store.load_encodings_all()
        self.camera_worker = CameraWorker(self.encodings, similarity_threshold=SIMILARITY_THRESHOLD)
        # simple overlay widget
        self.overlay = QtWidgets.QWidget()
        self.overlay.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.overlay.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
        self.overlay.setStyleSheet("background: black;")
        self.overlay.hide()
        self._monitoring = False
        self.init_ui()
        self.setup_signals()
        self.apply_styles()

    def init_ui(self):
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        card = QtWidgets.QFrame()
        card.setObjectName('card')
        card_layout = QtWidgets.QVBoxLayout()

        # controls
        controls = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Start Monitoring')
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.admin_panel_btn = QtWidgets.QPushButton('Admin Panel')
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        controls.addWidget(self.admin_panel_btn)
        card_layout.addLayout(controls)

        # camera preview
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(720, 480)
        self.video_label.setStyleSheet('background:#0f0f0f; border-radius:12px;')
        card_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignCenter)

        # status row
        info = QtWidgets.QHBoxLayout()
        self.status_chip = QtWidgets.QLabel()
        self.status_chip.setFixedHeight(28)
        self.status_chip.setAlignment(QtCore.Qt.AlignCenter)
        self.status_chip.setText('Idle — Monitoring stopped')
        self.threshold_label = QtWidgets.QLabel(f'Threshold: {SIMILARITY_THRESHOLD:.2f}')
        info.addWidget(self.status_chip)
        info.addStretch()
        info.addWidget(self.threshold_label)
        card_layout.addLayout(info)
        card.setLayout(card_layout)
        left.addWidget(card)

        # right sidebar
        right = QtWidgets.QVBoxLayout()
        rcard = QtWidgets.QFrame()
        rcard.setObjectName('card')
        rlay = QtWidgets.QVBoxLayout()
        rlay.addWidget(QtWidgets.QLabel('Profiles'))
        self.admin_scroll = QtWidgets.QListWidget()
        self.admin_scroll.setIconSize(QtCore.QSize(48, 48))
        self.admin_scroll.setFixedHeight(200)
        rlay.addWidget(self.admin_scroll)
        rlay.addSpacing(6)
        rlay.addWidget(QtWidgets.QLabel('Settings'))
        from PyQt5.QtWidgets import QDoubleSpinBox
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.80, 0.999)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(SIMILARITY_THRESHOLD)
        rlay.addWidget(QtWidgets.QLabel('Similarity threshold:'))
        rlay.addWidget(self.threshold_spin)
        self.capture_count_spin = QtWidgets.QSpinBox()
        self.capture_count_spin.setRange(3, 60)
        self.capture_count_spin.setValue(DEFAULT_CAPTURE_COUNT)
        rlay.addWidget(QtWidgets.QLabel('Default capture images:'))
        rlay.addWidget(self.capture_count_spin)
        rlay.addSpacing(8)
        self.reload_btn = QtWidgets.QPushButton('Reload Profiles')
        self.test_btn = QtWidgets.QPushButton('Quick Test')
        rlay.addWidget(self.reload_btn)
        rlay.addWidget(self.test_btn)
        rlay.addStretch()
        rcard.setLayout(rlay)
        right.addWidget(rcard)

        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=1)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # connect signals
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
        QLineEdit { background: #06121a; color: #e9eef8; border: 1px solid #1e2830; padding: 6px; border-radius: 6px; }
        QLineEdit:focus { border: 1px solid #2d89ef; background: #071826; }
        QLineEdit::placeholder { color: #9fb3d6; }
        QSpinBox QLineEdit, QDoubleSpinBox QLineEdit { background: #071226; color: #e9eef8; }
        """
        self.setStyleSheet(qss)
        self.status_chip.setStyleSheet('background:#223344; padding:4px 12px; border-radius:14px;')

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
        if self._monitoring:
            return
        self.status_chip.setText('Starting camera...')
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._monitoring = True
        self.encodings = self.store.load_encodings_all()
        self.camera_worker.update_encodings(self.encodings)
        self.camera_worker.similarity_threshold = float(self.threshold_spin.value())
        # camera_worker uses QThread methods
        self.camera_worker.start_worker()
        self.status_chip.setText('Monitoring — looking for admins')

    def on_stop(self):
        if not self._monitoring:
            return
        self._monitoring = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_chip.setText('Stopped')
        try:
            self.camera_worker.stop_worker()
        except:
            pass
        try:
            self.overlay.hide()
        except:
            pass

    def on_admin_panel(self):
        dlg = PasswordDialog(parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        pw = dlg.get_password()
        if not pw:
            show_message(self, 'Denied', 'Empty password.', QtWidgets.QMessageBox.Warning)
            return

        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        salt = base64.b64decode(settings['master_salt'].encode('utf-8'))
        derived = derive_key_from_password(pw, salt)
        if derived != self.master_key:
            show_message(self, 'Denied', 'Wrong master password.', QtWidgets.QMessageBox.Warning)
            return

        # ManagementDialog is defined in this module; instantiate directly
        dlg2 = ManagementDialog(self.store, self)
        dlg2.exec_()
        self.refresh_admin_list()
        self.encodings = self.store.load_encodings_all()
        self.camera_worker.update_encodings(self.encodings)

    def on_reload(self):
        self.encodings = self.store.load_encodings_all()
        self.camera_worker.update_encodings(self.encodings)
        show_message(self, 'Reloaded', 'Profiles reloaded.', QtWidgets.QMessageBox.Information)

    def on_test(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            show_message(self, 'Camera', 'Cannot open camera.', QtWidgets.QMessageBox.Warning)
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            show_message(self, 'Camera', 'Failed to capture.', QtWidgets.QMessageBox.Warning)
            return

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small, model='hog')
        if not locs:
            show_message(self, 'No face', 'No face detected.', QtWidgets.QMessageBox.Information)
            return
        encs = face_recognition.face_encodings(rgb_small, locs)
        if not encs:
            show_message(self, 'No encoding', 'Failed to compute encoding.', QtWidgets.QMessageBox.Information)
            return

        enc = encs[0]
        best_user, best_sim = None, -1.0
        for user, knowns in self.encodings.items():
            for known in knowns:
                sim = float(np.dot(enc, known) / (np.linalg.norm(enc) * np.linalg.norm(known) + 1e-12))
                if sim > best_sim:
                    best_sim = sim
                    best_user = user

        show_message(self, 'Test Result', f'Best: {best_user} (sim={best_sim:.4f})\\nThreshold={self.threshold_spin.value():.3f}', QtWidgets.QMessageBox.Information)

    def on_threshold_changed(self, v):
        self.threshold_label.setText(f'Threshold: {v:.3f}')

    def on_frame(self, frame):
        try:
            disp = cv2.resize(frame, (720, 480))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            qtimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qtimg)
            self.video_label.setPixmap(pix)
        except Exception:
            # avoid crashing UI rendering on unexpected frame data
            pass

    def on_recognized(self, username):
        self.status_chip.setText(f'Recognized: {username}')
        try:
            self.overlay.hide()
        except:
            pass

    def on_not_recognized(self):
        self.status_chip.setText('Unknown person — overlay active')
        if self._monitoring:
            try:
                self.overlay.showFullScreen()
                self.overlay.raise_()
            except:
                pass

    def on_no_face(self):
        self.status_chip.setText('No face — overlay active')
        if self._monitoring:
            try:
                self.overlay.showFullScreen()
                self.overlay.raise_()
            except:
                pass

    def closeEvent(self, event):
        try:
            self.camera_worker.stop_worker()
        except:
            pass
        event.accept()

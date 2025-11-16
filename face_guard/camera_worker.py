from PyQt5 import QtCore
import numpy as np
import cv2
import time
import traceback
import face_recognition

class CameraWorker(QtCore.QThread):
    recognized = QtCore.pyqtSignal(str)
    not_recognized = QtCore.pyqtSignal()
    no_face = QtCore.pyqtSignal()
    frame_ready = QtCore.pyqtSignal(object)

    def __init__(self, encodings_by_user: dict, similarity_threshold: float = 0.98, parent=None):
        super().__init__(parent)
        self._running = False
        self.encodings_by_user = encodings_by_user or {}
        self.similarity_threshold = similarity_threshold
        self._lock = QtCore.QReadWriteLock()
        self.cap = None
        # cached flattened knowns
        self._flat_users = []
        self._flat_encs = None  # numpy array (N,D)

        self._rebuild_cache()

    def update_encodings(self, encodings_by_user: dict):
        with QtCore.QWriteLocker(self._lock):
            self.encodings_by_user = encodings_by_user or {}
            self._rebuild_cache()

    def _rebuild_cache(self):
        users = []
        mats = []
        for user, knowns in (self.encodings_by_user or {}).items():
            for k in knowns:
                users.append(user)
                mats.append(np.asarray(k, dtype=np.float32))
        if mats:
            mat = np.stack(mats)  # (N, D)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            mat = mat / norms
            self._flat_users = users
            self._flat_encs = mat
        else:
            self._flat_users = []
            self._flat_encs = None

    def start_worker(self):
        if self.isRunning():
            return
        self._running = True
        self.start()

    def stop_worker(self):
        self._running = False
        # wait for thread to finish gracefully
        self.wait(timeout=2000)
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass

    def run(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print('Cannot open camera')
                return
            while self._running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    face_locs = face_recognition.face_locations(rgb_small, model='hog')
                    if len(face_locs) == 0:
                        self.no_face.emit()
                        self.frame_ready.emit(frame)
                        time.sleep(0.08)
                        continue
                    face_encs = face_recognition.face_encodings(rgb_small, face_locs)
                    recognized_any = False
                    recognized_user = None
                    if face_encs:
                        # use normalized enc
                        for enc in face_encs:
                            e = np.asarray(enc, dtype=np.float32)
                            norm = np.linalg.norm(e) + 1e-12
                            e = e / norm
                            # build local snapshot of cache
                            with QtCore.QReadLocker(self._lock):
                                flat_encs = self._flat_encs
                                flat_users = self._flat_users.copy()
                            if flat_encs is not None and flat_encs.shape[0] > 0:
                                sims = flat_encs.dot(e)
                                idx = int(np.argmax(sims))
                                best_sim = float(sims[idx])
                                best_user = flat_users[idx]
                                if best_sim >= self.similarity_threshold:
                                    recognized_any = True
                                    recognized_user = best_user
                                    break
                    if recognized_any:
                        self.recognized.emit(recognized_user)
                    else:
                        self.not_recognized.emit()
                    self.frame_ready.emit(frame)
                    time.sleep(0.05)
                except Exception as e:
                    print('CameraWorker loop error:', e)
                    traceback.print_exc()
                    time.sleep(0.5)
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            except:
                pass

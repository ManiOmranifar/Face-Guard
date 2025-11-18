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

    # --- Liveness Configuration ---
    # EAR (Eye Aspect Ratio) threshold for blink detection
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 1
    
    # MAR (Mouth Aspect Ratio) threshold for mouth movement detection
    MOUTH_AR_THRESH = 0.6  # If MAR exceeds this, mouth is considered open
    MOUTH_AR_CONSEC_FRAMES = 2
    
    # Head movement threshold (distance in pixels)
    HEAD_MOVEMENT_THRESH = 15  # Minimum movement to consider valid
    HEAD_MOVEMENT_FRAMES = 3   # Check movement over these frames
    
    # Time (seconds) allowed to pass the liveness check before resetting
    LIVENESS_TIMEOUT = 6.0
    # ------------------------------

    def __init__(self, encodings_by_user: dict, similarity_threshold: float = 0.98, parent=None):
        super().__init__(parent)
        self._running = False
        self.encodings_by_user = encodings_by_user or {}
        self.similarity_threshold = similarity_threshold
        self._lock = QtCore.QReadWriteLock()
        self.cap = None

        # Cached flattened knowns
        self._flat_users = []
        self._flat_encs = None  # numpy array (N,D)

        # --- Liveness State Variables ---
        # States: SEARCHING, CHECKING_LIVENESS, AUTHENTICATED
        self._liveness_state = "SEARCHING" 
        self._candidate_user = None
        self._liveness_timer_start = 0
        
        # Blink detection
        self._blink_counter = 0
        self._consec_frames_closed = 0
        
        # Mouth movement detection
        self._mouth_open_counter = 0
        self._consec_frames_mouth_open = 0
        
        # Head movement detection
        self._head_positions = []  # Store recent face center positions
        self._head_movement_detected = False

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
        self.wait(timeout=2000)
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass

    def _calculate_ear(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) to detect blinking.
        """
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        if C == 0:
            return 0
        ear = (A + B) / (2.0 * C)
        return ear

    def _calculate_mar(self, mouth_points):
        """
        Calculate Mouth Aspect Ratio (MAR) to detect mouth opening.
        """
        # Vertical distances (top to bottom of mouth)
        A = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[10]))
        B = np.linalg.norm(np.array(mouth_points[4]) - np.array(mouth_points[8]))
        # Horizontal distance (left to right corner)
        C = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))

        if C == 0:
            return 0
        mar = (A + B) / (2.0 * C)
        return mar

    def _check_head_movement(self, face_location):
        """
        Check if head has moved significantly by tracking face center position.
        Returns True if significant movement detected.
        """
        top, right, bottom, left = face_location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        current_pos = (center_x, center_y)
        
        # Add current position to history
        self._head_positions.append(current_pos)
        
        # Keep only recent positions
        if len(self._head_positions) > self.HEAD_MOVEMENT_FRAMES:
            self._head_positions.pop(0)
        
        # Check if we have enough frames to compare
        if len(self._head_positions) >= self.HEAD_MOVEMENT_FRAMES:
            # Calculate total movement distance
            first_pos = self._head_positions[0]
            last_pos = self._head_positions[-1]
            distance = np.sqrt((last_pos[0] - first_pos[0])**2 + 
                             (last_pos[1] - first_pos[1])**2)
            
            if distance >= self.HEAD_MOVEMENT_THRESH:
                return True
        
        return False

    def _reset_liveness_state(self):
        """Reset all liveness detection variables."""
        self._liveness_state = "SEARCHING"
        self._candidate_user = None
        self._blink_counter = 0
        self._consec_frames_closed = 0
        self._mouth_open_counter = 0
        self._consec_frames_mouth_open = 0
        self._head_positions = []
        self._head_movement_detected = False

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

                    # Resize for speed
                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    # Detect faces
                    face_locs = face_recognition.face_locations(
                        rgb_small, model='hog')

                    if len(face_locs) == 0:
                        if self._liveness_state != "SEARCHING":
                            self._reset_liveness_state()
                        
                        self.no_face.emit()
                        self.frame_ready.emit(frame)
                        time.sleep(0.08)
                        continue
                    
                    top, right, bottom, left = [c * 2 for c in face_locs[0]]
                    
                    if self._liveness_state == "AUTHENTICATED":
                        box_color = (0, 255, 0)
                        status_text = "AUTHENTICATED"
                    elif self._liveness_state == "CHECKING_LIVENESS":
                        box_color = (255, 165, 0)
                        status_text = "CHECKING LIVENESS..."
                    else:
                        box_color = (255, 0, 0)
                        status_text = "SEARCHING"
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                    cv2.putText(frame, status_text, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # --- STATE 3: AUTHENTICATED (فقط باید صورت را دنبال کند) ---
                    if self._liveness_state == "AUTHENTICATED":
                        self.recognized.emit(self._candidate_user)
                        self.frame_ready.emit(frame)
                        time.sleep(0.05)
                        continue

                    # --- STATE 1: SEARCHING FOR USER ---
                    if self._liveness_state == "SEARCHING":
                        face_encs = face_recognition.face_encodings(
                            rgb_small, face_locs)
                        found_user = None

                        if face_encs:
                            # Only process the first face found
                            enc = face_encs[0]
                            e = np.asarray(enc, dtype=np.float32)
                            norm = np.linalg.norm(e) + 1e-12
                            e = e / norm

                            with QtCore.QReadLocker(self._lock):
                                flat_encs = self._flat_encs
                                flat_users = self._flat_users.copy()

                            if flat_encs is not None and flat_encs.shape[0] > 0:
                                sims = flat_encs.dot(e)
                                idx = int(np.argmax(sims))
                                best_sim = float(sims[idx])
                                best_user = flat_users[idx]

                                if best_sim >= self.similarity_threshold:
                                    found_user = best_user

                        if found_user:
                            # print(f"Identity matched: {found_user}. Starting Liveness Check...")
                            self._liveness_state = "CHECKING_LIVENESS"
                            self._candidate_user = found_user
                            self._liveness_timer_start = time.time()
                            self._consec_frames_closed = 0
                            self._blink_counter = 0
                            self._consec_frames_mouth_open = 0
                            self._mouth_open_counter = 0
                            self._head_positions = []
                            self._head_movement_detected = False
                        else:
                            self.not_recognized.emit()
                    
                    # --- STATE 2: CHECKING LIVENESS (MULTI-METHOD DETECTION) ---
                    elif self._liveness_state == "CHECKING_LIVENESS":
                        # Check for timeout
                        if time.time() - self._liveness_timer_start > self.LIVENESS_TIMEOUT:
                            # print("Liveness Check Timed Out.")
                            self.not_recognized.emit()
                            self._reset_liveness_state()
                        else:
                            # Get Landmarks for the face (only for the first face)
                            landmarks_list = face_recognition.face_landmarks(
                                rgb_small, face_locs[:1])

                            liveness_passed = False

                            if landmarks_list:
                                landmarks = landmarks_list[0]
                                
                                # --- METHOD 1: BLINK DETECTION ---
                                left_eye = landmarks['left_eye']
                                right_eye = landmarks['right_eye']

                                leftEAR = self._calculate_ear(left_eye)
                                rightEAR = self._calculate_ear(right_eye)
                                ear = (leftEAR + rightEAR) / 2.0

                                if ear < self.EYE_AR_THRESH:
                                    self._consec_frames_closed += 1
                                else:
                                    if self._consec_frames_closed >= self.EYE_AR_CONSEC_FRAMES:
                                        # print("✓ Blink Detected! Liveness Confirmed.")
                                        liveness_passed = True
                                    self._consec_frames_closed = 0

                                # --- METHOD 2: MOUTH MOVEMENT DETECTION ---
                                if not liveness_passed and 'top_lip' in landmarks and 'bottom_lip' in landmarks:
                                    mouth_points = landmarks['top_lip'] + landmarks['bottom_lip']
                                    mar = self._calculate_mar(mouth_points)

                                    if mar > self.MOUTH_AR_THRESH:
                                        self._consec_frames_mouth_open += 1
                                        if self._consec_frames_mouth_open >= self.MOUTH_AR_CONSEC_FRAMES:
                                            # print("✓ Mouth Movement Detected! Liveness Confirmed.")
                                            liveness_passed = True
                                    else:
                                        self._consec_frames_mouth_open = 0

                            # --- METHOD 3: HEAD MOVEMENT DETECTION ---
                            if not liveness_passed:
                                # Scale face location back to original frame
                                scaled_face_loc = tuple([c * 2 for c in face_locs[0]])
                                if self._check_head_movement(scaled_face_loc):
                                    # print("✓ Head Movement Detected! Liveness Confirmed.")
                                    liveness_passed = True

                            # --- CHECK IF LIVENESS PASSED ---
                            if liveness_passed:
                                # SUCCESS: User is real and alive
                                self._liveness_state = "AUTHENTICATED"
                                self.recognized.emit(self._candidate_user)
                                time.sleep(0.1)
                            else:
                                # Still checking...
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

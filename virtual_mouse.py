"""
Real-Time Hand Gesture-Controlled PC Cursor System (AI Virtual Mouse)
---------------------------------------------------------------------
Tech Stack: Python 3.13+, OpenCV, MediaPipe Tasks API (HandLandmarker), PyAutoGUI, NumPy
Features:
  - Modern MediaPipe HandLandmarker (21-point 3D hand tracking in VIDEO mode)
  - Automatic one-time download of Google's official hand_landmarker.task model
  - Adaptive One Euro Filter for sub-pixel cursor stabilization (zero jitter)
  - Active Region Mapping for ergonomic corner-to-corner screen reach
  - Euclidean distance pinch detection with debounce & hysteresis
  - Gestures: Move, Left Click, Right Click, Scroll, Drag & Drop
"""

import os
import time
import math
import urllib.request
from typing import Tuple, Optional

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Disable PyAutoGUI's built-in pause and fail-safe corner trigger for zero-latency execution
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Standard 21-point hand skeleton connections for drawing landmarks manually
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky & Palm
]


def ensure_model_exists():
    """Downloads the official MediaPipe hand_landmarker.task model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model (one-time setup, ~7.5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")


class OneEuroFilter:
    """
    Adaptive low-pass filter for noisy real-time signals (Casiez et al., CHI 2012).
    - At low speeds (hovering/clicking): heavily smooths coordinates to eliminate micro-jitter.
    - At high speeds (flicking across screen): reduces smoothing to eliminate lag.
    """
    def __init__(self, t0: float, x0: float, dx0: float = 0.0,
                 min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class VirtualMouseController:
    def __init__(
        self,
        cam_width: int = 640,
        cam_height: int = 480,
        frame_margin_x: int = 110,
        frame_margin_y: int = 90,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        click_threshold: float = 32.0,
        release_threshold: float = 42.0,
        click_cooldown: float = 0.35
    ):
        ensure_model_exists()

        # Screen & Camera Dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        self.cam_w = cam_width
        self.cam_h = cam_height

        # Active Region Box
        self.margin_x = frame_margin_x
        self.margin_y = frame_margin_y

        # Noise Filter Parameters
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None

        # Gesture Thresholds
        self.click_thresh = click_threshold
        self.release_thresh = release_threshold
        self.cooldown = click_cooldown
        self.last_click_time = 0.0
        self.is_dragging = False
        self.pinch_start_time = 0.0

        # Modern MediaPipe Tasks API HandLandmarker (VIDEO mode for temporal tracking)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.70
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def get_fingers_up(self, lm_list: list) -> list:
        """Returns binary list [Thumb, Index, Middle, Ring, Pinky] indicating raised fingers."""
        fingers = []
        # Thumb check (works for mirrored palm-facing view)
        if lm_list[self.tip_ids[0]][1] < lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers: tip Y-coordinate above PIP joint Y-coordinate
        for i in range(1, 5):
            if lm_list[self.tip_ids[i]][2] < lm_list[self.tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    @staticmethod
    def find_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Computes Euclidean distance between two 2D landmark points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def map_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Maps camera active-box coordinates to full PC screen resolution with clamping."""
        screen_x = np.interp(x, (self.margin_x, self.cam_w - self.margin_x), (0, self.screen_w))
        screen_y = np.interp(y, (self.margin_y, self.cam_h - self.margin_y), (0, self.screen_h))

        screen_x = max(0, min(self.screen_w - 1, screen_x))
        screen_y = max(0, min(self.screen_h - 1, screen_y))
        return int(screen_x), int(screen_y)

    @staticmethod
    def draw_hand_skeleton(frame: np.ndarray, lm_list: list):
        """Draws clean, lightweight skeleton lines and joint nodes directly via OpenCV."""
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (lm_list[start_idx][1], lm_list[start_idx][2])
            pt2 = (lm_list[end_idx][1], lm_list[end_idx][2])
            cv2.line(frame, pt1, pt2, (200, 200, 200), 2)

        for _, cx, cy in lm_list:
            cv2.circle(frame, (cx, cy), 4, (0, 140, 255), cv2.FILLED)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError("Error: Could not access webcam. Ensure camera permissions are enabled.")

        prev_time = time.perf_counter()
        start_time = prev_time
        last_timestamp_ms = -1
        status_text = "Initializing..."

        try:
            while True:
                loop_start = time.perf_counter()
                success, frame = cap.read()
                if not success:
                    continue

                # Mirror frame horizontally for intuitive movement
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Monotonically increasing timestamp in ms required by VIDEO mode
                timestamp_ms = int((loop_start - start_time) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                # Draw ergonomic active tracking boundary
                cv2.rectangle(
                    frame,
                    (self.margin_x, self.margin_y),
                    (self.cam_w - self.margin_x, self.cam_h - self.margin_y),
                    (255, 180, 0), 2
                )

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]

                    lm_list = []
                    for idx, lm in enumerate(hand_landmarks):
                        cx, cy = int(lm.x * self.cam_w), int(lm.y * self.cam_h)
                        lm_list.append((idx, cx, cy))

                    self.draw_hand_skeleton(frame, lm_list)

                    fingers = self.get_fingers_up(lm_list)
                    thumb_tip = (lm_list[4][1], lm_list[4][2])
                    index_tip = (lm_list[8][1], lm_list[8][2])
                    middle_tip = (lm_list[12][1], lm_list[12][2])
                    index_mcp = (lm_list[5][1], lm_list[5][2])

                    now = time.perf_counter()

                    # 1. SCROLL MODE: Index + Middle + Ring fingers up
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                        status_text = "MODE: SCROLL"
                        if index_tip[1] < self.cam_h // 2 - 40:
                            pyautogui.scroll(35)
                        elif index_tip[1] > self.cam_h // 2 + 40:
                            pyautogui.scroll(-35)

                    # 2. RIGHT CLICK MODE: Index + Middle fingers up, touch tips together
                    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                        dist_idx_mid = self.find_distance(index_tip, middle_tip)
                        cv2.line(frame, index_tip, middle_tip, (0, 255, 255), 2)
                        status_text = "MODE: RIGHT-CLICK READY"

                        if dist_idx_mid < self.click_thresh and (now - self.last_click_time) > self.cooldown:
                            pyautogui.rightClick()
                            self.last_click_time = now
                            status_text = "ACTION: RIGHT CLICK"
                            cv2.circle(frame, index_tip, 12, (0, 0, 255), cv2.FILLED)

                    # 3. CURSOR MOVE & LEFT CLICK / DRAG MODE: Index finger raised
                    elif fingers[1] == 1 and fingers[2] == 0:
                        anchor_x = int(0.7 * index_tip[0] + 0.3 * index_mcp[0])
                        anchor_y = int(0.7 * index_tip[1] + 0.3 * index_mcp[1])

                        raw_x, raw_y = self.map_coordinates(anchor_x, anchor_y)

                        if self.filter_x is None or self.filter_y is None:
                            self.filter_x = OneEuroFilter(now, raw_x, min_cutoff=self.min_cutoff, beta=self.beta)
                            self.filter_y = OneEuroFilter(now, raw_y, min_cutoff=self.min_cutoff, beta=self.beta)
                            smooth_x, smooth_y = raw_x, raw_y
                        else:
                            smooth_x = int(self.filter_x(now, raw_x))
                            smooth_y = int(self.filter_y(now, raw_y))

                        pyautogui.moveTo(smooth_x, smooth_y)
                        cv2.circle(frame, (anchor_x, anchor_y), 8, (0, 255, 0), cv2.FILLED)

                        pinch_dist = self.find_distance(thumb_tip, index_tip)
                        cv2.line(frame, thumb_tip, index_tip, (255, 0, 255), 2)

                        if pinch_dist < self.click_thresh:
                            cv2.circle(frame, index_tip, 12, (0, 255, 255), cv2.FILLED)
                            if self.pinch_start_time == 0.0:
                                self.pinch_start_time = now

                            if (now - self.pinch_start_time) > 0.40 and not self.is_dragging:
                                pyautogui.mouseDown()
                                self.is_dragging = True
                                status_text = "ACTION: DRAGGING"
                            elif not self.is_dragging:
                                status_text = "PINCH DETECTED"

                        elif pinch_dist > self.release_thresh:
                            if self.is_dragging:
                                pyautogui.mouseUp()
                                self.is_dragging = False
                                status_text = "ACTION: DROP"
                            elif self.pinch_start_time != 0.0:
                                if (now - self.last_click_time) > self.cooldown:
                                    pyautogui.click()
                                    self.last_click_time = now
                                    status_text = "ACTION: LEFT CLICK"
                            self.pinch_start_time = 0.0
                        else:
                            if not self.is_dragging:
                                status_text = "MODE: CURSOR MOVE"

                    # 4. NEUTRAL / FIST: Pause tracking safely
                    else:
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                        self.pinch_start_time = 0.0
                        status_text = "MODE: PAUSED (NEUTRAL)"

                else:
                    if self.is_dragging:
                        pyautogui.mouseUp()
                        self.is_dragging = False
                    self.pinch_start_time = 0.0
                    self.filter_x = None
                    self.filter_y = None
                    status_text = "WARNING: NO HAND DETECTED"

                # Telemetry HUD (FPS & Latency)
                curr_time = time.perf_counter()
                latency_ms = (curr_time - loop_start) * 1000
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                cv2.rectangle(frame, (0, 0), (self.cam_w, 40), (20, 20, 20), cv2.FILLED)
                cv2.putText(
                    frame,
                    f"{status_text} | FPS: {int(fps)} | Latency: {latency_ms:.1f}ms",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2
                )

                cv2.imshow("AI Virtual Mouse - Gesture Controller (Press 'Q' to Exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            if self.is_dragging:
                pyautogui.mouseUp()
            self.landmarker.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = VirtualMouseController()
    controller.run()
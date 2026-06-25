"""
src/gesture/hand_controller.py — Hand Gesture Controller

Sử dụng MediaPipe Tasks API để nhận diện cử động tay từ webcam,
map thành action điều khiển con chim trong Flappy Bird.

Gesture logic:
  - Mở bàn tay (ngón tay duỗi ra) → FLAP (action=1)
  - Nắm tay (ngón tay co lại)     → KHÔNG FLAP (action=0)
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from typing import Optional

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    raise ImportError(
        "mediapipe chưa được cài đặt. Chạy: pip install mediapipe"
    )

class HandController:
    """
    Điều khiển Flappy Bird bằng cử động tay qua webcam.
    Sử dụng thư viện MediaPipe Tasks API (phiên bản mới thay cho solutions).
    """

    FLAP_FINGER_THRESHOLD = 3  # >= 3 ngón duỗi → flap
    FLAP_COOLDOWN = 0.15

    # Connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (5, 9), (9, 10), (10, 11), (11, 12),   # middle
        (9, 13), (13, 14), (14, 15), (15, 16), # ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky
    ]

    def __init__(
        self,
        camera_id: int = 0,
        flip_horizontal: bool = True,
        show_landmarks: bool = True,
    ):
        self.camera_id = camera_id
        self.flip_horizontal = flip_horizontal
        self.show_landmarks = show_landmarks

        # 1. Download model file if not exists
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            print(f"[HandController] Đang tải mô hình MediaPipe (chỉ tải 1 lần)...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print(f"[HandController] Đã tải xong.")

        # 2. Init MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 3. Mở Webcam
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể mở camera (id={camera_id}).")
        
        # Giảm resolution để tăng tốc
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # State
        self._last_flap_time: float = 0.0
        self._current_action: int = 0
        self._fingers_up: int = 0
        self._hand_detected: bool = False
        self._preview_frame: Optional[np.ndarray] = None
        self._fps_counter: float = 0.0
        self._fps_time: float = time.time()
        self._frame_count: int = 0

        print(f"[HandController] Camera {camera_id} đã mở. Giơ tay để điều khiển!")

    def get_action(self) -> int:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return 0

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Detect hands bằng MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        self._hand_detected = False
        self._fingers_up = 0

        # Phân tích kết quả
        if detection_result.hand_landmarks:
            self._hand_detected = True
            hand_landmarks = detection_result.hand_landmarks[0]
            self._fingers_up = self._count_fingers(hand_landmarks)

            if self.show_landmarks:
                self._draw_landmarks(frame, hand_landmarks)

        action = self._decide_action()
        self._current_action = action

        # Vẽ HUD
        self._draw_hud(frame)
        self._preview_frame = frame

        # Tính FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._fps_counter = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()

        return action

    def get_preview_frame(self) -> Optional[np.ndarray]:
        return self._preview_frame

    def close(self):
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
        if self._cap and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        print("[HandController] Camera đã đóng.")

    @property
    def is_hand_detected(self) -> bool:
        return self._hand_detected

    @property
    def fingers_up_count(self) -> int:
        return self._fingers_up

    @property
    def fps(self) -> float:
        return self._fps_counter

    # ──────────────────────────────────────────────
    # PRIVATE: Gesture Analysis
    # ──────────────────────────────────────────────

    def _count_fingers(self, landmarks) -> int:
        """
        Đếm số ngón tay đang duỗi ra (extended).
        landmarks: mảng gồm 21 điểm NormalizedLandmark (x, y, z)
        """
        fingers = 0

        # Thumb: Landmark 4 (tip) vs Landmark 3 (IP)
        # Sử dụng khoảng cách x giữa tip và mcp để xác định thumb (vì đã flip horizontal)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        if abs(thumb_tip.x - thumb_mcp.x) > 0.05:
            fingers += 1

        # Index finger: Landmark 8 (tip) vs Landmark 6 (PIP)
        if landmarks[8].y < landmarks[6].y:
            fingers += 1

        # Middle finger: Landmark 12 (tip) vs Landmark 10 (PIP)
        if landmarks[12].y < landmarks[10].y:
            fingers += 1

        # Ring finger: Landmark 16 (tip) vs Landmark 14 (PIP)
        if landmarks[16].y < landmarks[14].y:
            fingers += 1

        # Pinky: Landmark 20 (tip) vs Landmark 18 (PIP)
        if landmarks[20].y < landmarks[18].y:
            fingers += 1

        return fingers

    def _decide_action(self) -> int:
        if not self._hand_detected:
            return 0
        if self._fingers_up >= self.FLAP_FINGER_THRESHOLD:
            now = time.time()
            if now - self._last_flap_time >= self.FLAP_COOLDOWN:
                self._last_flap_time = now
                return 1
        return 0

    def _draw_landmarks(self, frame: np.ndarray, landmarks):
        """Vẽ landmarks tay thủ công."""
        h, w = frame.shape[:2]
        pts = []
        # Chuyển đổi normalized coords sang pixel
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        
        # Vẽ kết nối
        for connection in self.HAND_CONNECTIONS:
            pt1 = pts[connection[0]]
            pt2 = pts[connection[1]]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    def _draw_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if self._hand_detected:
            status = f"Hand: {self._fingers_up} fingers"
            color = (0, 255, 0)
        else:
            status = "No hand detected"
            color = (0, 0, 255)

        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        action_text = "FLAP!" if self._current_action == 1 else "---"
        action_color = (0, 255, 255) if self._current_action == 1 else (150, 150, 150)
        cv2.putText(frame, f"Action: {action_text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {self._fps_counter:.0f}", (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Open hand = FLAP | Fist = Fall", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    def __repr__(self):
        return f"HandController(camera={self.camera_id}, hand={'detected' if self._hand_detected else 'none'}, fingers={self._fingers_up})"

"""
src/vision/feature_extractor.py  — IMP302 Feature Extraction

Nhận ProcessedFrame từ ImageProcessor, tính state vector:
    state = [dx, dy, velocity]

  dx       : khoảng cách ngang từ chim đến pipe tiếp theo (normalized)
  dy       : khoảng cách dọc từ chim đến tâm khe hở gap (normalized)
  velocity : vận tốc ước tính của chim theo Y (normalized)

Tất cả features được normalize về [-1, 1].
"""

import numpy as np
from typing import Optional, Tuple

from .image_processor import ImageProcessor, ProcessedFrame


class FeatureExtractor:
    """
    Trích xuất state vector từ raw frame.

    State vector: [dx, dy, velocity]  — shape (3,)
    """

    STATE_DIM = 3  # Số chiều của state vector

    def __init__(self, frame_h: int = 512, frame_w: int = 288):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self._processor = ImageProcessor(frame_h, frame_w)

        # Tracking velocity bằng bird position qua các frame
        self._prev_bird_y: Optional[float] = None
        self._smoothed_velocity: float = 0.0
        self._velocity_alpha: float = 0.6  # EMA smoothing factor

    def extract(
        self,
        frame_rgb: np.ndarray,
        prev_frame_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ProcessedFrame]:
        """
        Xử lý frame và trả về state vector.

        Args:
            frame_rgb      : Frame hiện tại (H, W, 3) RGB
            prev_frame_rgb : Frame trước (để tính velocity), có thể None

        Returns:
            state  : np.ndarray shape (STATE_DIM,) — normalized [-1, 1]
            result : ProcessedFrame để debug/visualize
        """
        result = self._processor.process(frame_rgb)

        dx = self._compute_dx(result)
        dy = self._compute_dy(result)
        velocity = self._compute_velocity(result, prev_frame_rgb, frame_rgb)

        state = np.array([dx, dy, velocity], dtype=np.float32)
        return state, result

    def reset(self):
        """Reset tracking state giữa các episodes."""
        self._processor.reset()
        self._prev_bird_y = None
        self._smoothed_velocity = 0.0

    # ──────────────────────────────────────────────
    # PRIVATE: Feature computation
    # ──────────────────────────────────────────────

    def _compute_dx(self, result: ProcessedFrame) -> float:
        """
        dx = khoảng cách ngang từ chim đến pipe tiếp theo.
        Normalize: 0 = ngay trước pipe, 1 = xa nhất (bên phải màn hình).
        Negative: đã qua pipe.
        """
        if not result.has_bird:
            return 1.0  # Không thấy chim → giả sử xa

        bird_cx = result.bird.center_x

        pair = result.get_next_pipe_pair(bird_cx)
        if pair is None:
            return 1.0  # Không thấy pipe → xa nhất

        top_pipe, _ = pair
        # dx = khoảng cách từ chim đến mép trái pipe
        pipe_x = top_pipe.x
        raw_dx = pipe_x - bird_cx

        # Normalize về [-1, 1]
        # raw_dx: -frame_w (đã qua) → frame_w (phía trước xa)
        normalized = raw_dx / self.frame_w
        return float(np.clip(normalized, -1.0, 1.0))

    def _compute_dy(self, result: ProcessedFrame) -> float:
        """
        dy = khoảng cách dọc từ chim đến tâm khe hở (gap center).
        Normalize: 0 = đang ở tâm gap, 1 = ở đáy gap, -1 = ở đỉnh gap.
        Positive dy: chim ở dưới tâm (cần bay lên), negative: ở trên (không cần flap).
        """
        if not result.has_bird:
            return 0.0

        bird_cy = result.bird.center_y
        pair = result.get_next_pipe_pair(result.bird.center_x)

        if pair is None:
            # Không thấy pipe: giả sử gap ở giữa màn hình
            gap_center = self.frame_h / 2.0
        else:
            top_pipe, bot_pipe = pair
            # Gap: từ mép dưới top pipe đến mép trên bottom pipe
            gap_top = top_pipe.y + top_pipe.h      # Đỉnh gap
            gap_bot = bot_pipe.y                    # Đáy gap
            gap_center = (gap_top + gap_bot) / 2.0

        # dy = chim_y - gap_center (dương = chim ở dưới tâm)
        raw_dy = bird_cy - gap_center

        # Normalize bởi frame_h
        normalized = raw_dy / self.frame_h
        return float(np.clip(normalized, -1.0, 1.0))

    def _compute_velocity(
        self,
        result: ProcessedFrame,
        prev_frame: Optional[np.ndarray],
        curr_frame: np.ndarray,
    ) -> float:
        """
        Velocity ước tính bằng 2 phương pháp:
          1. Thay đổi vị trí Y của chim qua 2 frames (ưu tiên)
          2. Motion estimation của ImageProcessor (fallback)
        """
        # Phương pháp 1: Bird position tracking
        if result.has_bird:
            curr_y = result.bird.center_y

            if self._prev_bird_y is not None:
                raw_vel = curr_y - self._prev_bird_y  # pixels/frame (dương = đi xuống)
                # Normalize: max velocity ~10 pixels/frame
                normalized_vel = raw_vel / 15.0
                normalized_vel = float(np.clip(normalized_vel, -1.0, 1.0))

                # EMA smoothing
                self._smoothed_velocity = (
                    self._velocity_alpha * normalized_vel
                    + (1 - self._velocity_alpha) * self._smoothed_velocity
                )
            self._prev_bird_y = curr_y
            return self._smoothed_velocity

        # Phương pháp 2: Frame differencing
        if prev_frame is not None:
            self._processor._prev_gray = None  # Force re-compute
            vel = self._processor.compute_velocity(curr_frame)
            return vel

        return self._smoothed_velocity

    @property
    def processor(self) -> ImageProcessor:
        return self._processor

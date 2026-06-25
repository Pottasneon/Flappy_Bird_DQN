"""
src/vision/image_processor.py  — IMP302 Core

Xử lý frame RGB từ Flappy Bird bằng OpenCV để:
  1. Phát hiện vị trí con chim (bird position)
  2. Phát hiện vị trí các pipe (pipe position)
  3. Tính velocity của chim (motion estimation)

Kỹ thuật được dùng:
  - Grayscale conversion      : giảm chiều dữ liệu
  - Gaussian Blur             : giảm nhiễu
  - OTSU Thresholding         : tách vật thể khỏi nền
  - Color Masking (HSV)       : phân tách chim vàng / pipe xanh
  - Contour Detection         : tìm object
  - Bounding Box              : lấy tọa độ (x, y, w, h)
  - Frame Differencing        : motion estimation → velocity
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict


class BirdDetection:
    """Kết quả detect chim."""

    def __init__(self, x: int, y: int, w: int, h: int, confidence: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def __repr__(self):
        return f"BirdDetection(cx={self.center_x:.1f}, cy={self.center_y:.1f}, conf={self.confidence:.2f})"


class PipeDetection:
    """Kết quả detect một pipe (top hoặc bottom)."""

    def __init__(self, x: int, y: int, w: int, h: int, is_top: bool):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_top = is_top

    @property
    def right_edge(self) -> int:
        return self.x + self.w

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def bottom_y(self) -> int:
        """Mép dưới của pipe trên (= đỉnh của gap)."""
        return self.y + self.h if self.is_top else self.y

    def __repr__(self):
        kind = "TOP" if self.is_top else "BOT"
        return f"Pipe[{kind}](x={self.x}, y={self.y}, w={self.w}, h={self.h})"


class ProcessedFrame:
    """Tổng hợp kết quả xử lý một frame."""

    def __init__(
        self,
        bird: Optional[BirdDetection],
        top_pipes: List[PipeDetection],
        bot_pipes: List[PipeDetection],
        gray: np.ndarray,
        mask_bird: np.ndarray,
        mask_pipe: np.ndarray,
    ):
        self.bird = bird
        self.top_pipes = top_pipes
        self.bot_pipes = bot_pipes
        self.gray = gray
        self.mask_bird = mask_bird
        self.mask_pipe = mask_pipe

    @property
    def has_bird(self) -> bool:
        return self.bird is not None

    @property
    def has_pipes(self) -> bool:
        return len(self.top_pipes) > 0 or len(self.bot_pipes) > 0

    def get_next_pipe_pair(self, bird_cx: float) -> Optional[Tuple[PipeDetection, PipeDetection]]:
        """
        Lấy cặp pipe (top, bottom) gần nhất phía trước chim.
        Trả về None nếu không tìm thấy.
        """
        # Ghép top-bot theo x tương đồng
        paired = []
        for top in self.top_pipes:
            for bot in self.bot_pipes:
                if abs(top.center_x - bot.center_x) < 30:
                    paired.append((top, bot))

        # Lọc các pipe ở phía trước chim
        ahead = [(t, b) for t, b in paired if t.right_edge > bird_cx - 10]

        if not ahead:
            return None

        # Chọn cặp gần nhất
        ahead.sort(key=lambda p: p[0].center_x)
        return ahead[0]


class ImageProcessor:
    """
    IMP302 — Xử lý ảnh Flappy Bird.

    Pipeline:
        RGB frame → Grayscale → Blur → Threshold
                 ↘ HSV → ColorMask (bird/pipe) → Contours → BoundingBox
    """

    # ------- Màu sắc trong FlappyBird-v0 (HSV ranges) -------
    # Chim: vàng/cam
    BIRD_LOWER_HSV = np.array([15,  80, 100], dtype=np.uint8)
    BIRD_UPPER_HSV = np.array([40, 255, 255], dtype=np.uint8)

    # Pipe: xanh lá
    PIPE_LOWER_HSV = np.array([35,  40,  40], dtype=np.uint8)
    PIPE_UPPER_HSV = np.array([90, 255, 255], dtype=np.uint8)

    # Kích thước tối thiểu để accept một detection (pixels)
    BIRD_MIN_AREA = 80
    PIPE_MIN_AREA = 200

    def __init__(self, frame_h: int = 512, frame_w: int = 288):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self._prev_gray: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray) -> ProcessedFrame:
        """
        Xử lý một frame RGB, trả về ProcessedFrame.

        Args:
            frame_rgb: ndarray shape (H, W, 3), dtype uint8, không gian màu RGB
        Returns:
            ProcessedFrame chứa bird, top_pipes, bot_pipes và các mask debug
        """
        # 1. Chuyển sang BGR (OpenCV native) và HSV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 2. Grayscale + Blur (IMP302: Gaussian Blur)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. OTSU Thresholding (IMP302: Thresholding)
        _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Color masks (IMP302: phân tách theo màu)
        mask_bird = cv2.inRange(frame_hsv, self.BIRD_LOWER_HSV, self.BIRD_UPPER_HSV)
        mask_pipe = cv2.inRange(frame_hsv, self.PIPE_LOWER_HSV, self.PIPE_UPPER_HSV)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_bird = cv2.morphologyEx(mask_bird, cv2.MORPH_CLOSE, kernel)
        mask_pipe = cv2.morphologyEx(mask_pipe, cv2.MORPH_CLOSE, kernel)

        # 5. Detect bird (IMP302: Contours + Bounding Box)
        bird = self._detect_bird(mask_bird)

        # 6. Detect pipes
        top_pipes, bot_pipes = self._detect_pipes(mask_pipe)

        # 7. Cập nhật prev gray cho motion estimation
        self._prev_gray = gray.copy()

        return ProcessedFrame(
            bird=bird,
            top_pipes=top_pipes,
            bot_pipes=bot_pipes,
            gray=gray,
            mask_bird=mask_bird,
            mask_pipe=mask_pipe,
        )

    def preprocess_for_display(self, frame_rgb: np.ndarray, result: ProcessedFrame) -> np.ndarray:
        """
        Vẽ bounding boxes lên frame để debug / visualization.
        Returns: annotated frame (BGR)
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()

        # Bird
        if result.bird:
            x, y, w, h = result.bird.bbox
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cx, cy = int(result.bird.center_x), int(result.bird.center_y)
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 0, 255), -1)

        # Top pipes
        for pipe in result.top_pipes:
            cv2.rectangle(frame_bgr, (pipe.x, pipe.y),
                          (pipe.x + pipe.w, pipe.y + pipe.h), (0, 200, 0), 2)

        # Bottom pipes
        for pipe in result.bot_pipes:
            cv2.rectangle(frame_bgr, (pipe.x, pipe.y),
                          (pipe.x + pipe.w, pipe.y + pipe.h), (0, 100, 255), 2)

        return frame_bgr

    def reset(self):
        """Reset state giữa các episodes."""
        self._prev_gray = None

    # ──────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────

    def _detect_bird(self, mask: np.ndarray) -> Optional[BirdDetection]:
        """
        IMP302: Contour detection trên bird mask.
        Chọn contour lớn nhất ở phần 1/3 trái màn hình (chim luôn ở bên trái).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Lọc theo area tối thiểu
        valid = [c for c in contours if cv2.contourArea(c) >= self.BIRD_MIN_AREA]

        if not valid:
            return None

        # Chọn contour lớn nhất
        best = max(valid, key=cv2.contourArea)
        area = cv2.contourArea(best)
        x, y, w, h = cv2.boundingRect(best)

        # Confidence = area / expected area
        confidence = min(1.0, area / 500.0)

        return BirdDetection(x, y, w, h, confidence)

    def _detect_pipes(
        self, mask: np.ndarray
    ) -> Tuple[List[PipeDetection], List[PipeDetection]]:
        """
        IMP302: Phát hiện pipes.
        Pipe trên = bbox nằm phần trên màn hình (bottom_y < frame_h/2)
        Pipe dưới = bbox nằm phần dưới màn hình (y > frame_h/2)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        top_pipes: List[PipeDetection] = []
        bot_pipes: List[PipeDetection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.PIPE_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Pipe cần có width đủ lớn (loại bỏ noise nhỏ)
            if w < 15:
                continue

            mid_y = y + h / 2

            if mid_y < self.frame_h / 2:
                # Top pipe: bắt đầu từ trên xuống
                top_pipes.append(PipeDetection(x, y, w, h, is_top=True))
            else:
                # Bottom pipe: bắt đầu từ dưới lên
                bot_pipes.append(PipeDetection(x, y, w, h, is_top=False))

        return top_pipes, bot_pipes

    def compute_velocity(self, frame_rgb: np.ndarray) -> float:
        """
        IMP302: Motion Estimation bằng Frame Differencing.
        Ước tính velocity của chim theo trục Y.
        
        Returns:
            velocity: dương = đi xuống, âm = đi lên. Range [-1, 1] (normalized)
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None:
            return 0.0

        # Frame difference
        diff = cv2.absdiff(gray, self._prev_gray).astype(np.float32)

        # Chỉ xét vùng bên trái màn hình (nơi chim bay)
        bird_region = diff[:, : self.frame_w // 3]

        if bird_region.max() == 0:
            return 0.0

        # Tìm trọng tâm chuyển động theo Y
        rows = np.sum(bird_region, axis=1)  # (H,)
        total = rows.sum()
        if total == 0:
            return 0.0

        indices = np.arange(len(rows), dtype=np.float32)
        centroid_y = float(np.dot(rows, indices) / total)

        # Normalize về [-1, 1]
        # centroid_y: 0 (top) → frame_h (bottom)
        velocity = (centroid_y / self.frame_h) * 2 - 1
        return float(np.clip(velocity, -1.0, 1.0))

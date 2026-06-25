"""
src/environment/flappy_env.py

Wrapper quanh flappy-bird-gymnasium.
Hỗ trợ 2 chế độ:
  - use_obs=True  (mặc định): Dùng observation vector (12 dims) từ gym — chính xác, ổn định
  - use_obs=False            : Dùng frame RGB (để CV module xử lý)
"""

import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from typing import Tuple, Optional


class FlappyBirdEnv:
    """
    Wrapper quanh FlappyBird-v0.

    Khi use_obs=True (mặc định): reset() và step() trả về obs vector (12,)
    thay vì frame RGB. Đây là chế độ học nhanh nhất.

    Observation vector (12 dims) từ flappy-bird-gymnasium:
      [0]  last_pipe_horizontal_dist   (normalized)
      [1]  last_top_pipe_vert_dist     (normalized)
      [2]  last_bottom_pipe_vert_dist  (normalized)
      [3]  next_pipe_horizontal_dist   (normalized)  ← QUAN TRỌNG NHẤT
      [4]  next_top_pipe_vert_dist     (normalized)  ← QUAN TRỌNG NHẤT
      [5]  next_bottom_pipe_vert_dist  (normalized)  ← QUAN TRỌNG NHẤT
      [6]  next_next_pipe_horizontal_dist
      [7]  next_next_top_pipe_vert_dist
      [8]  next_next_bottom_pipe_vert_dist
      [9]  player_vel                  (normalized)  ← QUAN TRỌNG
      [10] player_rot                  (normalized)
      [11] score                       (normalized)
    """

    FRAME_WIDTH = 288
    FRAME_HEIGHT = 512
    OBS_DIM = 12

    def __init__(
        self,
        render_mode: str = "rgb_array",
        use_obs: bool = True,
    ):
        """
        Args:
            render_mode: "rgb_array" hoặc "human"
            use_obs    : True = dùng obs vector 12 chiều (khuyến nghị)
                         False = dùng frame RGB (cho Computer Vision)
        """
        self.render_mode = render_mode
        self.use_obs = use_obs

        self._env = gym.make(
            "FlappyBird-v0",
            render_mode=render_mode,
            use_lidar=False,   # 12-dim obs vector (lidar=True → 180 dims)
        )

        self._current_obs: Optional[np.ndarray] = None
        self._current_frame: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._score = 0
        self._steps = 0
        self._pipes_passed = 0

    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset môi trường.
        Returns:
            obs or frame: tùy use_obs
            info: dict
        """
        obs, info = self._env.reset()
        self._score = 0
        self._steps = 0
        self._pipes_passed = 0
        self._current_obs = obs.astype(np.float32)

        self._prev_frame = None
        if self.render_mode == "rgb_array":
            self._current_frame = self._get_frame()

        if self.use_obs:
            return self._current_obs, info
        return self._current_frame, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Thực hiện action.
        Returns:
            obs or frame, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        self._current_obs = obs.astype(np.float32)

        if self.render_mode == "rgb_array":
            self._prev_frame = self._current_frame
            self._current_frame = self._get_frame()

        # Track pipes passed
        if "score" in info:
            new_score = info["score"]
            if new_score > self._pipes_passed:
                self._pipes_passed = new_score

        augmented_info = {
            **info,
            "steps": self._steps,
            "pipes_passed": self._pipes_passed,
        }

        if self.use_obs:
            return self._current_obs, reward, terminated, truncated, augmented_info
        return self._current_frame, reward, terminated, truncated, augmented_info

    def _get_frame(self) -> np.ndarray:
        if self.render_mode == "rgb_array":
            frame = self._env.render()
            if frame is None:
                return np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)
            return frame.astype(np.uint8)
        return np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)

    @property
    def current_obs(self) -> Optional[np.ndarray]:
        return self._current_obs

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        return self._current_frame

    @property
    def prev_frame(self) -> Optional[np.ndarray]:
        return self._prev_frame

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def close(self):
        self._env.close()

    def __repr__(self):
        return (
            f"FlappyBirdEnv(render_mode={self.render_mode}, "
            f"use_obs={self.use_obs}, "
            f"steps={self._steps}, pipes={self._pipes_passed})"
        )

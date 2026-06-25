"""
src/training/rewards.py — Reward Engineering

Reward function tối ưu cho Flappy Bird DQN.

Thiết kế:
  - Khi dùng use_obs=True: reward từ gym đã đủ tốt (+0.1/step, +1 qua pipe, -1 chết)
    → bổ sung thêm pipe bonus lớn + penalty vị trí để học nhanh hơn
  - Mục tiêu: agent học SỐNG LÂU và đi qua pipe, không chỉ tránh chết
"""

import numpy as np
from typing import Optional


class RewardShaper:
    """
    Reward Engineering tối ưu cho DQN.

    Khi use_obs=True, state = obs vector 12 chiều.
    Thành phần reward:
      1. Gym reward passthrough     : giữ nguyên reward từ env (+0.1/step, -1 chết)
      2. Pipe bonus (lớn hơn gym)   : +5.0 thêm khi qua pipe mới
      3. Death penalty (bổ sung)    : -5.0 thêm khi chết
      4. Position bonus             : +0.1 khi gần tâm gap (dùng obs dims 4,5)
    """

    PIPE_BONUS: float = 5.0      # Thêm vào reward gym khi qua pipe
    DEATH_BONUS: float = -5.0   # Thêm vào reward gym khi chết
    ALIVE_BONUS: float = 0.05   # Thêm mỗi step sống
    CENTER_BONUS: float = 0.1   # Bonus khi ở gần tâm gap

    def __init__(self, use_obs: bool = True):
        """
        Args:
            use_obs: True nếu dùng obs vector (12 dims), False nếu dùng CV features (3 dims)
        """
        self.use_obs = use_obs
        self._prev_pipes_passed: int = 0
        self._prev_dy: float = 0.0

    def compute(
        self,
        env_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        state: Optional[np.ndarray] = None,
    ) -> float:
        """
        Tính tổng reward cho một step.

        Args:
            env_reward : Reward trả về trực tiếp từ gym env
            terminated : Agent chết
            truncated  : Hết thời gian
            info       : dict từ env (chứa 'pipes_passed', 'score')
            state      : State vector hiện tại

        Returns:
            reward: float
        """
        reward = env_reward  # Bắt đầu từ gym reward

        if terminated:
            reward += self.DEATH_BONUS
            self._reset()
            return reward

        # Alive bonus nhỏ để khuyến khích sống lâu
        reward += self.ALIVE_BONUS

        # Pipe bonus: thêm vào khi qua pipe mới
        pipes_passed = info.get("pipes_passed", info.get("score", 0))
        if pipes_passed > self._prev_pipes_passed:
            pipes_new = pipes_passed - self._prev_pipes_passed
            reward += self.PIPE_BONUS * pipes_new
            self._prev_pipes_passed = pipes_passed

        # Position bonus (dùng obs vector nếu có)
        if state is not None and self.use_obs and len(state) >= 6:
            # dims [4],[5]: next_top_pipe_vert_dist và next_bottom_pipe_vert_dist
            # Khi ở tâm gap: cả 2 giá trị gần bằng nhau
            top_dist = float(state[4])
            bot_dist = float(state[5])
            # Chênh lệch nhỏ = đang ở gần tâm gap
            gap_balance = abs(top_dist - bot_dist)
            if gap_balance < 0.15:
                reward += self.CENTER_BONUS

        elif state is not None and not self.use_obs and len(state) >= 2:
            # CV mode: dùng dy (state[1])
            dy = float(state[1])
            if abs(dy) < 0.2:
                reward += self.CENTER_BONUS
            reward -= 0.03 * abs(dy)

        return reward

    def reset(self):
        self._reset()

    def _reset(self):
        self._prev_pipes_passed = 0
        self._prev_dy = 0.0

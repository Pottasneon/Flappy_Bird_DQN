"""
src/agents/replay_buffer.py — Experience Replay Buffer

Lưu trữ transitions (s, a, r, s', done) để training DQN.

Tại sao cần Replay Buffer?
  1. Phá vỡ correlation giữa các samples liên tiếp
  2. Tăng data efficiency (dùng lại experiences nhiều lần)
  3. Ổn định training cho neural network

Cấu trúc: collections.deque với maxlen → auto-evict transitions cũ nhất.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch chưa được cài đặt. Chạy: pip install torch"
    )


class ReplayBuffer:
    """
    Experience Replay Buffer cho DQN.

    Mỗi transition: (state, action, reward, next_state, done)

    Usage:
        buffer = ReplayBuffer(capacity=50000)
        buffer.push(state, action, reward, next_state, done)
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(64)
    """

    def __init__(self, capacity: int = 50000):
        """
        Args:
            capacity: Số lượng transition tối đa lưu trữ.
                      Khi đầy, transition cũ nhất sẽ bị xóa.
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Thêm một transition vào buffer.

        Args:
            state      : State hiện tại (np.ndarray)
            action     : Action đã thực hiện (int)
            reward     : Reward nhận được (float)
            next_state : State tiếp theo (np.ndarray)
            done       : Episode kết thúc? (bool)
        """
        self._buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random sample mini-batch từ buffer.

        Args:
            batch_size : Số lượng samples
            device     : Device cho tensors ("cpu" hoặc "cuda")

        Returns:
            Tuple of tensors:
                states     : (batch_size, state_dim)
                actions    : (batch_size, 1)    — LongTensor
                rewards    : (batch_size, 1)
                next_states: (batch_size, state_dim)
                dones      : (batch_size, 1)
        """
        batch = random.sample(self._buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).unsqueeze(1).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self):
        return f"ReplayBuffer(size={len(self._buffer)}/{self.capacity})"

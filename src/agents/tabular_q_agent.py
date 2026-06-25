"""
src/agents/tabular_q_agent.py

Tabular Q-Learning Agent.
Sử dụng phương pháp rời rạc hóa (discretization) để chuyển continuous state thành discrete state.
"""

import numpy as np
import os
from typing import Optional, Tuple


class TabularQAgent:
    """
    Q-Learning với Q-Table.
    Rất hiệu quả cho không gian state nhỏ.

    State: [dx, dy, velocity] normalized trong khoảng [-1, 1]
    """

    N_ACTIONS = 2

    # Số lượng bins cho mỗi chiều
    BINS_DX = 15
    BINS_DY = 25
    BINS_VEL = 10

    def __init__(
        self,
        learning_rate: float = 0.1,    # Tabular Q có thể dùng LR lớn hơn Linear
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Khởi tạo Q-Table: shape (BINS_DX, BINS_DY, BINS_VEL, 2)
        self.q_table = np.zeros(
            (self.BINS_DX, self.BINS_DY, self.BINS_VEL, self.N_ACTIONS),
            dtype=np.float64
        )

        # Pre-compute bin edges
        # dx: [-1, 1] -> tập trung vào [0, 1] vì phần âm là đã qua ống
        self.bins_dx = np.linspace(-1.0, 1.0, self.BINS_DX - 1)
        # dy: [-1, 1] -> cần phân giải cao ở gần 0 (tâm ống)
        self.bins_dy = np.linspace(-1.0, 1.0, self.BINS_DY - 1)
        # vel: [-1, 1]
        self.bins_vel = np.linspace(-1.0, 1.0, self.BINS_VEL - 1)

        self.total_updates = 0
        self.episode_count = 0

    def discretize_state(self, state: np.ndarray) -> Tuple[int, int, int]:
        """Chuyển continuous vector [dx, dy, vel] thành tuple (i, j, k) indices."""
        dx, dy, vel = state

        # Dùng np.digitize để tìm index của bin
        idx_dx = int(np.digitize(dx, self.bins_dx))
        idx_dy = int(np.digitize(dy, self.bins_dy))
        idx_vel = int(np.digitize(vel, self.bins_vel))

        return (idx_dx, idx_dy, idx_vel)

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Lấy giá trị Q(s, a)."""
        idx = self.discretize_state(state)
        return self.q_table[idx][action]

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-Greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.N_ACTIONS)

        idx = self.discretize_state(state)
        return int(np.argmax(self.q_table[idx]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Standard Q-Learning TD Update."""
        idx = self.discretize_state(state)
        next_idx = self.discretize_state(next_state)

        current_q = self.q_table[idx][action]

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_idx])

        td_error = td_target - current_q

        # Q(s,a) <- Q(s,a) + alpha * TD_Error
        self.q_table[idx][action] += self.lr * td_error

        self.total_updates += 1
        return float(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "epsilon": np.array([self.epsilon]),
            "episode_count": np.array([self.episode_count]),
            "total_updates": np.array([self.total_updates]),
            "lr": np.array([self.lr]),
            "gamma": np.array([self.gamma]),
        }
        np.save(path, data, allow_pickle=True)
        print(f"[Agent] Saved Q-Table → {path} (Sparsity: {(self.q_table != 0).mean()*100:.2f}%)")

    def load_weights(self, path: str):
        if not os.path.exists(path):
            print(f"[Agent] WARNING: Checkpoint không tồn tại: {path}")
            return
        data = np.load(path, allow_pickle=True).item()
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"][0])
        self.episode_count = int(data["episode_count"][0])
        self.total_updates = int(data["total_updates"][0])
        print(f"[Agent] Loaded Q-Table ← {path} (episode={self.episode_count}, ε={self.epsilon:.4f})")

    def set_eval_mode(self):
        self.epsilon = 0.0

    def __repr__(self):
        sparsity = (self.q_table != 0).mean() * 100
        return (
            f"TabularQAgent("
            f"bins={self.BINS_DX}x{self.BINS_DY}x{self.BINS_VEL}, "
            f"lr={self.lr}, ε={self.epsilon:.4f}, "
            f"visited={sparsity:.1f}%)"
        )

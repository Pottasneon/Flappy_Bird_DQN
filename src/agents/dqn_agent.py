"""
src/agents/dqn_agent.py — Deep Q-Network Agent

Nâng cấp từ Linear Q-Learning lên Deep Q-Network (DQN) sử dụng PyTorch.

Cải tiến so với Linear Q-Agent:
  1. Neural Network thay vì Linear function → xấp xỉ Q-function phi tuyến
  2. Experience Replay → phá vỡ correlation, tăng data efficiency
  3. Target Network → ổn định training
  4. Double DQN → giảm overestimation bias

Kiến trúc Q-Network:
    Input (state_dim) → FC(128) → ReLU → FC(128) → ReLU → FC(n_actions)

Công thức:
    Online:   Q(s, a; θ)
    Target:   Q(s, a; θ⁻)  (cập nhật chậm)

    Double DQN TD Target:
        a* = argmax_a Q(s', a; θ)           ← online network chọn action
        y  = r + γ · Q(s', a*; θ⁻)          ← target network đánh giá
"""

import os
import copy
import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch chưa được cài đặt. Chạy: pip install torch"
    )

from .replay_buffer import ReplayBuffer


# ──────────────────────────────────────────────
# Q-Network Architecture
# ──────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Neural Network cho Q-function approximation.

    Architecture:
        state (state_dim) → FC(128) → ReLU → FC(128) → ReLU → FC(n_actions)

    Có thể cấu hình số hidden units.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: state tensor, shape (batch_size, state_dim)
        Returns:
            Q-values for all actions, shape (batch_size, n_actions)
        """
        return self.net(x)


# ──────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network Agent với Experience Replay + Target Network.

    Tương thích interface với LinearQLearningAgent:
        - choose_action(state) → int
        - update(state, action, reward, next_state, done) → float
        - decay_epsilon()
        - save_weights(path) / load_weights(path)
        - set_eval_mode()

    Actions:
        0 = không flap (rơi)
        1 = flap (bay lên)
    """

    N_ACTIONS = 2

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        # DQN-specific
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        min_replay_size: int = 500,
        double_dqn: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            state_dim          : Số chiều state vector
            hidden_dim         : Số neurons mỗi hidden layer
            learning_rate      : Learning rate cho Adam optimizer
            gamma              : Discount factor
            epsilon            : Epsilon ban đầu (exploration)
            epsilon_min        : Epsilon tối thiểu
            epsilon_decay      : Tốc độ giảm epsilon
            buffer_capacity    : Kích thước Replay Buffer
            batch_size         : Mini-batch size khi training
            target_update_freq : Cập nhật Target Network sau N steps
            min_replay_size    : Số transitions tối thiểu trước khi bắt đầu train
            double_dqn         : Sử dụng Double DQN?
            device             : "cpu" hoặc "cuda" (auto-detect nếu None)
        """
        # Hyperparameters
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.double_dqn = double_dqn

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Networks
        self.online_net = QNetwork(state_dim, self.N_ACTIONS, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.N_ACTIONS, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network luôn ở eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Statistics
        self.total_updates = 0
        self.episode_count = 0
        self._training_steps = 0
        self._target_updates = 0

    # ──────────────────────────────────────────────
    # CORE RL METHODS (tương thích LinearQLearningAgent)
    # ──────────────────────────────────────────────

    def choose_action(self, state: np.ndarray) -> int:
        """
        Epsilon-Greedy policy.

        Args:
            state: ndarray shape (state_dim,)
        Returns:
            action: int {0, 1}
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.N_ACTIONS)

        # Exploitation: chọn action có Q-value cao nhất
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        Lưu transition vào Replay Buffer + Train từ mini-batch.

        Pipeline:
            1. Push transition vào buffer
            2. Nếu buffer đủ lớn → sample mini-batch → gradient update
            3. Cập nhật Target Network theo schedule

        Args:
            state      : Current state
            action     : Action taken
            reward     : Reward received
            next_state : Next state
            done       : Episode ended?

        Returns:
            loss: float — training loss (0.0 nếu chưa đủ data để train)
        """
        # 1. Push vào Replay Buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_updates += 1

        # 2. Chưa đủ data → skip training
        if len(self.replay_buffer) < self.min_replay_size:
            return 0.0

        # 3. Sample mini-batch + train
        loss = self._train_step()

        # 4. Cập nhật Target Network
        self._training_steps += 1
        if self._training_steps % self.target_update_freq == 0:
            self._update_target_network()

        return loss

    def decay_epsilon(self):
        """Giảm epsilon sau mỗi episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    # ──────────────────────────────────────────────
    # DQN TRAINING
    # ──────────────────────────────────────────────

    def _train_step(self) -> float:
        """
        Một bước gradient descent trên mini-batch.

        Returns:
            loss: float
        """
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Tính Q(s, a) cho actions đã chọn
        q_values = self.online_net(states)                     # (batch, n_actions)
        q_sa = q_values.gather(1, actions)                     # (batch, 1)

        # Tính TD Target
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: online chọn action, target đánh giá
                next_q_online = self.online_net(next_states)   # (batch, n_actions)
                best_actions = next_q_online.argmax(dim=1, keepdim=True)  # (batch, 1)
                next_q_target = self.target_net(next_states)   # (batch, n_actions)
                next_q_sa = next_q_target.gather(1, best_actions)  # (batch, 1)
            else:
                # Standard DQN: target network chọn + đánh giá
                next_q_target = self.target_net(next_states)   # (batch, n_actions)
                next_q_sa = next_q_target.max(dim=1, keepdim=True)[0]  # (batch, 1)

            # TD Target: r + γ·Q_target(s', a*) × (1 - done)
            td_targets = rewards + self.gamma * next_q_sa * (1.0 - dones)

        # Loss: Huber Loss (Smooth L1) — ít nhạy cảm với outliers hơn MSE
        loss = F.smooth_l1_loss(q_sa, td_targets)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        return float(loss.item())

    def _update_target_network(self):
        """Hard copy: θ⁻ ← θ"""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self._target_updates += 1

    # ──────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────

    def save_weights(self, path: str):
        """
        Lưu model weights + hyperparameters.
        Dùng format .pt (PyTorch standard).
        """
        # Ensure .pt extension
        if not path.endswith('.pt'):
            path = path.rsplit('.', 1)[0] + '.pt'

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "total_updates": self.total_updates,
            "training_steps": self._training_steps,
            "target_updates": self._target_updates,
            # Hyperparameters (để restore)
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "double_dqn": self.double_dqn,
        }
        torch.save(checkpoint, path)
        print(
            f"[DQN] Saved checkpoint → {path} "
            f"(ep={self.episode_count}, ε={self.epsilon:.4f}, "
            f"buffer={len(self.replay_buffer)})"
        )

    def load_weights(self, path: str):
        """Load model weights từ checkpoint .pt"""
        # Try .pt extension
        if not os.path.exists(path) and not path.endswith('.pt'):
            pt_path = path.rsplit('.', 1)[0] + '.pt'
            if os.path.exists(pt_path):
                path = pt_path

        if not os.path.exists(path):
            print(f"[DQN] WARNING: Checkpoint không tồn tại: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        self.total_updates = checkpoint["total_updates"]
        self._training_steps = checkpoint.get("training_steps", 0)
        self._target_updates = checkpoint.get("target_updates", 0)

        print(
            f"[DQN] Loaded checkpoint ← {path} "
            f"(ep={self.episode_count}, ε={self.epsilon:.4f})"
        )

    def set_eval_mode(self):
        """Evaluation mode: tắt exploration + set network eval."""
        self.epsilon = 0.0
        self.online_net.eval()

    def set_train_mode(self):
        """Training mode: set network train."""
        self.online_net.train()

    # ──────────────────────────────────────────────
    # Q-VALUE ACCESS (for debugging / logging)
    # ──────────────────────────────────────────────

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Lấy Q(s, a) từ online network."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return float(q_values[0, action].item())

    def get_all_q_values(self, state: np.ndarray) -> np.ndarray:
        """Trả về Q-values cho tất cả actions."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values[0].cpu().numpy()

    def __repr__(self):
        return (
            f"DQNAgent("
            f"state_dim={self.state_dim}, "
            f"hidden={self.hidden_dim}, "
            f"lr={self.lr}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, "
            f"buffer={len(self.replay_buffer)}/{self.replay_buffer.capacity}, "
            f"double={self.double_dqn}, "
            f"device={self.device})"
        )

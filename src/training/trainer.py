"""
src/training/trainer.py  — Training Loop (DQN-Optimized)

Hỗ trợ 2 loại agent:
  - "tabular": TabularQAgent (Tabular Q-Learning)
  - "dqn"    : DQNAgent (Deep Q-Network — mặc định)

Chế độ observation:
  - use_obs=True  (mặc định): Dùng obs vector 12 chiều từ gym — nhanh & chính xác
  - use_obs=False            : Dùng frame RGB qua CV pipeline

Cấu hình DQN tối ưu cho target score >= 5:
  - state_dim = 12 (toàn bộ gym obs, thay vì 3 dims CV)
  - epsilon_decay = 0.998 (decay chậm hơn, exploration đủ lâu)
  - batch_size = 128
  - target_update_freq = 500
  - hidden_dim = 256
"""

import os
import time
import numpy as np
from typing import Optional
from tqdm import tqdm

from ..environment.flappy_env import FlappyBirdEnv
from ..vision.feature_extractor import FeatureExtractor
from ..agents.tabular_q_agent import TabularQAgent
from ..agents.dqn_agent import DQNAgent
from ..training.rewards import RewardShaper
from ..utils.logger import TrainingLogger


AGENT_EXTENSIONS = {
    "tabular": ".npy",
    "dqn": ".pt",
}


class Trainer:
    """
    Training loop cho Flappy Bird DQN.

    Mặc định dùng chế độ obs vector (use_obs=True) cho học nhanh nhất.
    """

    def __init__(
        self,
        n_episodes: int = 2000,
        max_steps_per_episode: int = 5000,
        save_every: int = 100,
        checkpoint_dir: str = "checkpoints",
        dataset_dir: str = "dataset",
        save_frames: bool = False,
        render: bool = False,
        # Agent selection
        agent_type: str = "dqn",
        # Obs mode
        use_obs: bool = True,
        # Agent hyperparams (shared)
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        # DQN-specific hyperparams
        hidden_dim: int = 256,
        batch_size: int = 128,
        buffer_size: int = 50000,
        target_update_freq: int = 500,
        double_dqn: bool = True,
        min_replay_size: int = 1000,
    ):
        self.n_episodes = n_episodes
        self.max_steps = max_steps_per_episode
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.save_frames = save_frames
        self.agent_type = agent_type.lower()
        self.use_obs = use_obs

        # Render mode
        render_mode = "human" if render else "rgb_array"
        self.env = FlappyBirdEnv(render_mode=render_mode, use_obs=use_obs)

        # State dimension
        if use_obs:
            state_dim = FlappyBirdEnv.OBS_DIM  # 12
        else:
            state_dim = FeatureExtractor.STATE_DIM  # 3

        # Feature extractor (chỉ dùng khi use_obs=False)
        self.extractor = FeatureExtractor() if not use_obs else None

        # Khởi tạo agent
        if self.agent_type == "dqn":
            dqn_lr = learning_rate if learning_rate != 0.01 else 3e-4
            self.agent = DQNAgent(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                learning_rate=dqn_lr,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                buffer_capacity=buffer_size,
                batch_size=batch_size,
                target_update_freq=target_update_freq,
                double_dqn=double_dqn,
                min_replay_size=min_replay_size,
            )
        elif self.agent_type == "tabular":
            self.agent = TabularQAgent(
                learning_rate=learning_rate if learning_rate != 0.01 else 0.1,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
            )
        else:
            raise ValueError(f"Agent type không hợp lệ: {self.agent_type}. Chọn: dqn, tabular")

        self.reward_shaper = RewardShaper(use_obs=use_obs)
        self.logger = TrainingLogger(log_dir="logs")

        self._ext = AGENT_EXTENSIONS.get(self.agent_type, ".npy")

        # Tracking
        self._best_score = -np.inf
        self._episode_rewards = []
        self._episode_scores = []
        self._episode_lengths = []
        self._td_errors = []
        self._epsilons = []

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "states"), exist_ok=True)

    def load_checkpoint(self, path: Optional[str] = None):
        """Load checkpoint để tiếp tục training."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"weights{self._ext}")
        self.agent.load_weights(path)

    def train(self) -> dict:
        """Chạy training loop."""
        print(f"\n{'='*65}")
        print(f"  Flappy Bird DQN — Training")
        print(f"{'='*65}")
        print(f"  Agent    : {self.agent_type.upper()}")
        print(f"  Obs mode : {'gym obs (12 dims)' if self.use_obs else 'CV frame (3 dims)'}")
        print(f"  Episodes : {self.n_episodes}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Agent    : {self.agent}")
        if self.agent_type == "dqn":
            print(f"  Buffer   : {self.agent.replay_buffer}")
            print(f"  Double   : {self.agent.double_dqn}")
            print(f"  Device   : {self.agent.device}")
        print(f"{'='*65}\n")

        start_time = time.time()

        for episode in tqdm(range(1, self.n_episodes + 1), desc="Training", unit="ep"):
            episode_reward, episode_score, episode_length, avg_td = self._run_episode(episode)

            self.agent.decay_epsilon()

            self._episode_rewards.append(episode_reward)
            self._episode_scores.append(episode_score)
            self._episode_lengths.append(episode_length)
            self._td_errors.append(avg_td)
            self._epsilons.append(self.agent.epsilon)

            self.logger.log_episode(
                episode=episode,
                reward=episode_reward,
                score=episode_score,
                length=episode_length,
                epsilon=self.agent.epsilon,
                td_error=avg_td,
            )

            if episode_score > self._best_score:
                self._best_score = episode_score
                self.agent.save_weights(
                    os.path.join(self.checkpoint_dir, f"best_model{self._ext}")
                )

            if episode % self.save_every == 0:
                self.agent.save_weights(
                    os.path.join(self.checkpoint_dir, f"weights{self._ext}")
                )
                self._print_summary(episode, start_time)

        self.agent.save_weights(os.path.join(self.checkpoint_dir, f"weights{self._ext}"))
        self.env.close()

        elapsed = time.time() - start_time
        print(f"\n{'='*65}")
        print(f"  Training Done! Thời gian: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Best score: {self._best_score}")
        print(f"{'='*65}\n")

        return self._get_history()

    # ──────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────

    def _run_episode(self, episode_num: int):
        """Chạy một episode đầy đủ."""
        frame_or_obs, info = self.env.reset()

        if self.use_obs:
            state = frame_or_obs  # obs vector (12,)
        else:
            self.extractor.reset()
            state, _ = self.extractor.extract(frame_or_obs, None)

        self.reward_shaper.reset()

        done = False
        episode_reward = 0.0
        td_errors = []
        step = 0
        prev_frame = None

        while not done and step < self.max_steps:
            action = self.agent.choose_action(state)

            next_frame_or_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if self.use_obs:
                next_state = next_frame_or_obs
            else:
                next_state, _ = self.extractor.extract(next_frame_or_obs, frame_or_obs)

            # Reward shaping (truyền cả env_reward)
            reward = self.reward_shaper.compute(
                env_reward=env_reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                state=next_state,
            )

            td_error = self.agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            td_errors.append(td_error)

            if self.save_frames and not self.use_obs and step % 10 == 0:
                self._save_frame(next_frame_or_obs, episode_num, step)

            episode_reward += reward
            state = next_state
            frame_or_obs = next_frame_or_obs
            step += 1

        pipes_passed = info.get("pipes_passed", info.get("score", 0))
        avg_td = float(np.mean(td_errors)) if td_errors else 0.0
        return episode_reward, pipes_passed, step, avg_td

    def _save_frame(self, frame: np.ndarray, episode: int, step: int):
        import cv2
        path = os.path.join(self.dataset_dir, "frames", f"ep{episode:04d}_step{step:05d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _print_summary(self, episode: int, start_time: float):
        recent_n = min(self.save_every, len(self._episode_rewards))
        recent = self._episode_rewards[-recent_n:]
        recent_scores = self._episode_scores[-recent_n:]
        elapsed = time.time() - start_time
        eps_per_sec = episode / elapsed

        summary = (
            f"\n[Ep {episode:5d}/{self.n_episodes}] "
            f"Avg Reward: {np.mean(recent):+7.2f} | "
            f"Avg Score: {np.mean(recent_scores):.2f} | "
            f"Max Score: {int(max(recent_scores, default=0))} | "
            f"Best: {int(self._best_score)} | "
            f"ε: {self.agent.epsilon:.4f} | "
            f"Speed: {eps_per_sec:.1f} ep/s"
        )
        if self.agent_type == "dqn":
            summary += f" | Buffer: {len(self.agent.replay_buffer)}"
        print(summary)

    def _get_history(self) -> dict:
        return {
            "rewards": self._episode_rewards,
            "scores": self._episode_scores,
            "lengths": self._episode_lengths,
            "td_errors": self._td_errors,
            "epsilons": self._epsilons,
        }

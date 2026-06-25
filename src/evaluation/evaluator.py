"""
src/evaluation/evaluator.py

Evaluate agent đã train:
  - epsilon = 0 (không random)
  - Log score, survival time
  - Optional: record video

Hỗ trợ cả 2 loại agent: tabular, dqn
"""

import os
import numpy as np
from typing import Optional, List


from ..environment.flappy_env import FlappyBirdEnv
from ..vision.feature_extractor import FeatureExtractor
from ..agents.tabular_q_agent import TabularQAgent
from ..agents.dqn_agent import DQNAgent
from ..training.rewards import RewardShaper


# Agent type → default checkpoint extension
AGENT_EXTENSIONS = {
    "tabular": ".npy",
    "dqn": ".pt",
}


class Evaluator:
    """Chạy agent đã train, không exploration."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        render: bool = True,
        record_video: bool = False,
        video_dir: str = "videos",
        agent_type: str = "dqn",
        use_obs: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.render = render
        self.record_video = record_video
        self.video_dir = video_dir
        self.agent_type = agent_type.lower()
        self.use_obs = use_obs

        render_mode = "human" if render else "rgb_array"
        self.env = FlappyBirdEnv(render_mode=render_mode, use_obs=use_obs)
        self.extractor = FeatureExtractor() if not use_obs else None

        state_dim = FlappyBirdEnv.OBS_DIM if use_obs else FeatureExtractor.STATE_DIM

        # Khởi tạo agent theo loại
        if self.agent_type == "dqn":
            self.agent = DQNAgent(state_dim=state_dim)
        elif self.agent_type == "tabular":
            self.agent = TabularQAgent()
        else:
            raise ValueError(f"Agent type không hợp lệ: {self.agent_type}")

        self.reward_shaper = RewardShaper(use_obs=use_obs)

        if record_video:
            os.makedirs(video_dir, exist_ok=True)

    def load(self):
        """Load weights từ checkpoint."""
        self.agent.load_weights(self.checkpoint_path)
        self.agent.set_eval_mode()
        print(f"[Eval] Loaded: {self.checkpoint_path} ({self.agent_type.upper()}), ε=0 (pure exploitation)")

    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Chạy n episodes đánh giá.

        Returns:
            dict với mean_score, max_score, mean_survival, scores
        """
        scores: List[int] = []
        rewards: List[float] = []
        survivals: List[int] = []

        for ep in range(1, n_episodes + 1):
            score, total_reward, steps, frames = self._run_episode()
            scores.append(score)
            rewards.append(total_reward)
            survivals.append(steps)

            print(
                f"[Eval] Ep {ep:3d} | Score: {score:4d} | "
                f"Reward: {total_reward:+7.2f} | Steps: {steps:5d}"
            )

            # Record video nếu cần
            if self.record_video and frames:
                self._save_video(frames, ep)

        results = {
            "scores": scores,
            "rewards": rewards,
            "survivals": survivals,
            "mean_score": float(np.mean(scores)),
            "max_score": int(np.max(scores)),
            "mean_survival": float(np.mean(survivals)),
            "mean_reward": float(np.mean(rewards)),
        }

        print(f"\n{'='*50}")
        print(f"  Evaluation Results ({n_episodes} episodes)")
        print(f"{'='*50}")
        print(f"  Agent Type   : {self.agent_type.upper()}")
        print(f"  Mean Score   : {results['mean_score']:.2f}")
        print(f"  Max Score    : {results['max_score']}")
        print(f"  Mean Survival: {results['mean_survival']:.1f} steps")
        print(f"  Mean Reward  : {results['mean_reward']:.2f}")
        print(f"{'='*50}\n")

        return results

    # ──────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────

    def _run_episode(self):
        """Một episode evaluation."""
        obs_or_frame, info = self.env.reset()
        self.reward_shaper.reset()

        if self.use_obs:
            state = obs_or_frame
            frame = None
        else:
            self.extractor.reset()
            state, _ = self.extractor.extract(obs_or_frame, None)
            frame = obs_or_frame

        done = False
        total_reward = 0.0
        step = 0
        all_frames = []
        if self.record_video and frame is not None:
            all_frames.append(frame)

        while not done and step < 10_000:
            action = self.agent.choose_action(state)
            next_obs_or_frame, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if self.use_obs:
                next_state = next_obs_or_frame
            else:
                next_state, _ = self.extractor.extract(next_obs_or_frame, frame)
                frame = next_obs_or_frame

            reward = self.reward_shaper.compute(
                env_reward=env_reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                state=next_state,
            )

            total_reward += reward
            state = next_state
            step += 1

            if self.record_video and not self.use_obs and frame is not None:
                all_frames.append(frame)

        score = info.get("pipes_passed", info.get("score", 0))
        return score, total_reward, step, all_frames

    def _save_video(self, frames: list, episode: int):
        """Lưu video episode dưới dạng mp4."""
        try:
            import cv2
            if not frames:
                return

            h, w = frames[0].shape[:2]
            path = os.path.join(self.video_dir, f"episode_{episode:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, 30, (w, h))

            for frame_rgb in frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            print(f"[Eval] Saved video → {path} ({len(frames)} frames)")
        except Exception as e:
            print(f"[Eval] Video save failed: {e}")

    def close(self):
        self.env.close()

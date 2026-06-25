"""
record_demo.py — Ghi video 20 giây gameplay của trained DQN agent.

Chạy: python record_demo.py
Output: videos/demo_gameplay.mp4

Dùng obs vector 12 chiều (gym) — phù hợp với model DQN mới nhất.
"""

import os
import sys
import numpy as np

# Headless SDL (không cần cửa sổ hiển thị khi record)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import flappy_bird_gymnasium
import cv2
from src.agents.dqn_agent import DQNAgent
from src.environment.flappy_env import FlappyBirdEnv

# ── Config ────────────────────────────────────────────
CHECKPOINT    = "checkpoints/best_model.pt"
OUTPUT_VIDEO  = "videos/demo_gameplay.mp4"
DURATION_SECS = 22         # Ghi đủ 22s để có 20s gameplay
TARGET_FPS    = 30
STATE_DIM     = FlappyBirdEnv.OBS_DIM  # 12
# ─────────────────────────────────────────────────────

os.makedirs("videos", exist_ok=True)

# ── Load agent (auto-detect dims từ checkpoint) ───────
import torch
print("=== Flappy Bird Demo Recorder ===")
if not os.path.exists(CHECKPOINT):
    print(f"⚠️  Không tìm thấy checkpoint: {CHECKPOINT}")
    print("   Hãy train trước: python src/main.py --train --episodes 2000")
    sys.exit(1)

# Đọc dims thực tế từ checkpoint để tránh mismatch
_ckpt_meta = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
_w = _ckpt_meta["online_net"]["net.0.weight"]
_state_dim  = int(_ckpt_meta.get("state_dim", _w.shape[1]))
_hidden_dim = int(_w.shape[0])
print(f"   Checkpoint: state_dim={_state_dim}, hidden_dim={_hidden_dim}, ep={_ckpt_meta.get('episode_count')}")

agent = DQNAgent(state_dim=_state_dim, hidden_dim=_hidden_dim)
agent.load_weights(CHECKPOINT)
agent.set_eval_mode()
print(f"✅ Loaded checkpoint: {CHECKPOINT}")

# ── Setup env (dùng rgb_array để lấy frame + obs) ─────
# Env game để lấy frame RGB (render)
render_env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

# ── Setup video writer ────────────────────────────────
obs, info = render_env.reset()
frame = render_env.render()
H, W = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (W, H))
print(f"📹 Recording {DURATION_SECS}s → {OUTPUT_VIDEO}  [{W}×{H} @ {TARGET_FPS}fps]")

# ── Record loop ───────────────────────────────────────
total_frames   = DURATION_SECS * TARGET_FPS
frames_written = 0
episode        = 0
total_score    = 0
episode_scores = []

# State = obs vector (12 dims) — khớp với model mới
state = obs.astype(np.float32)

print(f"Bắt đầu record... (target: {total_frames} frames)")

while frames_written < total_frames:
    # Chọn action từ DQN
    action = agent.choose_action(state)

    obs, reward, terminated, truncated, info = render_env.step(action)
    done = terminated or truncated

    # Lấy frame RGB để ghi video
    curr_frame = render_env.render()

    # Ghi frame vào video (convert RGB→BGR)
    bgr = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)

    # ── Overlay HUD ─────────────────────────────────
    score = info.get("score", 0)
    ep_str    = f"Ep: {episode+1}"
    score_str = f"Score: {score}"
    frame_str = f"Frame: {frames_written+1}/{total_frames}"

    # Background bar (semi-transparent)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0, bgr)

    cv2.putText(bgr, ep_str,    (6,  19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0),   1, cv2.LINE_AA)
    cv2.putText(bgr, score_str, (80, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(bgr, frame_str, (160,19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(bgr, "DQN Agent", (W-80, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1, cv2.LINE_AA)

    writer.write(bgr)
    frames_written += 1

    # Chuyển obs mới thành state
    state = obs.astype(np.float32)

    if done:
        episode += 1
        ep_score = info.get("score", 0)
        episode_scores.append(ep_score)
        total_score += ep_score
        print(f"  Episode {episode}: score={ep_score}, frames={frames_written}/{total_frames}")

        obs, info = render_env.reset()
        state = obs.astype(np.float32)

writer.release()
render_env.close()

# ── Summary ───────────────────────────────────────────
print(f"\n✅ Video saved → {OUTPUT_VIDEO}")
print(f"   Episodes recorded : {episode}")
print(f"   Scores            : {episode_scores}")
if episode_scores:
    print(f"   Avg score         : {np.mean(episode_scores):.2f}")
    print(f"   Best score        : {max(episode_scores)}")
print(f"   Total frames      : {frames_written}")

size_mb = os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024
print(f"   File size         : {size_mb:.1f} MB")

"""
test_env.py — Kiểm tra môi trường Flappy Bird hoạt động.
Chạy: python test_env.py
"""

import gymnasium as gym
import flappy_bird_gymnasium

print("=== Flappy Bird Environment Test ===")

env = gym.make(
    "FlappyBird-v0",
    render_mode="human"
)

obs, info = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

done = False
total_reward = 0
steps = 0

while not done:
    action = 0  # Không flap — rơi thẳng

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1
    done = terminated or truncated

print(f"\nKết quả:")
print(f"  Steps: {steps}")
print(f"  Total reward: {total_reward:.2f}")
print(f"\nMôi trường OK! ✅")

env.close()

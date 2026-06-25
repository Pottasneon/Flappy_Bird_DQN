"""
src/utils/visualization.py

Vẽ learning curves, reward curves, epsilon decay bằng matplotlib.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    window: int = 20,
    show: bool = True,
):
    """
    Vẽ 4 biểu đồ:
      1. Episode Reward (raw + moving average)
      2. Score (pipes passed)
      3. Epsilon Decay
      4. TD Error

    Args:
        history : dict với keys: rewards, scores, epsilons, td_errors
        save_path: Đường dẫn lưu hình (PNG). Nếu None thì không lưu
        window   : Cửa sổ moving average
        show     : Hiển thị plot
    """
    rewards = history.get("rewards", [])
    scores = history.get("scores", [])
    epsilons = history.get("epsilons", [])
    td_errors = history.get("td_errors", [])

    n = len(rewards)
    if n == 0:
        print("[Viz] Không có data để plot")
        return

    episodes = np.arange(1, n + 1)

    # ─── Style ──────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle(
        "Flappy Bird RL+CV — Training Dashboard",
        fontsize=16, color="white", fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    colors = {
        "reward": "#58a6ff",
        "ma": "#f0883e",
        "score": "#56d364",
        "epsilon": "#d2a8ff",
        "td": "#ff7b72",
    }

    # ─── 1. Episode Reward ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, rewards, alpha=0.3, color=colors["reward"], linewidth=0.8)
    if n >= window:
        ma = _moving_average(rewards, window)
        ax1.plot(episodes[window-1:], ma, color=colors["ma"], linewidth=2, label=f"MA-{window}")
        ax1.legend(fontsize=9)
    ax1.set_title("Episode Reward", color="white", fontsize=12)
    ax1.set_xlabel("Episode", color="gray")
    ax1.set_ylabel("Total Reward", color="gray")
    ax1.tick_params(colors="gray")
    ax1.set_facecolor("#161b22")
    ax1.grid(alpha=0.2)

    # ─── 2. Score (Pipes Passed) ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(episodes, scores, color=colors["score"], alpha=0.6, width=1.0)
    if n >= window:
        score_ma = _moving_average(scores, window)
        ax2.plot(episodes[window-1:], score_ma, color="white", linewidth=2, label=f"MA-{window}")
        ax2.legend(fontsize=9)
    ax2.set_title("Score (Pipes Passed)", color="white", fontsize=12)
    ax2.set_xlabel("Episode", color="gray")
    ax2.set_ylabel("Pipes", color="gray")
    ax2.tick_params(colors="gray")
    ax2.set_facecolor("#161b22")
    ax2.grid(alpha=0.2)

    # ─── 3. Epsilon Decay ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, epsilons, color=colors["epsilon"], linewidth=2)
    ax3.fill_between(episodes, epsilons, alpha=0.2, color=colors["epsilon"])
    ax3.set_title("Epsilon Decay (Exploration Rate)", color="white", fontsize=12)
    ax3.set_xlabel("Episode", color="gray")
    ax3.set_ylabel("Epsilon ε", color="gray")
    ax3.tick_params(colors="gray")
    ax3.set_facecolor("#161b22")
    ax3.grid(alpha=0.2)

    # ─── 4. TD Error ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, td_errors, color=colors["td"], alpha=0.5, linewidth=0.8)
    if n >= window:
        td_ma = _moving_average(td_errors, window)
        ax4.plot(episodes[window-1:], td_ma, color="white", linewidth=2, label=f"MA-{window}")
        ax4.legend(fontsize=9)
    ax4.set_title("TD Error (Learning Signal)", color="white", fontsize=12)
    ax4.set_xlabel("Episode", color="gray")
    ax4.set_ylabel("|δ|", color="gray")
    ax4.tick_params(colors="gray")
    ax4.set_facecolor("#161b22")
    ax4.grid(alpha=0.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Viz] Saved plot → {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_frame_debug(frame_rgb, result, save_path: Optional[str] = None, show: bool = True):
    """
    Hiển thị frame với detection overlay để debug vision module.

    Args:
        frame_rgb : ndarray (H, W, 3) RGB
        result    : ProcessedFrame từ ImageProcessor
        save_path : Nếu có, lưu hình
        show      : Hiển thị
    """
    import cv2

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Vision Debug — Frame Analysis", fontsize=13)

    # ─── Raw Frame với annotations ─────────────────
    annotated = frame_rgb.copy()
    if result.bird:
        x, y, w, h = result.bird.bbox
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 220, 0), 2)
        cx, cy = int(result.bird.center_x), int(result.bird.center_y)
        cv2.circle(annotated, (cx, cy), 4, (255, 0, 0), -1)

    for pipe in result.top_pipes + result.bot_pipes:
        color = (0, 255, 100) if pipe.is_top else (0, 150, 255)
        cv2.rectangle(annotated, (pipe.x, pipe.y),
                      (pipe.x + pipe.w, pipe.y + pipe.h), color, 2)

    axes[0].imshow(annotated)
    axes[0].set_title("Detection Overlay")
    axes[0].axis("off")

    # ─── Bird Mask ─────────────────────────────────
    axes[1].imshow(result.mask_bird, cmap="inferno")
    axes[1].set_title("Bird Mask (HSV)")
    axes[1].axis("off")

    # ─── Pipe Mask ─────────────────────────────────
    axes[2].imshow(result.mask_pipe, cmap="viridis")
    axes[2].set_title("Pipe Mask (HSV)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Viz] Saved debug frame → {save_path}")
    if show:
        plt.show()

    plt.close()


def _moving_average(data: List[float], window: int) -> np.ndarray:
    """Tính moving average."""
    data_arr = np.array(data, dtype=float)
    return np.convolve(data_arr, np.ones(window) / window, mode="valid")

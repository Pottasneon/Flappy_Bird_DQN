"""
src/utils/logger.py

CSV Logger cho training metrics.
Log mỗi episode: reward, score, length, epsilon, td_error
"""

import os
import csv
from datetime import datetime
from typing import Optional


class TrainingLogger:
    """Ghi log training metrics ra CSV."""

    COLUMNS = ["episode", "reward", "score", "length", "epsilon", "td_error", "timestamp"]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Tên file theo timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"training_{ts}.csv")

        # Write header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)

        print(f"[Logger] Log → {self.log_path}")

    def log_episode(
        self,
        episode: int,
        reward: float,
        score: int,
        length: int,
        epsilon: float,
        td_error: float,
    ):
        """Ghi một dòng log cho episode."""
        row = [
            episode,
            f"{reward:.4f}",
            score,
            length,
            f"{epsilon:.6f}",
            f"{td_error:.6f}",
            datetime.now().isoformat(),
        ]
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

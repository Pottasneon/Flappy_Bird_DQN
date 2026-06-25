"""
src/play_hand.py — Chơi Flappy Bird bằng cử động tay

Sử dụng webcam + MediaPipe để nhận diện tay:
  - Mở bàn tay (>= 3 ngón duỗi) → FLAP
  - Nắm tay → Rơi tự do

Chạy:
    python -m src.play_hand
    python -m src.play_hand --no-preview    # Không hiển thị camera
    python -m src.play_hand --camera 1      # Dùng camera khác
"""

import sys
import os
import argparse
import time

# Đảm bảo import từ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chơi Flappy Bird bằng cử động tay qua webcam"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera ID (default: 0 = webcam mặc định)"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Không hiển thị cửa sổ camera preview"
    )
    parser.add_argument(
        "--no-flip", action="store_true",
        help="Không flip camera (tắt mirror mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  🎮 Flappy Bird — Hand Gesture Control")
    print(f"{'='*60}")
    print(f"  Camera ID : {args.camera}")
    print(f"  Preview   : {'OFF' if args.no_preview else 'ON'}")
    print(f"  Controls  :")
    print(f"    ✋ Mở bàn tay  → FLAP (bay lên)")
    print(f"    ✊ Nắm tay     → Rơi tự do")
    print(f"    Q / ESC       → Thoát")
    print(f"{'='*60}\n")

    # Import after args parsing
    from src.gesture.hand_controller import HandController
    from src.environment.flappy_env import FlappyBirdEnv

    # Khởi tạo components
    print("[Play] Đang mở camera...")
    controller = HandController(
        camera_id=args.camera,
        flip_horizontal=not args.no_flip,
    )

    print("[Play] Đang khởi tạo game...")
    env = FlappyBirdEnv(render_mode="human")

    # Game loop
    total_games = 0
    total_score = 0
    best_score = 0

    try:
        while True:
            total_games += 1
            print(f"\n--- Game #{total_games} ---")

            frame, info = env.reset()
            done = False
            game_score = 0
            step = 0

            while not done:
                # 1. Lấy action từ camera
                action = controller.get_action()

                # 2. Thực hiện action trong game
                frame, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                # 3. Hiển thị camera preview
                if not args.no_preview:
                    preview = controller.get_preview_frame()
                    if preview is not None:
                        # Thêm score overlay
                        score = info.get("score", info.get("pipes_passed", 0))
                        cv2.putText(
                            preview,
                            f"Score: {score}",
                            (preview.shape[1] - 150, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Hand Controller", preview)

                # 4. Xử lý keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q hoặc ESC
                    print("\n[Play] Thoát game!")
                    controller.close()
                    env.close()
                    return

            # Game over
            game_score = info.get("score", info.get("pipes_passed", 0))
            total_score += game_score
            best_score = max(best_score, game_score)
            avg_score = total_score / total_games

            print(
                f"  Score: {game_score} | "
                f"Best: {best_score} | "
                f"Avg: {avg_score:.1f} | "
                f"Steps: {step}"
            )

            # Chờ 1 giây trước khi reset
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n[Play] Ctrl+C — Thoát!")
    finally:
        controller.close()
        env.close()

        print(f"\n{'='*40}")
        print(f"  Session Summary")
        print(f"{'='*40}")
        print(f"  Games played : {total_games}")
        print(f"  Best score   : {best_score}")
        print(f"  Avg score    : {total_score / max(1, total_games):.1f}")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    main()

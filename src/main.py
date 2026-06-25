"""
src/main.py — CLI Entrypoint

Chạy từ root directory của project:

  Training:
    python src/main.py --train --episodes 500                    # DQN (mặc định)
    python src/main.py --train --episodes 500 --agent tabular    # Tabular Q
    python src/main.py --train --episodes 500 --agent dqn --double-dqn
    python src/main.py --train --episodes 200 --render

  Evaluation:
    python src/main.py --evaluate --episodes 10
    python src/main.py --evaluate --record-video --agent dqn

  Hand Gesture Play:
    python src/main.py --play-hand
    python src/main.py --play-hand --camera 1
"""

import argparse
import sys
import os

# Đảm bảo import từ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flappy Bird RL+CV — AI tự học chơi bằng Computer Vision + Deep Q-Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/main.py --train --episodes 500
  python src/main.py --train --episodes 500 --agent dqn --double-dqn
  python src/main.py --train --episodes 200 --agent tabular --render
  python src/main.py --evaluate --episodes 10 --agent dqn
  python src/main.py --evaluate --record-video --checkpoint checkpoints/best_model.pt
  python src/main.py --play-hand
  python src/main.py --play-hand --camera 1 --no-preview
        """,
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Chạy training")
    mode_group.add_argument("--evaluate", action="store_true", help="Chạy evaluation")
    mode_group.add_argument("--play-hand", action="store_true",
                            help="Chơi bằng cử động tay qua webcam")

    # Agent selection
    parser.add_argument("--agent", type=str, default="dqn",
                        choices=["tabular", "dqn"],
                        help="Loại agent RL (default: dqn)")

    # Common
    parser.add_argument("--episodes", type=int, default=500, help="Số episodes (default: 500)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint từ file (.npy cho linear/tabular, .pt cho dqn)")
    parser.add_argument("--render", action="store_true", help="Hiển thị game window (chậm hơn)")

    # Training — shared
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon ban đầu (default: 1.0)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay (default: 0.995)")
    parser.add_argument("--save-every", type=int, default=50, help="Lưu checkpoint mỗi N episodes")
    parser.add_argument("--save-frames", action="store_true", help="Lưu raw frames vào dataset/")
    parser.add_argument("--no-plot", action="store_true", help="Không hiển thị plot sau train")

    # Training — DQN specific
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="DQN hidden layer size (default: 128)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="DQN mini-batch size (default: 64)")
    parser.add_argument("--buffer-size", type=int, default=50000,
                        help="DQN replay buffer capacity (default: 50000)")
    parser.add_argument("--target-update", type=int, default=1000,
                        help="DQN target network update frequency (default: 1000)")
    parser.add_argument("--double-dqn", action="store_true", default=True,
                        help="Sử dụng Double DQN (default: True)")
    parser.add_argument("--no-double-dqn", action="store_true",
                        help="Tắt Double DQN")

    # Evaluation
    parser.add_argument("--record-video", action="store_true", help="Quay video evaluation")

    # Hand gesture play
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID cho hand gesture (default: 0)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Không hiển thị camera preview")

    return parser.parse_args()


# Agent type → default checkpoint extension
AGENT_EXTENSIONS = {
    "tabular": ".npy",
    "dqn": ".pt",
}


def run_training(args):
    from src.training.trainer import Trainer
    from src.utils.visualization import plot_training_history

    double_dqn = args.double_dqn and not args.no_double_dqn

    trainer = Trainer(
        n_episodes=args.episodes,
        save_every=args.save_every,
        save_frames=args.save_frames,
        render=args.render,
        agent_type=args.agent,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        # DQN-specific
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update,
        double_dqn=double_dqn,
    )

    # Load checkpoint nếu có
    ext = AGENT_EXTENSIONS[args.agent]
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    elif os.path.exists(f"checkpoints/weights{ext}"):
        ans = input(f"Tìm thấy checkpoint cũ (weights{ext}). Load tiếp? [y/N]: ").strip().lower()
        if ans == "y":
            trainer.load_checkpoint(f"checkpoints/weights{ext}")

    history = trainer.train()

    # Plot kết quả
    if not args.no_plot:
        print("\n[Main] Đang vẽ learning curves...")
        os.makedirs("logs", exist_ok=True)
        plot_training_history(
            history,
            save_path="logs/training_curves.png",
            show=True,
        )
        print("[Main] Plot đã lưu → logs/training_curves.png")


def run_evaluation(args):
    from src.evaluation.evaluator import Evaluator

    ext = AGENT_EXTENSIONS[args.agent]
    checkpoint = args.checkpoint or f"checkpoints/best_model{ext}"
    if not os.path.exists(checkpoint):
        # Fallback to weights
        checkpoint = f"checkpoints/weights{ext}"
        if not os.path.exists(checkpoint):
            print(f"[Error] Không tìm thấy checkpoint ({ext}). Hãy train trước: python src/main.py --train --agent {args.agent}")
            sys.exit(1)

    evaluator = Evaluator(
        checkpoint_path=checkpoint,
        render=args.render or True,  # Mặc định render khi evaluate
        record_video=args.record_video,
        agent_type=args.agent,
    )

    evaluator.load()
    results = evaluator.evaluate(n_episodes=args.episodes)
    evaluator.close()

    return results


def run_play_hand(args):
    """Chạy game với điều khiển bằng tay."""
    # Import and run play_hand module
    from src.play_hand import main as play_hand_main

    # Override sys.argv for play_hand's argparse
    play_args = ["play_hand"]
    if args.camera != 0:
        play_args.extend(["--camera", str(args.camera)])
    if args.no_preview:
        play_args.append("--no-preview")

    sys.argv = play_args
    play_hand_main()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Flappy Bird AI — Computer Vision + Reinforcement Learning")
    if args.play_hand:
        print(f"  Mode: 🖐️ Hand Gesture Control")
    elif args.train:
        print(f"  Mode: 🎓 Training ({args.agent.upper()})")
    else:
        print(f"  Mode: 📊 Evaluation ({args.agent.upper()})")
    print(f"{'='*60}")

    if args.train:
        run_training(args)
    elif args.evaluate:
        run_evaluation(args)
    elif args.play_hand:
        run_play_hand(args)


if __name__ == "__main__":
    main()

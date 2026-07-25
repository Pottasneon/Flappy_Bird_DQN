# Flappy Bird AI — Deep Q-Network + Computer Vision 

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-green)](https://gymnasium.farama.org)

**Ứng dụng Deep Reinforcement Learning (DQN) và Computer Vision vào game Flappy Bird.**  
Agent tự học chơi game và đạt **best score 17 pipes** sau 3000 episodes.

### 🎮 Demo Video

Click vào ảnh bên dưới để xem video demo gameplay của DQN agent:

<div align="center">

[![Flappy Bird DQN Demo](https://img.youtube.com/vi/J0Cci3VAWKc/maxresdefault.jpg)](https://youtube.com/shorts/J0Cci3VAWKc)

</div>

**🏆 Best Score: 17 pipes | 📊 Mean Score: 2.40**

</div>

---

## Tổng Quan

Dự án xây dựng một AI agent thông minh cho game Flappy Bird với hai mô-đun chính:

1. **Deep Q-Network (DQN)** — Agent tự học bằng Reinforcement Learning với Neural Network (PyTorch), sử dụng các kỹ thuật tiên tiến: Double DQN, Experience Replay, Target Network

### Kết Quả Đạt Được

| Phương pháp | Episodes | Best Score | Mean Score |
|---|---|---|---|
| Linear Q-Learning + CV (baseline) | 5,000 | 1 | 0.011 |
| **DQN + Gym Obs (kết quả cuối)** | **3,000** | **17** 🏆 | **2.40** |

---

## Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────┐
│                     Flappy Bird AI                           │
│                                                              │
│   ┌──────────┐    ┌──────────────────┐    ┌──────────────┐   │
│   │  Game    │───▶│  Observation     │───▶│  DQN Agent   │   │
│   │ (Pygame) │    │  Vector (12 dim) │    │  (PyTorch)   │   │
│   │          │◀───│  use_lidar=False │◀───│              │   │
│   └──────────┘    └──────────────────┘    └──────────────┘   │
│                                                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### DQN Training Pipeline

```
State (12 dims)
      ↓
QNetwork: FC(256) → ReLU → FC(256) → ReLU → FC(2)
      ↓
Q-values: [Q(s, no_flap), Q(s, flap)]
      ↓
ε-Greedy Policy → Action
      ↓
Reward + Next State → Replay Buffer (50K)
      ↓
Mini-batch (128) → Double DQN Update → Target Network
```

---

## Cài Đặt

### Yêu Cầu

- Python 3.12+
- macOS / Linux

### Cài Đặt Môi Trường

```bash
# 1. Tạo virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux

# 2. Cài dependencies
pip install -r requirements.txt
```

### Dependencies Chính

| Package | Version | Mục đích |
|---|---|---|
| `torch` | 2.x | Neural Network, DQN |
| `gymnasium` | latest | RL environment |
| `flappy-bird-gymnasium` | latest | Game environment |
| `opencv-python` | 4.x | Computer Vision, webcam |
| `numpy` | latest | Tính toán |
| `matplotlib` | latest | Visualize learning curves |

---

## Cách Chạy

### 1. Train DQN Agent (khuyến nghị)

```bash
# Train mới từ đầu — 3000 episodes (~30 phút)
python src/main.py --train --episodes 3000 --agent dqn

# Train thêm từ checkpoint đã có
python src/main.py --train --episodes 2000 --agent dqn

# Train không hiện plot (nhanh hơn)
python src/main.py --train --episodes 3000 --agent dqn --no-plot
```

### 2. Xem Bot Chơi (Evaluate)

```bash
# Xem bot chơi với model tốt nhất (best_model.pt)
python src/main.py --evaluate --episodes 20 --agent dqn --render

# Evaluate không render (chỉ xem điểm)
python src/main.py --evaluate --episodes 50 --agent dqn
```

### 3. Ghi Video Demo

```bash
# Ghi 22 giây gameplay → videos/demo_gameplay.mp4
python record_demo.py
```
### 5. Test Môi Trường

```bash
python test_env.py
```

---

## Cấu Trúc Thư Mục

```
Flappy_bird_Project/
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py          # Deep Q-Network Agent (PyTorch)
│   │   ├── tabular_q_agent.py    # Tabular Q-Learning (baseline)
│   │   ├── replay_buffer.py      # Experience Replay Buffer
│   │   └── __init__.py
│   ├── environment/
│   │   └── flappy_env.py         # Gymnasium wrapper (obs/frame mode)
│   ├── training/
│   │   ├── trainer.py            # Training loop (DQN-optimized)
│   │   └── rewards.py            # Reward Shaping
│   ├── evaluation/
│   │   └── evaluator.py          # Agent evaluation + video recording
│   ├── vision/
│   │   ├── image_processor.py    # OpenCV frame processing
│   │   └── feature_extractor.py  # CV feature extraction (dx, dy, v)
│   ├── utils/
│   │   ├── logger.py             # CSV training logger
│   │   └── visualization.py      # Matplotlib learning curves
│   └── main.py                   # CLI entrypoint
├── checkpoints/
│   ├── best_model.pt             # Model tốt nhất (score 17)
│   └── weights.pt                # Checkpoint cuối cùng
├── logs/                         # CSV training logs
├── videos/
│   └── demo_gameplay_final.mp4   # Demo video (22s, H.264)
├── record_demo.py                # Script ghi video demo
├── RESEARCH_NOTES.md             # Tài liệu nghiên cứu đầy đủ
├── requirements.txt
└── Dockerfile
```

---

## Kỹ Thuật Sử Dụng

### Deep Q-Network (DQN)

| Kỹ thuật | Chi tiết |
|---|---|
| **Neural Network** | FC(256) → ReLU → FC(256) → ReLU → FC(2) |
| **Double DQN** | Online net chọn action, target net đánh giá |
| **Experience Replay** | Buffer 50,000 transitions, mini-batch 128 |
| **Target Network** | Hard update mỗi 500 steps |
| **Optimizer** | Adam, lr = 3×10⁻⁴ |
| **Loss Function** | Huber Loss (Smooth L1) |
| **Gradient Clipping** | max_norm = 10.0 |
| **Discount Factor γ** | 0.99 |
| **Epsilon Decay** | 1.0 → 0.01 (decay = 0.998/episode) |

### Observation Space (12 dims)

State vector lấy trực tiếp từ `flappy-bird-gymnasium` (`use_lidar=False`):

| Dim | Tên | Ý nghĩa |
|---|---|---|
| [0-2] | `last_pipe_*` | Thông tin pipe trước |
| [3-5] | `next_pipe_*` | **Khoảng cách đến pipe tiếp theo** ← quan trọng nhất |
| [6-8] | `next_next_pipe_*` | Pipe kế kế tiếp (look-ahead) |
| [9] | `player_vel` | **Vận tốc chim** |
| [10] | `player_rot` | Góc xoay chim |
| [11] | `score` | Điểm hiện tại |

### Reward Shaping

| Sự kiện | Gym reward | Bonus | Tổng |
|---|---|---|---|
| Mỗi step sống | +0.1 | +0.05 | **+0.15** |
| Qua pipe mới | +1.0 | +5.0 | **+6.0** |
| Chết | -1.0 | -5.0 | **-6.0** |
| Ở gần tâm gap | — | +0.1 | **+0.1** |

### Computer Vision (Legacy / CV Mode)

Pipeline CV ban đầu (dùng khi `use_obs=False`):

| Kỹ thuật | Vai trò |
|---|---|
| HSV Color Segmentation | Tách chim (vàng) và pipe (xanh) khỏi nền |
| Contour Detection | Tìm viền vật thể |
| Bounding Box | Lấy tọa độ chim và pipe |
| EMA Velocity Estimation | Ước tính vận tốc chim qua 2 frames |

---

## Kết Quả Training

### Learning Curve (3000 episodes)

| Giai đoạn | Mean Score | Max Score | Avg Reward |
|---|---|---|---|
| Ep 0–200 | 0.00 | 0 | -8.8 |
| Ep 200–600 | 0.14 | 2 | +2.0 |
| Ep 600–1000 | 0.35 | 3 | +4.5 |
| Ep 1000–1400 | 0.49 | **5** ✅ | +6.5 |
| Ep 1400–2000 | 1.18 | 9 | +15.4 |
| Ep 2000–2200 | 2.00 | **17** 🏆 | +23.5 |
| Ep 2800–3000 | 2.40 | 15 | +27.7 |

### So Sánh Phương Pháp

```
Linear Q + CV (5000 ep):  ████░░░░░░░░░░░░░░  max = 1
DQN + Obs (3000 ep):      ████████████████████ max = 17
```

<<<<<<< HEAD
> **Kết luận**: Chất lượng state representation (12-dim obs) quan trọng hơn độ phức tạp thuật toán. Chuyển từ CV (3 dims, noisy) → gym obs (12 dims, chính xác) cải thiện 17× chỉ với 60% số episodes.

---

## Demo Video

Click vào ảnh bên dưới để xem video demo gameplay của DQN agent:

<div align="center">

[![Flappy Bird DQN Demo](https://img.youtube.com/vi/J0Cci3VAWKc/maxresdefault.jpg)](https://youtube.com/shorts/J0Cci3VAWKc)

</div>

Kết quả trong video:
- Episode 1: score = 1
- Episode 2: score = 2
- Episode 4: **score = 5** ✅
- Best episode: **score = 17** (trong training)
=======
> **Kết luận**: Chất lượng state representation (12-dim obs) quan trọng hơn độ phức tạp thuật toán. Chuyển từ CV (3 dims, noisy) → gym obs (12 dims, chính xác) cải thiện hiệu suất đáng kể.
>>>>>>> c8e86840eabd3af93ff5d5098c7a43f9e6f8c42a

---

## Tài Liệu Nghiên Cứu

Xem [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) để tìm hiểu chi tiết về tất cả thuật toán, kỹ thuật, và tài liệu tham khảo được dùng trong dự án.

---

## Tham Khảo

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* — Nature
- Van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning* — AAAI
- [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- [Spinning Up in Deep RL — OpenAI](https://spinningup.openai.com)

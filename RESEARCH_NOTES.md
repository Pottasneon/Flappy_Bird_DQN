# Tổng Hợp Kỹ Thuật & Thuật Toán — Dự Án Flappy Bird AI

> **Mục đích:** Tài liệu tham khảo toàn diện cho việc học, nghiên cứu và viết báo cáo về dự án ứng dụng Reinforcement Learning + Computer Vision vào game Flappy Bird.

---

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Reinforcement Learning — Nền Tảng](#2-reinforcement-learning--nền-tảng)
3. [Q-Learning](#3-q-learning)
4. [Deep Q-Network (DQN)](#4-deep-q-network-dqn)
5. [Experience Replay](#5-experience-replay)
6. [Target Network](#6-target-network)
7. [Double DQN](#7-double-dqn)
8. [Epsilon-Greedy Exploration](#8-epsilon-greedy-exploration)
9. [Reward Shaping](#9-reward-shaping)
10. [Neural Network Architecture](#10-neural-network-architecture)
11. [Computer Vision — Nhận Diện Môi Trường](#11-computer-vision--nhận-diện-môi-trường)
12. [Hand Gesture Control (MediaPipe)](#12-hand-gesture-control-mediapipe)
13. [Kỹ Thuật Tối Ưu Hóa](#13-kỹ-thuật-tối-ưu-hóa)
14. [So Sánh Kết Quả Thực Nghiệm](#14-so-sánh-kết-quả-thực-nghiệm)
15. [Thuật Ngữ Quan Trọng](#15-thuật-ngữ-quan-trọng)
16. [Tài Liệu Tham Khảo](#16-tài-liệu-tham-khảo)

---

## 1. Tổng Quan Hệ Thống

### Kiến Trúc Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Flappy Bird AI                         │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Game   │───▶│  Observation │───▶│    DQN Agent      │  │
│  │  Engine  │    │  (12 dims)   │    │  (Neural Network) │  │
│  │(Pygame)  │◀───│              │◀───│                   │  │
│  └──────────┘    └──────────────┘    └───────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Hand Gesture Mode (tùy chọn)            │   │
│  │  Webcam → MediaPipe → HandController → Action {0,1}  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Các Thành Phần Chính

| Module | Vai trò |
|---|---|
| `FlappyBirdEnv` | Wrapper môi trường game (gymnasium) |
| `DQNAgent` | Agent học Deep Q-Learning |
| `QNetwork` | Mạng Neural xấp xỉ Q-function |
| `ReplayBuffer` | Bộ nhớ kinh nghiệm |
| `RewardShaper` | Thiết kế hàm thưởng |
| `HandController` | Điều khiển bằng cử chỉ tay (MediaPipe) |
| `Trainer` | Vòng lặp huấn luyện |
| `Evaluator` | Đánh giá agent sau huấn luyện |

---

## 2. Reinforcement Learning — Nền Tảng

### Định Nghĩa

**Reinforcement Learning (RL)** — Học Tăng Cường — là một nhánh của Machine Learning trong đó một **agent** học cách hành động trong một **môi trường** thông qua việc tối đa hóa **phần thưởng tích lũy** theo thời gian, mà không cần dữ liệu được gán nhãn trước.

### Bộ 5 Thành Phần (MDP)

Bài toán RL được mô hình hóa bằng **Markov Decision Process (MDP)** gồm:

| Ký hiệu | Tên | Trong Flappy Bird |
|---|---|---|
| **S** | State Space | Obs vector 12 chiều từ gym |
| **A** | Action Space | {0: không flap, 1: flap} |
| **R** | Reward Function | Sống: +0.1, Pipe: +6, Chết: -6 |
| **P** | Transition Probability | Do game engine quyết định |
| **γ** | Discount Factor | 0.99 |

### Tính Chất Markov

> "Trạng thái hiện tại chứa đủ thông tin để dự đoán tương lai — không cần lịch sử."

Obs vector của `flappy-bird-gymnasium` thỏa mãn tính chất này vì nó chứa:
- Khoảng cách đến pipe tiếp theo
- Vận tốc của chim
- Thông tin pipe kế kế tiếp (look-ahead)

### Mục Tiêu Học

Agent học policy **π(s) → a** tối đa hóa **expected cumulative discounted reward**:

```
G_t = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ... = Σ γᵏ · R_{t+k+1}
```

Trong đó γ ∈ [0, 1] là discount factor — điều tiết sự quan trọng của phần thưởng tương lai.

---

## 3. Q-Learning

### Khái Niệm Q-Value

**Q-function** (Action-Value Function) `Q(s, a)` cho biết: "Nếu đang ở trạng thái **s** và thực hiện hành động **a**, sau đó làm theo policy tối ưu, thì expected total reward là bao nhiêu?"

```
Q*(s, a) = E[G_t | S_t=s, A_t=a, π*]
```

### Phương Trình Bellman

Nền tảng toán học của Q-Learning:

```
Q*(s, a) = E[R + γ · max_{a'} Q*(s', a')]
```

Giải thích:
- `R` — phần thưởng ngay lập tức
- `γ` — hệ số chiết khấu (0.99 trong dự án)
- `max Q*(s', a')` — Q-value tốt nhất ở trạng thái kế tiếp

### Thuật Toán Q-Learning (Tabular)

```
Khởi tạo: Q(s, a) = 0 cho tất cả s, a
Lặp mỗi step:
  1. Chọn action: a = argmax Q(s, .) hoặc random (ε-greedy)
  2. Thực hiện a, nhận reward r, trạng thái mới s'
  3. TD Error: δ = r + γ·max Q(s', a') - Q(s, a)
  4. Cập nhật: Q(s, a) ← Q(s, a) + α·δ
```

### Hạn Chế Của Tabular Q-Learning

| Vấn đề | Giải thích |
|---|---|
| **Curse of Dimensionality** | State space liên tục → không thể lưu bảng Q |
| **Không tổng quát hóa** | Không học từ các state tương tự nhau |
| **Bộ nhớ lớn** | Cần O(|S| × |A|) bộ nhớ |

→ **Giải pháp**: Dùng Neural Network để xấp xỉ Q-function = **Deep Q-Network**

---

## 4. Deep Q-Network (DQN)

### Lịch Sử

- **2013**: DeepMind giới thiệu DQN trong paper *"Playing Atari with Deep Reinforcement Learning"*
- **2015**: Cải tiến với Experience Replay + Target Network, đạt trình độ con người ở 49 game Atari
- **Paper gốc**: Mnih et al., *"Human-level control through deep reinforcement learning"*, Nature 2015

### Ý Tưởng Cốt Lõi

Thay thế bảng Q bằng **mạng Neural**:

```
Q(s, a; θ) ≈ Q*(s, a)
```

Mạng nhận **state vector** làm input, trả về **Q-value cho tất cả actions** cùng lúc.

### Kiến Trúc Mạng Trong Dự Án

```
Input Layer  (12 neurons)  ← obs vector: [dist_pipe, vel_bird, ...]
     ↓
Hidden Layer (256 neurons) + ReLU activation
     ↓
Hidden Layer (256 neurons) + ReLU activation
     ↓
Output Layer (2 neurons)   ← [Q(s, no_flap), Q(s, flap)]
```

```python
# Code thực tế trong dqn_agent.py
self.net = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),   # 12 → 256
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),  # 256 → 256
    nn.ReLU(),
    nn.Linear(hidden_dim, n_actions),   # 256 → 2
)
```

### Hàm Loss — Huber Loss (Smooth L1)

Thay vì MSE thuần túy, DQN dùng **Huber Loss** (còn gọi là Smooth L1 Loss):

```
L(δ) = {
    0.5 · δ²           nếu |δ| < 1
    |δ| - 0.5          nếu |δ| ≥ 1
}
```

**Lý do**: MSE nhạy cảm với outliers (reward lớn bất thường) → Huber Loss ổn định hơn.

### TD Target (DQN)

```
y = r + γ · max_{a'} Q(s', a'; θ⁻)       (1 - done)
```

Trong đó `θ⁻` là tham số của **target network** (cập nhật chậm hơn online network).

### Quá Trình Training

```
Mỗi step:
  1. Agent quan sát state s (12 dims)
  2. Chọn action a theo ε-greedy
  3. Nhận reward r và state mới s'
  4. Lưu (s, a, r, s', done) vào Replay Buffer
  5. Sample mini-batch 128 transitions từ buffer
  6. Tính TD target: y = r + γ · Q_target(s', a*)
  7. Tính loss: L = Huber(Q_online(s,a) - y)
  8. Backpropagation + gradient descent
  9. Mỗi 500 steps: cập nhật Target Network
```

---

## 5. Experience Replay

### Vấn Đề Không Có Experience Replay

Khi train trực tiếp từ các bước liên tiếp:
- **Temporal correlation**: các sample liên kết nhau (s_t → s_{t+1} rất giống nhau)
- → Mạng bị overfit vào trạng thái hiện tại
- → Gradient update không ổn định

### Giải Pháp: Replay Buffer

**Experience Replay** lưu trữ các transitions (kinh nghiệm) vào một bộ nhớ tròn (circular buffer), sau đó sample ngẫu nhiên để train.

```
Buffer D = { (s₁,a₁,r₁,s₁',d₁), (s₂,a₂,r₂,s₂',d₂), ... }
                          ↑
                   Capacity = 50,000 transitions
```

### Lợi Ích

| Lợi ích | Giải thích |
|---|---|
| **Phá vỡ correlation** | Sampling ngẫu nhiên → các sample độc lập hơn |
| **Data efficiency** | Mỗi transition được dùng nhiều lần |
| **Ổn định training** | Phân phối training data đồng đều hơn |

### Implementation

```python
# Trong replay_buffer.py
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)  # Tự động xóa cũ nhất

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Chuyển thành PyTorch tensors...
```

### Hyperparameters Đã Dùng

| Tham số | Giá trị |
|---|---|
| Buffer capacity | 50,000 transitions |
| Batch size | 128 |
| Min replay size (warm-up) | 1,000 |

---

## 6. Target Network

### Vấn Đề "Moving Target"

Trong DQN cơ bản, khi cập nhật Q-function:
- **TD Target** `y = r + γ·max Q(s', a'; θ)` phụ thuộc vào chính `θ`
- Khi `θ` thay đổi → target cũng thay đổi → như "đuổi theo mục tiêu đang di chuyển"
- → Divergence (phân kỳ), training không ổn định

### Giải Pháp: Hai Mạng Riêng Biệt

```
Online Network  θ   : Học liên tục, dùng để chọn action
Target Network  θ⁻  : Cập nhật chậm (frozen), dùng tính TD target
```

```python
# Trong dqn_agent.py
self.online_net = QNetwork(state_dim, n_actions, hidden_dim)
self.target_net = QNetwork(state_dim, n_actions, hidden_dim)

# Copy toàn bộ weights ban đầu
self.target_net.load_state_dict(self.online_net.state_dict())
self.target_net.eval()  # Không train target network

# Mỗi 500 steps: Hard update
def _update_target_network(self):
    self.target_net.load_state_dict(self.online_net.state_dict())
```

### Hard Update vs Soft Update

| Loại | Công thức | Đặc điểm |
|---|---|---|
| **Hard Update** (dùng trong dự án) | θ⁻ ← θ (mỗi N steps) | Đơn giản, hiệu quả |
| **Soft Update (Polyak)** | θ⁻ ← τ·θ + (1-τ)·θ⁻ | Mượt mà hơn, cần tune τ |

---

## 7. Double DQN

### Vấn Đề Overestimation Bias

DQN tiêu chuẩn có xu hướng **overestimate** Q-values:

```
y = r + γ · max_{a'} Q(s', a'; θ⁻)
```

Cùng một mạng `θ⁻` vừa **chọn action** vừa **đánh giá action** đó → chọn action ngẫu nhiên tốt nhất → estimate bị thiên cao.

### Giải Pháp: Double DQN

**Paper**: van Hasselt et al., *"Deep Reinforcement Learning with Double Q-learning"*, AAAI 2016.

**Ý tưởng**: Tách biệt bước **chọn action** và **đánh giá action**:

```
Bước 1 (chọn action): a* = argmax_{a'} Q(s', a'; θ)        ← Online network
Bước 2 (đánh giá)  : y  = r + γ · Q(s', a*; θ⁻)           ← Target network
```

```python
# Trong dqn_agent.py — _train_step()
if self.double_dqn:
    # Online network chọn action tốt nhất
    next_q_online = self.online_net(next_states)
    best_actions = next_q_online.argmax(dim=1, keepdim=True)

    # Target network đánh giá action đó
    next_q_target = self.target_net(next_states)
    next_q_sa = next_q_target.gather(1, best_actions)
```

### So Sánh DQN vs Double DQN

| | DQN | Double DQN |
|---|---|---|
| **Action selection** | Target network | Online network |
| **Action evaluation** | Target network | Target network |
| **Overestimation** | Cao | Thấp |
| **Hiệu quả** | Tốt | **Tốt hơn** |

---

## 8. Epsilon-Greedy Exploration

### Exploration vs Exploitation Dilemma

- **Exploration**: Thử hành động ngẫu nhiên để khám phá môi trường
- **Exploitation**: Chọn action tốt nhất đã biết

Nếu chỉ exploit → bị kẹt ở local optimum.
Nếu chỉ explore → không hội tụ.

### Epsilon-Greedy Policy

```
π(s) = {
    random action    với xác suất ε   (exploration)
    argmax Q(s, a)   với xác suất 1-ε (exploitation)
}
```

```python
def choose_action(self, state):
    if np.random.random() < self.epsilon:
        return np.random.randint(0, self.N_ACTIONS)   # Exploration
    else:
        q_values = self.online_net(state_tensor)
        return q_values.argmax().item()                # Exploitation
```

### Epsilon Decay Schedule

Trong dự án dùng **Exponential Decay**:

```
ε_{t+1} = max(ε_min, ε_t × decay_rate)
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `epsilon` (ban đầu) | 1.0 | 100% random lúc đầu |
| `epsilon_min` | 0.01 | Luôn giữ 1% random |
| `epsilon_decay` | 0.998 | Giảm 0.2% mỗi episode |

**Tại sao 0.998?** Với 3000 episodes:
- Ep 0: ε = 1.0 (100% explore)
- Ep ~800: ε ≈ 0.2 (bắt đầu exploit nhiều hơn)
- Ep ~1100: ε = 0.01 (99% exploit)

---

## 9. Reward Shaping

### Tại Sao Cần Reward Shaping?

Reward tự nhiên của game (`+0.1/step`, `+1/pipe`, `-1/chết`) có vấn đề:
- **Sparse reward**: Agent chỉ nhận +1 khi qua pipe — rất hiếm lúc đầu
- **Delayed reward**: Khó biết action nào thực sự tốt
- → Agent học rất chậm hoặc không học được

### Reward Function Đã Thiết Kế

```python
def compute(env_reward, terminated, info, state):
    reward = env_reward        # Base: reward từ gym (+0.1/step, -1/chết)

    if terminated:
        reward += -5.0         # Death penalty bổ sung
        return reward

    reward += 0.05             # Alive bonus (sống lâu hơn)

    if pipes_new > 0:
        reward += 5.0 × pipes  # Pipe bonus bổ sung

    # Position bonus: ở gần tâm gap
    if gap_balance < 0.15:
        reward += 0.1

    return reward
```

### Bảng Reward

| Sự kiện | Gym reward | Bonus | **Tổng** |
|---|---|---|---|
| Mỗi step sống | +0.1 | +0.05 | **+0.15** |
| Qua 1 pipe | +1.0 | +5.0 | **+6.0** |
| Chết | -1.0 | -5.0 | **-6.0** |
| Ở gần tâm gap | +0 | +0.1 | **+0.1** |

### Nguyên Tắc Thiết Kế Reward

1. **Death penalty mạnh** → Agent tránh chết
2. **Pipe bonus lớn** → Agent có động lực qua pipe
3. **Survival reward nhỏ** → Khuyến khích sống lâu
4. **Position bonus** → Hướng dẫn bay đúng vị trí
5. **Không quá nhiều dense reward** → Tránh agent bị "lười" (farm easy rewards)

---

## 10. Neural Network Architecture

### Activation Function — ReLU

**ReLU (Rectified Linear Unit)**:
```
ReLU(x) = max(0, x)
```

Ưu điểm so với Sigmoid/Tanh:
- Không bị vanishing gradient problem
- Tính toán nhanh
- Hiệu quả cho deep networks

### Optimizer — Adam

**Adam (Adaptive Moment Estimation)** — Dùng trong dự án với `lr = 3×10⁻⁴`:

```
m_t = β₁·m_{t-1} + (1-β₁)·∇L      (momentum - bậc 1)
v_t = β₂·v_{t-1} + (1-β₂)·∇L²     (variance - bậc 2)
θ_t = θ_{t-1} - α · m_t / √(v_t + ε)
```

Adam tự điều chỉnh learning rate cho từng tham số — phù hợp hơn SGD thuần túy cho RL.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
```

**Tại sao?** Trong RL, TD errors có thể rất lớn → gradient exploding → Clipping giới hạn norm của gradient ≤ 10.

### Batch Normalization

*Không dùng trong dự án này* — trong RL, batch normalization có thể gây vấn đề vì phân phối state thay đổi theo thời gian training.

---

## 11. Computer Vision — Nhận Diện Môi Trường

> **Lưu ý**: Pipeline CV ban đầu được thiết kế nhưng sau đó được thay thế bằng obs vector từ gym (chính xác hơn). CV vẫn được dùng trong chế độ camera (hand gesture).

### Các Kỹ Thuật CV Đã Dùng

#### HSV Color Segmentation
```
RGB → HSV → Threshold → Binary Mask
```
- HSV (Hue-Saturation-Value) ổn định hơn RGB với ánh sáng thay đổi
- Dùng để detect màu vàng của chim, màu xanh của pipe

#### Contour Detection (OpenCV)
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
Tìm biên của các vùng màu → xác định bounding box của chim và pipe.

#### Feature Extraction từ Pixels

Từ frame RGB (288×512×3), pipeline CV tính:
```
state = [dx, dy, velocity]  ← chỉ 3 chiều

dx : khoảng cách ngang đến pipe tiếp theo (normalized)
dy : khoảng cách dọc đến tâm gap (normalized)
v  : vận tốc ước tính theo Y (EMA-smoothed)
```

### Vì Sao CV Kém Hơn Obs Vector?

| | CV Pipeline (3 dims) | Gym Obs (12 dims) |
|---|---|---|
| **Độ chính xác** | Phụ thuộc color detection | Chính xác tuyệt đối |
| **Noise** | Cao (frame nhiễu) | Không |
| **Thông tin** | Ít (3 features) | Nhiều (12 features) |
| **Tốc độ** | Chậm (xử lý ảnh) | Nhanh |
| **Best score** | 1 (5000 ep) | 17 (3000 ep) |

### Observation Vector 12 Chiều

| Index | Tên | Ý nghĩa |
|---|---|---|
| [0] | last_pipe_h_dist | Khoảng cách ngang đến pipe trước |
| [1] | last_top_v_dist | Khoảng cách đến top pipe trước |
| [2] | last_bot_v_dist | Khoảng cách đến bottom pipe trước |
| [3] | next_pipe_h_dist | **Khoảng cách ngang đến pipe tiếp theo** |
| [4] | next_top_v_dist | **Khoảng cách đến top pipe tiếp theo** |
| [5] | next_bot_v_dist | **Khoảng cách đến bottom pipe tiếp theo** |
| [6-8] | next_next_pipe_* | Pipe kế kế tiếp (look-ahead) |
| [9] | player_vel | **Vận tốc của chim** |
| [10] | player_rot | Góc xoay của chim |
| [11] | score | Điểm hiện tại |

---

## 12. Hand Gesture Control (MediaPipe)

### MediaPipe Hands

**MediaPipe** là thư viện ML của Google để xử lý media real-time. Module **Hands** detect 21 landmarks của bàn tay trong 3D (x, y, z).

```
         4 (THUMB_TIP)
         |
    3 ---+
    |
    2 (THUMB_MCP)
    |
    1
    |
    0 (WRIST)
         |
    5---6---7---8 (INDEX)
    |
    9--10--11--12 (MIDDLE)
    |
   13--14--15--16 (RING)
    |
   17--18--19--20 (PINKY)
```

### API Mới: MediaPipe Tasks

Dự án dùng **MediaPipe Tasks API** (phiên bản mới, >= 0.10):
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

detector = vision.HandLandmarker.create_from_options(options)
result = detector.detect(mp_image)
```

*API cũ* (`mp.solutions.hands`) đã bị loại bỏ trong phiên bản >= 0.10.

### Finger Detection Algorithm

```python
def _count_fingers(landmarks):
    fingers = 0

    # Thumb: |tip.x - mcp.x| > 0.05
    if abs(landmarks[4].x - landmarks[2].x) > 0.05:
        fingers += 1

    # Index/Middle/Ring/Pinky: tip.y < pip.y (tip cao hơn PIP)
    for tip_idx, pip_idx in [(8,6), (12,10), (16,14), (20,18)]:
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            fingers += 1

    return fingers  # 0-5
```

### Gesture → Action Mapping

```
>= 3 ngón tay duỗi  →  FLAP (action = 1)  → Chim bay lên
< 3 ngón tay duỗi   →  FALL (action = 0)  → Chim rơi
```

### Debounce Mechanism

```python
FLAP_COOLDOWN = 0.15  # 150ms giữa 2 lần flap
if now - last_flap_time >= FLAP_COOLDOWN:
    return 1  # Flap
```
Tránh flap liên tục khi giơ tay lên.

---

## 13. Kỹ Thuật Tối Ưu Hóa

### 1. Hyperparameter Tuning

| Hyperparameter | Giá trị thử nghiệm | Giá trị cuối |
|---|---|---|
| Learning rate | 0.01, 0.001, **3e-4** | **3e-4** |
| Hidden dim | 128, **256** | **256** |
| Batch size | 64, **128** | **128** |
| Target update freq | 1000, **500** | **500** |
| Epsilon decay | 0.995, **0.998** | **0.998** |
| Buffer size | **50,000** | **50,000** |

### 2. Network Capacity

Tăng từ 2 layers × 128 neurons → 2 layers × **256 neurons**:
- **Lý do**: State space 12 chiều cần capacity lớn hơn để học phi tuyến phức tạp

### 3. Warm-up Phase

```python
min_replay_size = 1000  # Chờ 1000 transitions trước khi train
```
Tránh train từ dữ liệu quá ít → gradient update không ổn định.

### 4. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
```

### 5. Logging & Monitoring

CSV logger ghi lại mỗi episode:
```
episode, reward, score, length, epsilon, td_error, timestamp
```

→ Phân tích hậu kỳ để detect:
- Underfitting (reward không tăng)
- Catastrophic forgetting (reward tụt đột ngột)
- Exploration không đủ (epsilon giảm quá nhanh)

---

## 14. So Sánh Kết Quả Thực Nghiệm

### Experiment 1: Linear Q-Learning + CV (Baseline)

| Chỉ số | Giá trị |
|---|---|
| State dim | 3 (CV: dx, dy, velocity) |
| Episodes | 5,000 |
| **Best score** | **1** |
| Mean score | 0.011 |
| Mean reward | -6.1 |
| Thời gian | ~20 phút |

**Nhận xét**: CV pipeline không ổn định, state quá ít thông tin.

### Experiment 2: DQN + Gym Obs (Kết Quả Tốt)

| Chỉ số | Giá trị |
|---|---|
| State dim | 12 (gym obs vector) |
| Episodes | 3,000 |
| **Best score** | **17** |
| Mean score (cuối) | 2.40 |
| Mean reward (cuối) | +27.7 |
| Thời gian | 32.4 phút |

**Learning Curve:**

```
Ep 0-200   : max=0   → Pure exploration
Ep 200-600 : max=2   → Bắt đầu học
Ep 600-1000: max=3   → Tiến bộ chậm
Ep 1000-1200: max=5  → Đạt mục tiêu ✅
Ep 1400-2000: max=9  → Tiếp tục cải thiện
Ep 2000-2200: max=17 → Kỷ lục! 🏆
```

### Kết Luận Thực Nghiệm

> **Chất lượng của state representation quan trọng hơn độ phức tạp của thuật toán.** Switching từ CV (3 dims) sang gym obs (12 dims) cải thiện best score từ 1 → 17 chỉ với 60% số episodes.

---

## 15. Thuật Ngữ Quan Trọng

| Thuật ngữ | Định nghĩa |
|---|---|
| **Agent** | Thực thể ra quyết định (bot AI) |
| **Environment** | Môi trường agent tương tác (game Flappy Bird) |
| **State (s)** | Biểu diễn trạng thái hiện tại |
| **Action (a)** | Hành động agent thực hiện {0, 1} |
| **Reward (r)** | Tín hiệu phản hồi từ môi trường |
| **Policy (π)** | Chiến lược chọn action: s → a |
| **Value Function** | Expected return từ state s |
| **Q-Function** | Expected return từ state s + action a |
| **Episode** | Một lần chơi từ đầu đến chết |
| **Discount Factor (γ)** | Hệ số chiết khấu reward tương lai |
| **TD Error (δ)** | Temporal Difference Error — tín hiệu học |
| **Bellman Equation** | Phương trình tối ưu giá trị |
| **MDP** | Markov Decision Process |
| **Exploration** | Thử ngẫu nhiên để khám phá |
| **Exploitation** | Dùng kiến thức đã có |
| **On-policy** | Learn từ current policy (VD: SARSA) |
| **Off-policy** | Learn từ data bất kỳ (VD: Q-Learning, DQN) |
| **Replay Buffer** | Bộ nhớ lưu trữ kinh nghiệm |
| **Batch** | Tập hợp samples dùng 1 lần update |
| **Backpropagation** | Thuật toán tính gradient trong NN |
| **Gradient Descent** | Tối ưu hóa weights bằng gradient |
| **Overestimation Bias** | Xu hướng đánh giá Q-value cao hơn thực tế |
| **Catastrophic Forgetting** | NN quên kiến thức cũ khi học mới |
| **Convergence** | Trạng thái model không thay đổi nữa |
| **Hyperparameter** | Tham số cấu hình không được học |
| **Learning Curve** | Đồ thị performance theo thời gian |
| **FPS** | Frames Per Second |
| **Landmark** | Điểm đặc trưng trên bàn tay (MediaPipe) |
| **Debounce** | Kỹ thuật tránh trigger liên tục |

---

## 16. Tài Liệu Tham Khảo

### Papers Gốc

1. **DQN (2015)**
   > Mnih, V., et al. *"Human-level control through deep reinforcement learning."* Nature, 518(7540), 529-533.
   > https://www.nature.com/articles/nature14236

2. **Double DQN (2016)**
   > Van Hasselt, H., Guez, A., & Silver, D. *"Deep Reinforcement Learning with Double Q-learning."* AAAI 2016.
   > https://arxiv.org/abs/1509.06461

3. **Experience Replay**
   > Lin, L. J. *"Self-improving reactive agents based on reinforcement learning, planning and teaching."* Machine Learning, 8(3-4), 293-321. (1992)

4. **Adam Optimizer**
   > Kingma, D. P., & Ba, J. *"Adam: A Method for Stochastic Optimization."* ICLR 2015.
   > https://arxiv.org/abs/1412.6980

5. **Reward Shaping**
   > Ng, A. Y., Harada, D., & Russell, S. *"Policy Invariance Under Reward Transformations."* ICML 1999.

### Thư Viện Sử Dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `PyTorch` | 2.x | Neural Network, GPU training |
| `gymnasium` | latest | RL environment interface |
| `flappy-bird-gymnasium` | latest | Môi trường game Flappy Bird |
| `mediapipe` | 0.10.x | Hand landmark detection |
| `opencv-python` | 4.x | Computer Vision, webcam |
| `numpy` | latest | Tính toán số học |
| `matplotlib` | latest | Visualization, plot |
| `tqdm` | latest | Progress bar |

### Tài Nguyên Học Thêm

- **Sách**: *Reinforcement Learning: An Introduction* — Sutton & Barto (2018) — [miễn phí online](http://incompleteideas.net/book/the-book.html)
- **Course**: David Silver's RL Course (DeepMind/UCL) — [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- **Blog**: Lilian Weng — [lilianweng.github.io](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- **MediaPipe Docs**: [developers.google.com/mediapipe](https://developers.google.com/mediapipe)

---

*Tài liệu này được tổng hợp từ project Flappy Bird AI — Reinforcement Learning + Computer Vision.*
*Cập nhật lần cuối: 2026-05-31*

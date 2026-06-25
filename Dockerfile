FROM python:3.10-slim

# Tránh các prompt hỏi (interactive) khi cài đặt apt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và Pygame
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    xvfb \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements vào trước để tối ưu Docker cache
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Lệnh mặc định khi chạy container
# Sử dụng xvfb-run để giả lập màn hình ảo (tránh lỗi Pygame không tìm thấy video device khi chạy trong Docker)
CMD ["xvfb-run", "-s", "-screen 0 640x480x24", "python", "src/main.py", "--train", "--episodes", "500"]

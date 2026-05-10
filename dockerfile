# =========================
# Base: PyTorch + CUDA (devel)
# =========================
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# =========================
# 작업 디렉토리
# =========================
WORKDIR /app

# =========================
# 시스템 패키지 (devel이지만 최소 추가)
# =========================
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python 패키지 설치
# =========================
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =========================
# 기본 실행
# =========================
CMD ["bash"]
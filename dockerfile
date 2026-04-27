FROM j1seon/supercoding:v1

WORKDIR /workspace

# 1. 시스템 패키지 및 최신 pip 업데이트 (의존성 해결 능력 향상)
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir --upgrade pip

# 2. 의존성 지옥을 피하기 위해 dvc를 먼저 설치
# dvc 버전을 특정(예: 3.0.0 이상)하면 pip가 길을 찾기 훨씬 쉬워집니다.
RUN pip install --no-cache-dir "dvc[s3]>=3.0.0" dagshub

# 3. 나머지 패키지 설치
COPY requirements.txt /workspace/requirements.txt
# requirements.txt에서 dvc[s3]는 지우거나 주석 처리해도 됩니다.
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace
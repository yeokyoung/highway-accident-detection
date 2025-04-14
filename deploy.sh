#!/bin/bash
set -e

echo "고속도로 CCTV 사고 감지 시스템 배포 스크립트"
echo "----------------------------------------"

# 가상환경 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 모델 다운로드
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mkdir -p models
cp yolov8n.pt models/

# Docker 이미지 빌드
docker build -t accident-detection:latest .

# Docker 컨테이너 실행
docker run -d -p 8000:8000   -e DATABASE_URL=sqlite:///./accidents.db   -e DEBUG_MODE=False   -e HEADLESS_MODE=True   --name accident-detection   accident-detection:latest

echo "배포 완료!"

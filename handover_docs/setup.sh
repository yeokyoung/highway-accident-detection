#!/bin/bash
set -e

echo "고속도로 CCTV 사고 감지 시스템 설치 스크립트"
echo "----------------------------------------"

# 패키지 설치
echo "필수 패키지 설치 중..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 디렉토리 생성
echo "디렉토리 구조 생성 중..."
mkdir -p static/html static/accident_images models videos

# 모델 다운로드
echo "YOLO 모델 다운로드 중..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cp yolov8n.pt models/

echo "설치 완료!"
echo "서버를 시작하려면: source venv/bin/activate && python -m app.main"

# 고속도로 CCTV 사고 감지 시스템

## 개요
고속도로 CCTV 영상 내 자동차 사고 상황 감지를 위한 인공지능 모델 기반 시스템

## 기능
- 실시간 CCTV 영상 모니터링
- 딥러닝 기반 차량 감지 및 추적 (YOLOv8 + DeepSort)
- 사고 상황 자동 감지 및 알림
- 웹소켓 기반 실시간 알림 시스템
- RESTful API 제공

## 시스템 구조
- FastAPI 기반 백엔드 서버
- SQLite 데이터베이스
- 웹소켓 실시간 통신
- 사고 감지 인공지능 모듈

## 설치 방법
```bash
# 환경 설정
git clone [저장소 URL]
cd accident-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 모델 다운로드
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mkdir -p models
cp yolov8n.pt models/

# 서버 실행
source venv/bin/activate
python -m app.main

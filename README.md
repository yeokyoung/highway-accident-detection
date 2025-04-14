# 고속도로 CCTV 사고 감지 시스템

## 프로젝트 개요
AI 기반 고속도로 CCTV 사고 감지 및 모니터링 시스템

## 주요 기능
- 실시간 CCTV 영상 처리
- YOLO 및 DeepSort 기반 차량 감지
- 자동 사고 감지 알고리즘
- 웹소켓 실시간 알림

## 기술 스택
- Python
- FastAPI
- YOLO
- DeepSort
- SQLite

## 설치 및 실행
```bash
git clone https://github.com/yeokyoung/highway-accident-detection.git
cd highway-accident-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## 라이선스
MIT License

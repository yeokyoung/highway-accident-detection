#config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # 기본 설정
    APP_NAME: str = "고속도로 사고 감지 시스템"
    DEBUG_MODE: bool = False
    HEADLESS_MODE: bool = True  # 서버 환경에서는 True로 설정
    
    # 데이터베이스 설정
    DATABASE_URL: str = "sqlite:///./accidents.db"
    
    # API 키
    ITS_API_KEY: str = "6104f223eb4d4821b44f23d9a1616dbe"
    
    # CCTV 설정
    CCTV_REGIONS: List[str] = []  # 지역별 CCTV 필터링 (비어있으면 모든 CCTV)
    DETECTION_INTERVAL: int = 1  # 프레임 처리 간격 (초)
    
    # 모델 설정
    MODEL_PATH: str = "models/yolov8s.pt"
    CONFIDENCE_THRESHOLD: float = 0.3
    VEHICLE_CLASSES: List[int] = [2, 5, 7]  # 자동차, 버스, 트럭
    
    # 로컬 비디오 경로 (API 연결 실패 시 사용)
    LOCAL_VIDEO_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "videos", "videos.mp4")
    
    # 웹소켓 설정
    WEBSOCKET_PATH: str = "/ws"
    
    # 사고 감지 설정
    MIN_ACCIDENT_DURATION: int = 3  # 사고로 간주하기 위한 최소 시간 (초)
    ACCIDENT_DETECTION_THRESHOLD: float = 0.7  # 사고 확률 임계값
    
    class Config:
        env_file = ".env"

settings = Settings()

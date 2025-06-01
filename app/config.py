#config.py
import os
import json
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any

class Settings(BaseSettings):
    # 기본 설정 (기존 코드 유지)
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
    LOCAL_VIDEO_PATH: str = "/mnt/c/Users/user/Desktop/videos.mp4"
    # 웹소켓 설정
    WEBSOCKET_PATH: str = "/ws"
    
    # 사고 감지 설정
    MIN_ACCIDENT_DURATION: int = 3  # 사고로 간주하기 위한 최소 시간 (초)
    ACCIDENT_DETECTION_THRESHOLD: float = 0.7  # 사고 확률 임계값
    
    # 추가: 심각도 계산 파라미터 (동적 설정)
    SEVERITY_PARAMS: Dict[str, Any] = {
        # 기본 심각도 계산 파라미터
        "BASE_SCORE": 0.5,
        "TIME_FACTOR_COEF": 0.02,
        "TIME_FACTOR_POWER": 0.7,
        "NEARBY_FACTOR_COEF": 0.1,
        "SIZE_LARGE_THRESHOLD": 30000,
        "SIZE_MEDIUM_THRESHOLD": 15000,
        "SIZE_LARGE_FACTOR": 0.1,
        "SIZE_MEDIUM_FACTOR": 0.05,
        "ROADSIDE_FACTOR": 0.05,
        "INTERSECTION_FACTOR": 0.1,
        "AVOIDANCE_FACTOR": 0.1
    }
    
    # 추가: 다양한 감지 기준 세분화
    DETECTION_THRESHOLDS: Dict[str, Any] = {
        # 정지 상태 감지 세분화
        "STATIONARY": {
            "SHORT_STOP_DURATION": 3,   # 짧은 정지 (초)
            "MEDIUM_STOP_DURATION": 10, # 중간 정지 (초)
            "LONG_STOP_DURATION": 30,   # 긴 정지 (초)
            "DISTANCE_THRESHOLD": 6     # 정지 상태 속도 임계값 (픽셀)
        },
        # 급감속 감지 세분화
        "DECELERATION": {
            "SLIGHT_THRESHOLD": 15,     # 약한 감속
            "MEDIUM_THRESHOLD": 30,     # 중간 감속
            "SEVERE_THRESHOLD": 50      # 심한 감속
        },
        # 비정상 움직임 감지 세분화
        "MOVEMENT": {
            "SLIGHT_DEVIATION": 15,     # 약한 이탈
            "MEDIUM_DEVIATION": 25,     # 중간 이탈
            "SEVERE_DEVIATION": 40      # 심한 이탈
        },
        # 주변 차량 분석 세분화
        "PROXIMITY": {
            "CLOSE_RANGE": 100,         # 매우 가까운 거리 (픽셀)
            "MEDIUM_RANGE": 200,        # 중간 거리 (픽셀)
            "FAR_RANGE": 350            # 먼 거리 (픽셀)
        }
    }
    
    # 파라미터 설정 파일 경로
    SETTINGS_FILE: str = "detection_params.json"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 저장된 파라미터 설정 로드
        self.load_params()
    
    def load_params(self):
        """저장된 파라미터 설정 로드"""
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r') as f:
                    saved_params = json.load(f)
                    # 저장된 파라미터로 업데이트
                    if "SEVERITY_PARAMS" in saved_params:
                        self.SEVERITY_PARAMS.update(saved_params["SEVERITY_PARAMS"])
                    if "DETECTION_THRESHOLDS" in saved_params:
                        for category, values in saved_params["DETECTION_THRESHOLDS"].items():
                            if category in self.DETECTION_THRESHOLDS:
                                self.DETECTION_THRESHOLDS[category].update(values)
            except Exception as e:
                print(f"파라미터 설정 파일 로드 오류: {str(e)}")
    
    def save_params(self):
        """현재 파라미터 설정 저장"""
        try:
            params_to_save = {
                "SEVERITY_PARAMS": self.SEVERITY_PARAMS,
                "DETECTION_THRESHOLDS": self.DETECTION_THRESHOLDS
            }
            
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(params_to_save, f, indent=4)
                
            print(f"파라미터 설정이 {self.SETTINGS_FILE}에 저장되었습니다.")
        except Exception as e:
            print(f"파라미터 설정 파일 저장 오류: {str(e)}")
    
    def update_severity_params(self, params: Dict[str, Any]):
        """심각도 계산 파라미터 업데이트"""
        self.SEVERITY_PARAMS.update(params)
        self.save_params()
    
    def update_detection_thresholds(self, category: str, thresholds: Dict[str, Any]):
        """감지 임계값 업데이트"""
        if category in self.DETECTION_THRESHOLDS:
            self.DETECTION_THRESHOLDS[category].update(thresholds)
            self.save_params()
        else:
            raise ValueError(f"Unknown threshold category: {category}")
    
    class Config:
        env_file = ".env"

settings = Settings()

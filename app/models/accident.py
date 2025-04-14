from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.database import Base

class Accident(Base):
    __tablename__ = "accidents"

    id = Column(Integer, primary_key=True, index=True)
    cctv_id = Column(String, index=True)
    cctv_name = Column(String)
    detected_at = Column(DateTime, default=func.now())
    location = Column(String)
    accident_type = Column(String)  # 추돌, 전복, 화재 등
    severity = Column(Float)  # 심각도 점수 (0-1)
    vehicles_involved = Column(Integer)  # 관련 차량 수
    image_path = Column(String, nullable=True)  # 사고 이미지 캡처 경로
    notification_sent = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    details = Column(JSON, nullable=True)  # 추가 상세 정보 (JSON 형식)

# Pydantic 모델 (API 요청/응답용)
class AccidentBase(BaseModel):
    cctv_id: str
    cctv_name: str
    location: str
    accident_type: str
    severity: float
    vehicles_involved: int
    details: Optional[Dict[str, Any]] = None

class AccidentCreate(AccidentBase):
    pass

class AccidentUpdate(BaseModel):
    resolved: Optional[bool] = None
    resolved_at: Optional[datetime] = None
    notification_sent: Optional[bool] = None
    details: Optional[Dict[str, Any]] = None

class AccidentResponse(AccidentBase):
    id: int
    detected_at: datetime
    image_path: Optional[str] = None
    notification_sent: bool
    resolved: bool
    resolved_at: Optional[datetime] = None

    class Config:
        orm_mode = True
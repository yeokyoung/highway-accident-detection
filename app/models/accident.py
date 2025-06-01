from sqlalchemy import Boolean, Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.database import Base

class Accident(Base):
    __tablename__ = "accidents"

    id = Column(Integer, primary_key=True, index=True)
    cctv_id = Column(String, index=True)
    cctv_name = Column(String)
    detected_at = Column(DateTime, default=func.now())
    location = Column(String)
    accident_type = Column(String)
    vehicles_involved = Column(Integer)
    image_path = Column(String, nullable=True)
    notification_sent = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    details = Column(JSON, nullable=True)

class AccidentBase(BaseModel):
    cctv_id: str
    cctv_name: str
    location: str
    accident_type: str
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
        from_attributes = True

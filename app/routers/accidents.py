# accidents.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import base64
import json
from app.models.accident import Accident, AccidentCreate, AccidentResponse, AccidentUpdate
from app.database import get_db
from app.websocket import notify_clients
from app.config import settings
# from ..services.accident_detection import run_sync  # 이 줄을 주석 처리

router = APIRouter(
    prefix="/accidents",
    tags=["accidents"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=AccidentResponse)
async def create_accident(
    accident: AccidentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """사고 정보를 새로 등록합니다."""
    db_accident = Accident(**accident.dict())
    db.add(db_accident)
    db.commit()
    db.refresh(db_accident)
    
    # 웹소켓을 통해 클라이언트에 알림
    background_tasks.add_task(notify_clients, {
        "type": "accident_detected",
        "data": {
            "id": db_accident.id,
            "cctv_name": db_accident.cctv_name,
            "location": db_accident.location,
            "detected_at": db_accident.detected_at.isoformat()
        }
    })
    
    return db_accident

@router.get("/", response_model=List[AccidentResponse])
async def get_accidents(
    skip: int = 0,
    limit: int = 100,
    cctv_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """사고 목록을 조회합니다."""
    query = db.query(Accident)
    
    if cctv_id:
        query = query.filter(Accident.cctv_id == cctv_id)
    
    return query.order_by(Accident.detected_at.desc()).offset(skip).limit(limit).all()

@router.get("/{accident_id}", response_model=AccidentResponse)
async def get_accident(
    accident_id: int,
    db: Session = Depends(get_db)
):
    """특정 사고 정보를 조회합니다."""
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if not db_accident:
        raise HTTPException(status_code=404, detail="사고 정보를 찾을 수 없습니다.")
    return db_accident

@router.put("/{accident_id}", response_model=AccidentResponse)
async def update_accident(
    accident_id: int,
    accident_update: AccidentUpdate,
    db: Session = Depends(get_db)
):
    """사고 정보를 업데이트합니다."""
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if not db_accident:
        raise HTTPException(status_code=404, detail="사고 정보를 찾을 수 없습니다.")
    
    update_data = accident_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_accident, key, value)
    
    db.add(db_accident)
    db.commit()
    db.refresh(db_accident)
    return db_accident

@router.delete("/{accident_id}")
async def delete_accident(
    accident_id: int,
    db: Session = Depends(get_db)
):
    """사고 정보를 삭제합니다."""
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if not db_accident:
        raise HTTPException(status_code=404, detail="사고 정보를 찾을 수 없습니다.")
    
    db.delete(db_accident)
    db.commit()
    return {"message": "사고 정보가 삭제되었습니다."}

@router.post("/{accident_id}/image")
async def upload_accident_image(
    accident_id: int,
    image_data: str,
    db: Session = Depends(get_db)
):
    """사고 이미지를 업로드합니다."""
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if not db_accident:
        raise HTTPException(status_code=404, detail="사고 정보를 찾을 수 없습니다.")
    
    try:
        # 정적 파일 디렉토리 확인 및 생성
        image_dir = os.path.join("static", "accident_images")
        os.makedirs(image_dir, exist_ok=True)
        
        # 이미지 파일 저장
        image_filename = f"accident_{accident_id}_{int(time.time())}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        # Base64 디코딩 및 파일 저장
        image_data = base64.b64decode(image_data)
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # 사고 정보 업데이트
        db_accident.image_path = f"/static/accident_images/{image_filename}"
        db.add(db_accident)
        db.commit()
        
        return {"message": "이미지 업로드 성공", "image_path": db_accident.image_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")
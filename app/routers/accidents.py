from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import os
import base64
from app.models.accident import Accident, AccidentCreate, AccidentResponse, AccidentUpdate
from app.database import get_db
from app.websocket import notify_clients

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
    """
    사고 정보를 새로 등록합니다. CCTV 시스템에서 사고 감지 시 호출됩니다.
    """
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
            "accident_type": db_accident.accident_type,
            "severity": db_accident.severity,
            "detected_at": db_accident.detected_at.isoformat()
        }
    })
    
    return db_accident

@router.get("/", response_model=List[AccidentResponse])
def get_accidents(
    skip: int = 0, 
    limit: int = 100, 
    resolved: Optional[bool] = None,
    cctv_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    사고 목록을 조회합니다. 필터링 옵션을 제공합니다.
    """
    query = db.query(Accident)
    
    if resolved is not None:
        query = query.filter(Accident.resolved == resolved)
    
    if cctv_id:
        query = query.filter(Accident.cctv_id == cctv_id)
    
    return query.order_by(Accident.detected_at.desc()).offset(skip).limit(limit).all()

@router.get("/{accident_id}", response_model=AccidentResponse)
def get_accident(accident_id: int, db: Session = Depends(get_db)):
    """
    특정 ID의 사고 정보를 조회합니다.
    """
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if db_accident is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    return db_accident

@router.patch("/{accident_id}", response_model=AccidentResponse)
async def update_accident(
    accident_id: int, 
    accident_update: AccidentUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    사고 정보를 업데이트합니다. 사고 해결 여부 등을 업데이트할 수 있습니다.
    """
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if db_accident is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    
    update_data = accident_update.dict(exclude_unset=True)
    
    # resolved가 True로 설정되면 resolved_at 자동 설정
    if update_data.get("resolved") and not db_accident.resolved:
        update_data["resolved_at"] = datetime.now()
    
    for key, value in update_data.items():
        setattr(db_accident, key, value)
    
    db.commit()
    db.refresh(db_accident)
    
    # 웹소켓을 통해 클라이언트에 알림 (사고 상태 변경)
    if "resolved" in update_data:
        background_tasks.add_task(notify_clients, {
            "type": "accident_status_changed",
            "data": {
                "id": db_accident.id,
                "resolved": db_accident.resolved,
                "resolved_at": db_accident.resolved_at.isoformat() if db_accident.resolved_at else None
            }
        })
    
    return db_accident

@router.delete("/{accident_id}", response_model=AccidentResponse)
def delete_accident(accident_id: int, db: Session = Depends(get_db)):
    """
    사고 정보를 삭제합니다.
    """
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if db_accident is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    
    db.delete(db_accident)
    db.commit()
    
    return db_accident

@router.post("/{accident_id}/image", response_model=AccidentResponse)
def upload_accident_image(
    accident_id: int,
    image_data: str,  # Base64로 인코딩된 이미지
    db: Session = Depends(get_db)
):
    """
    사고 이미지를 업로드합니다.
    """
    db_accident = db.query(Accident).filter(Accident.id == accident_id).first()
    if db_accident is None:
        raise HTTPException(status_code=404, detail="Accident not found")
    
    # 이미지 저장 디렉토리 확인 및 생성
    img_dir = os.path.join("static", "accident_images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Base64 디코딩 및 이미지 저장
    image_data = image_data.split(",")[1] if "," in image_data else image_data
    image_bytes = base64.b64decode(image_data)
    
    filename = f"{accident_id}_{int(datetime.now().timestamp())}.jpg"
    file_path = os.path.join(img_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    
    # 데이터베이스 업데이트
    db_accident.image_path = f"/static/accident_images/{filename}"
    db.commit()
    db.refresh(db_accident)
    
    return db_accident
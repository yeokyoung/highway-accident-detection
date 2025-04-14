from fastapi import APIRouter, HTTPException, BackgroundTasks
import requests
import json
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.accident_detection import start_detection, stop_detection
from app.services.cctv_manager import cctv_manager

router = APIRouter(
    prefix="/cctv",
    tags=["cctv"],
    responses={404: {"description": "Not found"}},
)

# 실행 중인 CCTV 모니터링 작업 추적
active_monitors = {}

@router.get("/streams")
async def get_cctv_streams(region: Optional[str] = None):
    """
    고속도로 CCTV 스트림 목록을 가져옵니다.
    """
    url = (
        "https://openapi.its.go.kr:9443/cctvInfo"
        f"?apiKey=6104f223eb4d4821b44f23d9a1616dbe"
        "&type=ex&cctvType=1&minX=126.8&maxX=127.89&minY=34.9&maxY=35.5&getType=json"
    )
    
    try:
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            raise HTTPException(status_code=503, detail="ITS API 서비스를 사용할 수 없습니다.")
        
        data = response.json()
        if "response" not in data or "data" not in data["response"]:
            raise HTTPException(status_code=503, detail="ITS API에서 잘못된 응답 형식이 반환되었습니다.")
        
        cctv_list = data["response"]["data"]
        
        # 지역 필터링
        if region:
            cctv_list = [cctv for cctv in cctv_list if region in cctv.get("cctvname", "")]
        
        # 응답 형식 정리
        result = []
        for i, cctv in enumerate(cctv_list):
            # 고유 ID 생성 (원래 ID가 없으므로 인덱스 기반 ID 생성)
            cctv_id = str(i) if not cctv.get("cctvid") else cctv.get("cctvid")
            
            result.append({
                "id": cctv_id,
                "name": cctv.get("cctvname", ""),
                "url": cctv.get("cctvurl", ""),
                "location": cctv.get("addressjibun", "") or cctv.get("addressdoro", ""),
                "coordinates": {
                    "latitude": cctv.get("coordy", 0),
                    "longitude": cctv.get("coordx", 0)
                }
            })
            
        # 테스트용 CCTV 추가
        result.append({
            "id": "test",
            "name": "테스트 CCTV",
            "url": "",  # 비어있으면 로컬 비디오 사용
            "location": "테스트 위치",
            "coordinates": {
                "latitude": 0,
                "longitude": 0
            }
        })
        
        return result
    
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="ITS API에 연결할 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

@router.post("/monitor/{cctv_id}")
async def start_monitoring(cctv_id: str, background_tasks: BackgroundTasks):
    """
    특정 CCTV 스트림의 모니터링을 시작합니다.
    """
    if cctv_id in active_monitors:
        raise HTTPException(status_code=400, detail=f"CCTV {cctv_id}는 이미 모니터링 중입니다.")
    
    # CCTV 스트림 URL 가져오기
    try:
        streams = await get_cctv_streams()
        cctv = next((c for c in streams if c["id"] == cctv_id), None)
        
        if not cctv:
            # 테스트 ID인 경우 테스트 CCTV 정보 사용
            if cctv_id == "test" or cctv_id.startswith("test"):
                cctv = {
                    "id": cctv_id,
                    "name": f"테스트 CCTV {cctv_id}",
                    "url": "",  # 비어있으면 로컬 비디오 사용
                    "location": "테스트 위치",
                    "coordinates": {
                        "latitude": 0,
                        "longitude": 0
                    }
                }
            else:
                raise HTTPException(status_code=404, detail=f"CCTV ID {cctv_id}를 찾을 수 없습니다.")
        
        # 모니터링 시작 (백그라운드에서 실행)
        active_monitors[cctv_id] = {
            "cctv_info": cctv,
            "status": "starting"
        }
        
        background_tasks.add_task(start_detection, cctv, on_complete=lambda: active_monitors.pop(cctv_id, None))
        
        return {"message": f"CCTV {cctv_id} 모니터링을 시작했습니다.", "cctv_info": cctv}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모니터링 시작 오류: {str(e)}")

@router.post("/monitor")
async def start_multiple_monitoring(cctv_ids: List[str], background_tasks: BackgroundTasks):
    """
    여러 CCTV 스트림의 모니터링을 동시에 시작합니다.
    """
    results = []
    
    for cctv_id in cctv_ids:
        try:
            result = await start_monitoring(cctv_id, background_tasks)
            results.append({"cctv_id": cctv_id, "status": "started", "info": result})
        except HTTPException as e:
            results.append({"cctv_id": cctv_id, "status": "error", "detail": e.detail})
    
    return results

@router.delete("/monitor/{cctv_id}")
async def stop_monitoring(cctv_id: str):
    """
    특정 CCTV 스트림의 모니터링을 중지합니다.
    """
    if cctv_id not in active_monitors:
        raise HTTPException(status_code=404, detail=f"CCTV {cctv_id}는 현재 모니터링되지 않고 있습니다.")
    
    try:
        stop_detection(cctv_id)
        active_monitors.pop(cctv_id, None)
        return {"message": f"CCTV {cctv_id} 모니터링을 중지했습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모니터링 중지 오류: {str(e)}")

@router.get("/active")
async def get_active_monitors():
    """
    현재 활성 상태인 모니터링 작업 목록을 반환합니다.
    """
    return active_monitors
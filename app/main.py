#main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from app.database import engine, Base
from app.routers import accidents, cctv
from app.websocket import manager
from app.config import settings
import uvicorn

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# 정적 파일 디렉토리 생성
os.makedirs(os.path.join("static", "accident_images"), exist_ok=True)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME,
    description="고속도로 CCTV 영상 내 자동차 사고 상황 감지 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 환경에서는 명시적인 도메인으로 제한해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(accidents.router)
app.include_router(cctv.router)

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")), name="static")

@app.get("/")
async def root():
    """API 서버 상태 확인"""
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "endpoints": {
            "accidents": "/accidents",
            "cctv": "/cctv/streams",
            "websocket": settings.WEBSOCKET_PATH
        }
    }

@app.websocket(settings.WEBSOCKET_PATH)
async def websocket_endpoint(websocket: WebSocket):
    """
    웹소켓 연결 처리. 사고 알림 등을 실시간으로 클라이언트에 전달합니다.
    """
    await manager.connect(websocket)
    try:
        # 연결 성공 메시지
        await websocket.send_json({
            "type": "connection_established",
            "message": "웹소켓 연결이 설정되었습니다."
        })
        
        while True:
            # 클라이언트로부터 메시지 수신 (필요한 경우 활용)
            data = await websocket.receive_text()
            # 에코 (예시)
            await websocket.send_json({
                "type": "echo",
                "data": data
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    # 개발용 서버 실행
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

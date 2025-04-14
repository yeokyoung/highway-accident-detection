#websocket.py
from fastapi import WebSocket, WebSocketDisconnect
import json
from typing import Dict, List, Any

# 클라이언트 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

# 연결 관리자 인스턴스 생성
manager = ConnectionManager()

# 다른 서비스에서 호출하여 클라이언트에 알림을 보내는 함수
async def notify_clients(message: Dict[str, Any]):
    await manager.broadcast(message)
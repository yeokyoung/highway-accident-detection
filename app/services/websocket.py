# websocket.py
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active = []  # WebSocket 연결 목록

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)
        print(f"새 웹소켓 연결 수락: 현재 {len(self.active)}개 연결")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)
            print(f"웹소켓 연결 종료: 현재 {len(self.active)}개 연결")

    async def broadcast(self, message: dict):
        print(f"브로드캐스트 메시지 전송: {len(self.active)}개 연결에 {message['type']} 유형 메시지")
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception as e:
                print(f"웹소켓 전송 오류: {str(e)}")
                disconnected.append(ws)
        
        # 연결이 끊긴 웹소켓 제거
        for ws in disconnected:
            self.disconnect(ws)

manager = ConnectionManager()

async def notify_clients(message: dict):
    """모든 연결된 클라이언트에게 메시지를 전송합니다."""
    await manager.broadcast(message)
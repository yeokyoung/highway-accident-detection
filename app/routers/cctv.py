#app/routers/cctv.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
import requests
import json
import os
from typing import List, Dict, Any, Optional
from fastapi.responses import HTMLResponse, FileResponse
from app.config import settings
from app.services.accident_detection import start_detection, stop_detection, get_test_data
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

# 추가: 테스트 모니터링 엔드포인트 (GET 메서드)
@router.get("/monitor/test")
async def test_monitor():
    """
    모니터링 테스트 페이지를 반환합니다.
    """
    # HTML 파일 반환
    try:
        # static 폴더에서 monitor.html 파일 찾기
        html_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "monitor.html")
        if os.path.exists(html_file):
            return FileResponse(html_file)
        else:
            # HTML 파일이 없으면 대체 HTML 생성
            return generate_monitor_html()
    except Exception as e:
        # 오류 발생시 테스트 데이터 반환
        test_data = get_test_data()
        return test_data

def generate_monitor_html():
    """
    모니터링 페이지용 HTML을 생성합니다.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>고속도로 사고 감지 실시간 모니터링</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            h1 { color: #333; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .btn { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; margin-right: 10px; }
            .btn:hover { background-color: #45a049; }
            .btn-danger { background-color: #f44336; }
            .btn-danger:hover { background-color: #d32f2f; }
            .status { display: flex; align-items: center; }
            .status-indicator { width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
            .connected { background-color: green; }
            .disconnected { background-color: red; }
            .alerts { margin-top: 20px; }
            .alert-item { background-color: #ffdddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
            .alert-title { font-weight: bold; color: #d33; margin: 0 0 5px 0; }
            .alert-details { margin: 5px 0; }
            .log-window { background-color: #000; color: #fff; padding: 10px; height: 300px; overflow-y: auto; font-family: monospace; margin-left: 20px; }
            .control-panel { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>고속도로 사고 감지 실시간 모니터링</h1>
        
        <div class="control-panel">
            <button id="connect-btn" class="btn">웹소켓 연결</button>
            <button id="disconnect-btn" class="btn btn-danger">연결 끊기</button>
            <button id="test-alert-btn" class="btn">테스트 알림 생성</button>
        </div>
        
        <div class="header">
            <div class="status">
                <div id="status-indicator" class="status-indicator disconnected"></div>
                <span id="status-text">상태: 연결 끊김</span>
            </div>
        </div>
        
        <div class="container" style="display: flex;">
            <div class="alerts" style="flex: 1;">
                <h2>실시간 알림</h2>
                <div id="alerts-container">
                    <!-- 여기에 알림이 추가됩니다 -->
                </div>
            </div>
            
            <div class="log-window" style="flex: 1;">
                <div id="log-container">
                    <!-- 여기에 로그가 추가됩니다 -->
                </div>
            </div>
        </div>
        
        <script>
            let socket;
            let connected = false;  // 초기 상태는 연결 끊김
            const connectBtn = document.getElementById('connect-btn');
            const disconnectBtn = document.getElementById('disconnect-btn');
            const testAlertBtn = document.getElementById('test-alert-btn');
            const statusIndicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');
            const alertsContainer = document.getElementById('alerts-container');
            const logContainer = document.getElementById('log-container');
            
            function connect() {
                if (connected) {
                    logMessage("이미 연결되어 있습니다.");
                    return;
                }
                
                // 웹소켓 연결
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                logMessage(`웹소켓 연결 시도: ${wsUrl}`);
                
                try {
                    socket = new WebSocket(wsUrl);
                    
                    socket.onopen = function(e) {
                        logMessage("웹소켓 연결 성공");
                        setConnectedStatus(true);
                    };
                    
                    socket.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        logMessage(`메시지 수신: ${JSON.stringify(data)}`);
                        
                        if (data.type === "accident_detected") {
                            addAlert(data.data);
                        }
                    };
                    
                    socket.onclose = function(event) {
                        logMessage("웹소켓 연결 종료");
                        setConnectedStatus(false);
                    };
                    
                    socket.onerror = function(error) {
                        logMessage(`웹소켓 오류 발생`);
                        setConnectedStatus(false);
                    };
                } catch (error) {
                    logMessage(`웹소켓 연결 실패: ${error.message}`);
                    setConnectedStatus(false);
                }
            }
            
            function disconnect() {
                if (!connected || !socket) {
                    logMessage("연결되어 있지 않습니다.");
                    return;
                }
                
                logMessage("웹소켓 연결 종료 중...");
                socket.close();
                setConnectedStatus(false);
            }
            
            function setConnectedStatus(isConnected) {
                connected = isConnected;
                if (isConnected) {
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                    statusIndicator.className = "status-indicator connected";
                    statusText.textContent = "상태: 연결됨";
                    logMessage("연결 상태로 변경됨");
                } else {
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                    statusIndicator.className = "status-indicator disconnected";
                    statusText.textContent = "상태: 연결 끊김";
                    logMessage("연결 끊김 상태로 변경됨");
                    socket = null;
                }
            }
            
            function addAlert(data) {
                const alertItem = document.createElement('div');
                alertItem.className = 'alert-item';
                
                const now = new Date();
                const timestamp = data.detected_at ? new Date(data.detected_at) : now;
                const formattedTime = `${timestamp.getFullYear()}. ${timestamp.getMonth() + 1}. ${timestamp.getDate()}. ${timestamp.getHours()}:${timestamp.getMinutes()}:${timestamp.getSeconds()}`;
                
                alertItem.innerHTML = `
                    <div class="alert-title">사고 감지!</div>
                    <div class="alert-details">시간: ${formattedTime}</div>
                    <div class="alert-details">위치: ${data.location || '테스트 위치'}</div>
                    <div class="alert-details">CCTV: ${data.cctv_name || '테스트 CCTV'}</div>
                `;
                
                alertsContainer.insertBefore(alertItem, alertsContainer.firstChild);
                logMessage(`새로운 알림 추가: ${data.id}`);
            }
            
            function createTestAlert() {
                const testData = {
                    id: "test_accident_" + Date.now(),
                    location: "테스트 위치",
                    cctv_name: "테스트 CCTV",
                    detected_at: new Date().toISOString()
                };
                
                addAlert(testData);
                logMessage("테스트 알림이 생성되었습니다.");
            }
            
            function logMessage(message) {
                const now = new Date();
                const timestamp = `${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}.${now.getMilliseconds()}`;
                
                const logItem = document.createElement('div');
                logItem.textContent = `[${timestamp}] ${message}`;
                logContainer.appendChild(logItem);
                logContainer.scrollTop = logContainer.scrollHeight;
            }
            
            // 버튼 이벤트 리스너
            connectBtn.addEventListener('click', connect);
            disconnectBtn.addEventListener('click', disconnect);
            testAlertBtn.addEventListener('click', createTestAlert);
            
            // 초기 상태 설정
            setConnectedStatus(false);
            disconnectBtn.disabled = true;
            
            // 페이지 로드 시 로그 메시지
            logMessage("페이지가 로드되었습니다. '웹소켓 연결' 버튼을 클릭하여 연결을 시작하세요.");
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)
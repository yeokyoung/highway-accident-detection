#cctv_manager.py
from typing import Dict, List, Any, Optional
import requests
import asyncio
import time
from app.config import settings

# cctv_manager.py에 상황 인식 기능 추가

class CCTVManager:
    def __init__(self):
        # 기존 초기화 코드 유지
        self.active_streams = {}
        self.stream_info = {}
        
        # 카메라 상태 및 상황 관리 추가
        self.camera_states = {}  # 카메라별 상태 정보 저장
    
    async def analyze_camera_state(self, cctv_id, frame):
        """카메라 상태(줌 레벨, 움직임 등) 분석"""
        if cctv_id not in self.camera_states:
            self.camera_states[cctv_id] = {
                'prev_frame': None,
                'zoom_level': 1.0,
                'movement': 0.0,
                'scene_type': 'normal'
            }
        
        state = self.camera_states[cctv_id]
        
        # 이전 프레임이 있으면 움직임 분석
        if state['prev_frame'] is not None:
            # 카메라 움직임 감지
            movement = self._detect_camera_movement(state['prev_frame'], frame)
            state['movement'] = movement
            
            # 줌 레벨 변화 감지
            zoom_change = self._detect_zoom_change(state['prev_frame'], frame)
            state['zoom_level'] *= (1 + zoom_change)
            
            # 장면 유형 분류
            state['scene_type'] = self._classify_scene(frame, movement, state['zoom_level'])
        
        # 현재 프레임 저장
        state['prev_frame'] = frame.copy()
        
        return state
    
    def _detect_camera_movement(self, prev_frame, curr_frame):
        """카메라 움직임 감지 (옵티컬 플로우 활용)"""
        # 이미지 크기 축소로 계산 속도 향상
        prev_small = cv2.resize(prev_frame, (320, 180))
        curr_small = cv2.resize(curr_frame, (320, 180))
        
        # 그레이스케일 변환
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        
        # 옵티컬 플로우 계산
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # 움직임 크기 계산
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        
        return mean_magnitude
    
    def _detect_zoom_change(self, prev_frame, curr_frame):
        """줌 레벨 변화 감지"""
        # SIFT 또는 SURF 특징점 추출 및 매칭으로 구현 가능
        # 간단한 구현을 위해 더미 값 반환
        return 0.0
    
    def _classify_scene(self, frame, movement, zoom_level):
        """장면 유형 분류"""
        if movement > 5.0:
            return 'camera_moving'
        elif zoom_level > 1.5:
            return 'zoomed_in'
        else:
            return 'normal'
        
    async def get_cctv_streams(self, region: Optional[str] = None) -> List[Dict[str, Any]]:
        """ITS API에서 CCTV 스트림 목록 가져오기"""
        url = (
            "https://openapi.its.go.kr:9443/cctvInfo"
            f"?apiKey=6104f223eb4d4821b44f23d9a1616dbe"
            "&type=ex&cctvType=1&minX=126.8&maxX=127.89&minY=34.9&maxY=35.5&getType=json"
        )
        
        try:
            # API 호출 (SSL 검증 비활성화)
            response = requests.get(url, verify=False, timeout=10)
            
            if response.status_code != 200:
                print(f"API 오류: 상태 코드 {response.status_code}")
                return []
                
            data = response.json()
            
            if "response" not in data or "data" not in data["response"]:
                print("API 응답 형식 오류")
                return []
                
            cctv_list = data["response"]["data"]
            
            # 지역 필터링
            if region:
                cctv_list = [cctv for cctv in cctv_list if region in cctv.get("cctvname", "")]
                
            # 응답 형식 정리
            result = []
            for i, cctv in enumerate(cctv_list):
                # 고유 ID 생성 (원래 ID가 없으므로 인덱스 기반 ID 생성)
                cctv_id = str(i) if not cctv.get("cctvid") else cctv.get("cctvid")
                
                cctv_info = {
                    "id": cctv_id,
                    "name": cctv.get("cctvname", ""),
                    "url": cctv.get("cctvurl", ""),
                    "location": cctv.get("addressjibun", "") or cctv.get("addressdoro", ""),
                    "coordinates": {
                        "latitude": cctv.get("coordy", 0),
                        "longitude": cctv.get("coordx", 0)
                    }
                }
                
                result.append(cctv_info)
                
                # 정보 캐싱
                self.stream_info[cctv_id] = cctv_info
                
            # 테스트용 CCTV 추가 (만약 API에서 가져온 목록에 없다면)
            if "test" not in self.stream_info:
                test_cctv = {
                    "id": "test",
                    "name": "테스트 CCTV",
                    "url": "",  # 비어있으면 로컬 비디오 사용
                    "location": "테스트 위치",
                    "coordinates": {
                        "latitude": 0,
                        "longitude": 0
                    }
                }
                result.append(test_cctv)
                self.stream_info["test"] = test_cctv
                
            return result
                
        except requests.RequestException as e:
            print(f"API 요청 오류: {str(e)}")
            return []
        except Exception as e:
            print(f"CCTV 스트림 정보 가져오기 오류: {str(e)}")
            return []
            
    def get_stream_url(self, cctv_id: str) -> Optional[str]:
        """특정 CCTV ID의 스트림 URL 반환"""
        # 캐시된 정보에서 URL 가져오기
        if cctv_id in self.stream_info:
            return self.stream_info[cctv_id].get("url")
            
        # 캐시에 없으면 기본값 반환
        return None
        
    def get_cctv_info(self, cctv_id: str) -> Optional[Dict[str, Any]]:
        """특정 CCTV ID의 정보 반환"""
        # 테스트 CCTV ID 처리
        if cctv_id == "test" or cctv_id.startswith("test"):
            if cctv_id != "test" and cctv_id not in self.stream_info:
                # 새로운 테스트 CCTV 정보 생성
                self.stream_info[cctv_id] = {
                    "id": cctv_id,
                    "name": f"테스트 CCTV {cctv_id}",
                    "url": "",  # 비어있으면 로컬 비디오 사용
                    "location": "테스트 위치",
                    "coordinates": {
                        "latitude": 0,
                        "longitude": 0
                    }
                }
            return self.stream_info.get(cctv_id, self.stream_info.get("test"))
                
        return self.stream_info.get(cctv_id)

# 전역 인스턴스 생성
cctv_manager = CCTVManager()
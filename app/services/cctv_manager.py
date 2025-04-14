#cctv_manager.py
from typing import Dict, List, Any, Optional
import requests
import asyncio
import time
from app.config import settings

class CCTVManager:
    """CCTV 스트림 관리 클래스"""
    
    def __init__(self):
        self.active_streams = {}  # 활성 CCTV 스트림 추적
        self.stream_info = {}  # CCTV 정보 캐싱
        
        # 초기화 시 테스트 CCTV 정보 추가
        self.stream_info["test"] = {
            "id": "test",
            "name": "테스트 CCTV",
            "url": "",  # 비어있으면 로컬 비디오 사용
            "location": "테스트 위치",
            "coordinates": {
                "latitude": 0,
                "longitude": 0
            }
        }
        
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
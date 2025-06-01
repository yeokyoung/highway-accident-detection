# app/services/accident_detection.py
import cv2
import threading
import numpy as np
import time
import sys
import os
from collections import defaultdict
from typing import Dict, Any, Callable, Optional, List, Tuple
from datetime import datetime
import asyncio
import json

# model.py import (경로 추가)
sys.path.append('/home/kimyeokyoung/accident-detection')
from model import (
    model, tracker, detect_accident, 
    CONFIDENCE_THRESHOLD, iou, smooth_box,
    previous_boxes, recent_ids, 
    ACCELERATION_THRESHOLD, OVERLAP_IOU_THRESHOLD,
    STATIONARY_DISTANCE, STATIONARY_FRAMES,
    PERPENDICULAR_MOVEMENT_THRESHOLD, SUSPICION_SCORE_THRESHOLD,
    SUSPICION_DECAY, ID_TRANSFER_IOU_THRESHOLD, ID_TRANSFER_TIME_WINDOW
)

from app.config import settings
from app.websocket import notify_clients

# 활성 스레드 추적
detection_threads = {}
stop_events = {}  # CCTV별 stop_event

# 이미 보고된 사고 정보 저장 (중복 방지)
reported_accidents = {}
accident_expiry_time = 60  # 사고 기록 만료 시간(초)

class AccidentDetectionService:
    """사고 감지 서비스 - 조장님 AI 모델과 백엔드를 연결하는 브릿지"""
    
    def __init__(self):
        self.active_detections = {}
        self.reported_accidents = {}
        self.accident_expiry_time = 100
    
    async def start_detection(self, cctv_info: dict):
        """CCTV 사고 감지 시작"""
        cctv_id = cctv_info["id"]
        
        if cctv_id in detection_threads:
            print(f"⚠️ CCTV {cctv_id}는 이미 모니터링 중입니다.")
            return {"status": "already_running", "cctv_id": cctv_id}
        
        # CCTV별 stop_event 생성
        stop_events[cctv_id] = threading.Event()
        
        # 별도 스레드에서 감지 실행
        detection_thread = threading.Thread(
            target=self._detection_worker,
            args=(cctv_info,),
            daemon=True
        )
        
        detection_threads[cctv_id] = {
            "thread": detection_thread,
            "status": "running",
            "started_at": datetime.now()
        }
        
        detection_thread.start()
        await asyncio.sleep(1)
        
        return {"status": "started", "cctv_id": cctv_id}
    
    def _detection_worker(self, cctv_info: dict):
        """사고 감지 워커 - model.py를 실시간으로 실행"""
        cctv_id = cctv_info["id"]
        video_source = cctv_info.get("url", "")
        stop_event = stop_events[cctv_id]
        
        try:
            print(f"CCTV {cctv_id} 사고 감지 시작")
            
            # 비디오 소스 열기
            if not video_source:
                local_path = getattr(settings, 'LOCAL_VIDEO_PATH', 'test_video.mp4')
                cap = cv2.VideoCapture(local_path)
                print(f"로컬 비디오 사용: {local_path}")
            else:
                cap = cv2.VideoCapture(video_source)
                print(f"CCTV 스트림 연결: {video_source}")
            
            if not cap.isOpened():
                print(f"비디오 소스 열기 실패")
                return
            
            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = 1 / fps if fps > 0 else 1 / 30
            frame_count = 0
            
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if not video_source:  # 로컬 비디오 반복 재생
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("비디오 스트림 종료")
                        break
                
                frame_count += 1
                current_time = time.time()
                
                # 성능 최적화를 위해 3프레임마다 처리
                if frame_count % 3 != 0:
                    continue
                
                # 조장님의 차량 감지 로직 실행
                results = model(frame)[0]
                detections = []
                
                for det in results.boxes:
                    cls = int(det.cls)
                    conf = float(det.conf)
                    if cls in [2, 3, 5, 7] and conf > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        if w * h < 1000:
                            continue
                        aspect_ratio = w / (h + 1e-5)
                        if aspect_ratio < 0.2 or aspect_ratio > 5:
                            continue
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        detections.append(([x1, y1, w, h], conf, "vehicle"))
                
                # 중복 검출 필터링 (조장님 로직)
                filtered = []
                for i, det1 in enumerate(detections):
                    box1, conf1, _ = det1
                    x1, y1, w1, h1 = box1
                    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
                    keep = True
                    for j, det2 in enumerate(detections):
                        if i == j:
                            continue
                        box2, conf2, _ = det2
                        x2, y2, w2, h2 = box2
                        box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
                        if iou(box1_xyxy, box2_xyxy) > 0.7 and conf1 < conf2:
                            keep = False
                            break
                    if keep:
                        filtered.append(det1)
                detections = filtered
                
                # 추적 및 사고 감지 (조장님 로직)
                tracks = tracker.update_tracks(detections, frame=frame)
                track_results = {}
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    tid = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    if r <= l or b <= t:
                        continue
                    cx, cy = (l + r) // 2, (t + b) // 2
                    new_box = [l, t, r, b]
                    smoothed_box = smooth_box(previous_boxes[tid], new_box)
                    previous_boxes[tid] = smoothed_box
                    track_results[tid] = {
                        'bbox': smoothed_box,
                        'center': (cx, cy),
                        'speed': None
                    }
                    recent_ids[tid] = {'last_seen': frame_count, 'box': smoothed_box}
                
                # ID 계승 처리 (조장님 로직)
                for new_tid in list(track_results.keys()):
                    for old_tid, data in recent_ids.items():
                        if old_tid == new_tid:
                            continue
                        if 0 < frame_count - data['last_seen'] < ID_TRANSFER_TIME_WINDOW:
                            if iou(track_results[new_tid]['bbox'], data['box']) > ID_TRANSFER_IOU_THRESHOLD:
                                # suspicion_scores는 조장님 model.py에서 가져온 전역 변수
                                from model import suspicion_scores
                                suspicion_scores[new_tid] += suspicion_scores.get(old_tid, 0)
                                print(f"[ID 계승] ID {old_tid} → ID {new_tid}")
                                break
                
                # 조장님의 사고 감지 알고리즘 실행
                suspicious_ids = detect_accident(track_results, frame_count, frame_interval)
                
                # 만료된 사고 기록 정리
                self._cleanup_expired_accidents()
                
                # 사고 처리 및 알림 (백엔드 로직)
                for tid in suspicious_ids:
                    if tid in track_results:
                        accident_data = {
                            "track_id": tid,
                            "frame": frame_count,
                            "bbox": track_results[tid]['bbox'],
                            "location": track_results[tid]['center'],
                            "timestamp": current_time,
                            "cctv_info": cctv_info
                        }
                        
                        # 중복 방지 및 알림 전송
                        if self._should_send_alert(accident_data):
                            asyncio.run(self._send_accident_alert(accident_data))
                
                # 디버그 시각화 (선택사항)
                if getattr(settings, 'DEBUG_MODE', False) and not getattr(settings, 'HEADLESS_MODE', True):
                    for tid, data in track_results.items():
                        x1, y1, x2, y2 = data['bbox']
                        color = (0, 0, 255) if tid in suspicious_ids else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID {tid}"
                        if tid in suspicious_ids:
                            label += " ACCIDENT"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    cv2.imshow(f"CCTV {cctv_id}", cv2.resize(frame, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.033)  # ~30fps
            
        except Exception as e:
            print(f"Detection error for CCTV {cctv_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if cctv_id in detection_threads:
                del detection_threads[cctv_id]
            if cctv_id in stop_events:
                del stop_events[cctv_id]
            print(f"CCTV {cctv_id} 모니터링 종료")
    
    def _cleanup_expired_accidents(self):
        """만료된 사고 기록 정리"""
        current_time = time.time()
        expired = [k for k, v in self.reported_accidents.items() 
                  if current_time - v['timestamp'] > self.accident_expiry_time]
        for k in expired:
            del self.reported_accidents[k]
    
    def _should_send_alert(self, accident_data: dict) -> bool:
        """중복 사고 알림 방지"""
        location = accident_data["location"]
        
        # 같은 위치(50픽셀 이내) 사고 중복 확인
        for acc_id, data in self.reported_accidents.items():
            distance = np.sqrt((location[0] - data['location'][0])**2 + 
                             (location[1] - data['location'][1])**2)
            if distance < 150:
                return False
        # 같은 track_id는 30초 쿨다운           
        if any(data['track_id'] == accident_data['track_id'] and 
                current_time - data['timestamp'] < 30 
                for data in self.reported_accidents.values()):
            return False
                                  
        # 새로운 사고로 기록
        acc_id = f"accident_{accident_data['track_id']}_{int(time.time())}"
        self.reported_accidents[acc_id] = {
            'timestamp': time.time(),
            'location': location,
            'track_id': accident_data['track_id']
        }
        
        return True
    
    async def _send_accident_alert(self, accident_data: dict):
        """웹소켓으로 사고 알림 전송"""
        try:
            message = {
                "type": "accident_detected",
                "data": {
                    "id": f"accident_{accident_data['track_id']}_{int(time.time())}",
                    "cctv_name": accident_data['cctv_info'].get('name', 'Unknown'),
                    "location": accident_data['cctv_info'].get('location', 'Unknown'),
                    "detected_at": datetime.now().isoformat(),
                    "bbox": accident_data['bbox']
                }
            }
            
            await notify_clients(message)
            print(f"✅ 사고 알림 전송 성공: {message['data']['id']}")
        except Exception as e:
            print(f"❌ 알림 전송 실패: {str(e)}")
    
    def stop_detection(self, cctv_id: str):
        """사고 감지 중지"""
        if cctv_id not in detection_threads:
            print(f"⚠️ CCTV {cctv_id}는 현재 모니터링되지 않고 있습니다.")
            return {"status": "not_running", "cctv_id": cctv_id}
        
        # 해당 CCTV의 stop_event 설정
        if cctv_id in stop_events:
            stop_events[cctv_id].set()
        
        # 스레드 종료 대기
        thread_info = detection_threads[cctv_id]
        thread = thread_info["thread"]
        if thread.is_alive():
            thread.join(timeout=5)
        
        print(f"🛑 CCTV {cctv_id} 모니터링 중지")
        return {"status": "stopped", "cctv_id": cctv_id}
    
    def get_status(self):
        """현재 감지 상태 반환"""
        return {
            "active_cctvs": list(detection_threads.keys()),
            "total_reported_accidents": len(self.reported_accidents)
        }

# 전역 서비스 인스턴스
detection_service = AccidentDetectionService()

# 기존 함수들을 서비스 인스턴스로 래핑 (기존 코드 호환성 유지)
async def start_detection(cctv_info, on_complete=None):
    return await detection_service.start_detection(cctv_info)

def stop_detection(cctv_id):
    return detection_service.stop_detection(cctv_id)

def get_test_data():
    """테스트용 사고 감지 데이터를 반환합니다."""
    return {
        "status": "실시간 모니터링 테스트",
        "data": [
            {
                "id": "test_accident_1",
                "detected_at": datetime.now().isoformat(),
                "location": "테스트 위치 1",
                "cctv_name": "테스트 CCTV"
            },
            {
                "id": "test_accident_2",
                "detected_at": datetime.now().isoformat(),
                "location": "테스트 위치 2",
                "cctv_name": "테스트 CCTV"
            }
        ]
    }
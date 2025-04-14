#accident_detection.py
import cv2
import threading
import numpy as np
import time
import requests
import os
import base64
from typing import Dict, Any, Callable, Optional, List, Tuple
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from app.config import settings

# 활성 스레드 추적
detection_threads = {}
stop_event = threading.Event()

# 사고 감지 AI 모델 (YOLO 및 DeepSort 사용)
class AccidentDetectionModel:
    def __init__(self):
        # YOLO 모델 로딩
        self.model = YOLO(settings.MODEL_PATH)
        # DeepSort 트래커 초기화
        self.tracker = DeepSort(max_age=30)
        # 차량 클래스
        self.vehicle_classes = settings.VEHICLE_CLASSES
        # 신뢰도 임계값
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        # 사고 감지 모델 (여기서는 간단한 규칙 기반 로직 사용)
        # 실제로는 별도의 사고 감지 모델을 사용할 수 있습니다.
        
        # 차량 추적 정보 저장용
        self.vehicle_tracks = {}  # 차량 ID별 추적 정보
        self.stopped_vehicles = {}  # 정지된 차량 정보
        self.abnormal_movements = {}  # 비정상적 움직임 감지
        
    def detect_vehicles(self, frame):
        """프레임에서 차량 객체 감지"""
        results = self.model(frame)[0]
        detections = []
        
        for box in results.boxes:
            cid = int(box.cls[0])
            score = float(box.conf[0])
            if score < self.confidence_threshold or cid not in self.vehicle_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, cid))
        
        return detections
    
    def track_vehicles(self, frame, detections):
        """차량 추적"""
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []
        
        current_time = time.time()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            # 차량 위치 및 속도 정보 업데이트
            vehicle_info = self.vehicle_tracks.get(track_id, {
                'positions': [],
                'last_seen': current_time,
                'first_seen': current_time,
                'speeds': [],
                'stopped_duration': 0,
                'vehicle_class': detections[0][2] if detections else None  # 차량 종류
            })
            
            # 위치 기록 업데이트
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            vehicle_info['positions'].append((center_x, center_y, current_time))
            
            # 최근 5개 위치만 유지
            if len(vehicle_info['positions']) > 5:
                vehicle_info['positions'] = vehicle_info['positions'][-5:]
                
            # 속도 계산 (픽셀 단위 이동 거리)
            if len(vehicle_info['positions']) >= 2:
                prev_x, prev_y, prev_time = vehicle_info['positions'][-2]
                dx = center_x - prev_x
                dy = center_y - prev_y
                dt = current_time - prev_time
                if dt > 0:
                    speed = (dx**2 + dy**2)**0.5 / dt  # 유클리드 거리
                    vehicle_info['speeds'].append(speed)
                    
                    # 최근 3개 속도만 유지
                    if len(vehicle_info['speeds']) > 3:
                        vehicle_info['speeds'] = vehicle_info['speeds'][-3:]
            
            # 차량이 멈춰있는지 확인 (속도가 매우 낮으면)
            avg_speed = sum(vehicle_info['speeds']) / len(vehicle_info['speeds']) if vehicle_info['speeds'] else 0
            if avg_speed < 5:  # 임계값 (필요에 따라 조정)
                if 'stop_start_time' not in vehicle_info:
                    vehicle_info['stop_start_time'] = current_time
                vehicle_info['stopped_duration'] = current_time - vehicle_info['stop_start_time']
            else:
                if 'stop_start_time' in vehicle_info:
                    del vehicle_info['stop_start_time']
                vehicle_info['stopped_duration'] = 0
            
            # 마지막 확인 시간 업데이트
            vehicle_info['last_seen'] = current_time
            
            # 갑작스러운 가속/감속 확인 (이상 행동 패턴 감지)
            if len(vehicle_info['speeds']) >= 3:
                speed_changes = [abs(vehicle_info['speeds'][i] - vehicle_info['speeds'][i-1]) 
                               for i in range(1, len(vehicle_info['speeds']))]
                
                if max(speed_changes) > 50:  # 갑작스러운 속도 변화 임계값 (필요에 따라 조정)
                    if track_id not in self.abnormal_movements:
                        self.abnormal_movements[track_id] = {
                            'first_detected': current_time,
                            'count': 1
                        }
                    else:
                        self.abnormal_movements[track_id]['count'] += 1
            
            # 추적 정보 업데이트
            self.vehicle_tracks[track_id] = vehicle_info
            
            # 정지된 시간이 임계값을 초과하면 정지 차량으로 등록
            if vehicle_info['stopped_duration'] > 5:  # 5초 이상 정지한 차량
                if track_id not in self.stopped_vehicles:
                    self.stopped_vehicles[track_id] = {
                        'position': (center_x, center_y),
                        'first_stopped': current_time - vehicle_info['stopped_duration'],
                        'box': (x1, y1, x2, y2)
                    }
            else:
                # 다시 움직이면 정지 차량 목록에서 제거
                if track_id in self.stopped_vehicles:
                    del self.stopped_vehicles[track_id]
            
            # 활성 추적 정보 반환 (시각화 용도)
            active_tracks.append({
                'id': track_id,
                'box': (x1, y1, x2, y2),
                'speed': avg_speed,
                'stopped': vehicle_info['stopped_duration'] > 0,
                'stopped_duration': vehicle_info['stopped_duration']
            })
                
        return active_tracks
    
    def detect_accidents(self, frame, active_tracks):
        """사고 감지 로직
        여기서는 간단한 규칙 기반 사고 감지를 사용합니다.
        실제 구현에서는 더 복잡한 ML 모델이나 규칙을 사용할 수 있습니다.
        """
        accidents = []
        current_time = time.time()
        
        # 1. 오랫동안 정지해 있는 차량이 있는지 확인 (도로 위 사고 차량 가능성)
        for track_id, info in list(self.stopped_vehicles.items()):
            # 정지 시간이 임계값을 초과하면 사고로 간주
            stopped_duration = current_time - info['first_stopped']
            if stopped_duration > settings.MIN_ACCIDENT_DURATION:
                # 정지 차량 위치 주변에 다른 정지 차량이 있는지 확인 (복수 차량 사고 가능성)
                nearby_stopped = 0
                x1, y1, x2, y2 = info['box']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                for other_id, other_info in self.stopped_vehicles.items():
                    if other_id != track_id:
                        other_cx = (other_info['box'][0] + other_info['box'][2]) // 2
                        other_cy = (other_info['box'][1] + other_info['box'][3]) // 2
                        
                        # 두 차량 사이의 거리
                        distance = ((center_x - other_cx)**2 + (center_y - other_cy)**2)**0.5
                        
                        if distance < 200:  # 임계값 (픽셀 단위)
                            nearby_stopped += 1
                
                # 추가적인 조건 확인 (예: 교차로나 갓길이 아닌 일반 도로에서 정지 등)
                # 이 예제에서는 단순화를 위해 생략
                
                accident_severity = min(1.0, 0.6 + nearby_stopped * 0.1 + (stopped_duration - settings.MIN_ACCIDENT_DURATION) * 0.02)
                
                # 사고 감지 결과 추가
                accidents.append({
                    'type': 'stopped_vehicle',
                    'location': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'vehicles_involved': 1 + nearby_stopped,
                    'severity': accident_severity,
                    'confidence': 0.7 + (nearby_stopped * 0.1)
                })
        
        # 2. 비정상적인 움직임 패턴 확인 (급정거, 급가속 등)
        for track_id, info in list(self.abnormal_movements.items()):
            # 비정상 움직임이 여러 번 감지된 경우
            if info['count'] >= 3 and current_time - info['first_detected'] < 10:
                if track_id in self.vehicle_tracks:
                    v_info = self.vehicle_tracks[track_id]
                    if v_info['positions']:
                        last_pos = v_info['positions'][-1]
                        cx, cy = last_pos[0], last_pos[1]
                        
                        # 주변에 다른 차량이 있는지 확인
                        nearby_vehicles = 0
                        for other_id, other_info in self.vehicle_tracks.items():
                            if other_id != track_id and other_info['positions']:
                                other_pos = other_info['positions'][-1]
                                other_cx, other_cy = other_pos[0], other_pos[1]
                                
                                distance = ((cx - other_cx)**2 + (cy - other_cy)**2)**0.5
                                
                                if distance < 150:  # 임계값 (픽셀 단위)
                                    nearby_vehicles += 1
                        
                        # 추가 조건 확인 (급격한 속도 변화 등)
                        # ...
                        
                        # 빠르게 움직이다가 갑자기 멈춘 경우 (충돌 가능성)
                        if track_id in self.stopped_vehicles and info['count'] >= 5:
                            accidents.append({
                                'type': 'sudden_stop',
                                'location': (cx, cy),
                                'bbox': (cx-50, cy-50, cx+50, cy+50),  # 대략적인 영역
                                'vehicles_involved': 1 + nearby_vehicles,
                                'severity': 0.7 + (nearby_vehicles * 0.1),
                                'confidence': 0.65 + (info['count'] * 0.05)
                            })
        
        # 3. 차선 이탈 감지 (선택적 구현 - 실제로는 차선 검출 로직 필요)
        # ...
        
        # 사고 감지 결과 반환
        return accidents
    
    def visualize_tracking(self, frame, active_tracks, accidents=None):
        """추적 및 사고 감지 결과 시각화"""
        # 추적 중인 차량 시각화
        for track in active_tracks:
            track_id = track['id']
            x1, y1, x2, y2 = track['box']
            
            # 정지 차량은 빨간색, 움직이는 차량은 녹색으로 표시
            if track['stopped']:
                color = (0, 0, 255)  # 빨간색 (정지 차량)
                thickness = 2
                label = f"ID:{track_id} (stopped: {track['stopped_duration']:.1f}s)"
            else:
                color = (0, 255, 0)  # 녹색 (움직이는 차량)
                thickness = 2
                label = f"ID:{track_id} (speed: {track['speed']:.1f})"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 사고 감지 시각화
        if accidents:
            for accident in accidents:
                x1, y1, x2, y2 = accident['bbox']
                severity = accident['severity']
                vehicles = accident['vehicles_involved']
                
                # 사고 심각도에 따라 색상 조정 (0.8 이상이면 밝은 빨간색)
                if severity >= 0.8:
                    color = (0, 0, 255)  # 빨간색 (심각한 사고)
                else:
                    color = (0, 165, 255)  # 주황색 (경미한 사고)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # 사고 정보 표시
                label = f"ACCIDENT! Type: {accident['type']}, Severity: {severity:.2f}"
                cv2.putText(frame, label, (x1, y1 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                label2 = f"Vehicles: {vehicles}, Conf: {accident['confidence']:.2f}"
                cv2.putText(frame, label2, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

def report_accident(accident_data, cctv_info):
    """사고 정보를 API 서버에 보고"""
    try:
        api_url = "http://localhost:8000/accidents/"  # 실제 서버 URL로 변경 필요
        
        # 사고 정보 구성
        accident_type = accident_data['type']
        if accident_type == 'stopped_vehicle':
            accident_type_kr = '차량 정지'
        elif accident_type == 'sudden_stop':
            accident_type_kr = '급정거 사고'
        else:
            accident_type_kr = '미확인 사고'
            
        payload = {
            "cctv_id": cctv_info["id"],
            "cctv_name": cctv_info["name"],
            "location": cctv_info["location"],
            "accident_type": accident_type_kr,
            "severity": accident_data["severity"],
            "vehicles_involved": accident_data["vehicles_involved"],
            "details": {
                "detection_confidence": accident_data["confidence"],
                "accident_location_x": accident_data["location"][0],
                "accident_location_y": accident_data["location"][1],
                "accident_type_original": accident_data["type"]
            }
        }
        
        # API 호출
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        accident_id = response.json().get("id")
        print(f"사고 정보가 성공적으로 보고되었습니다. 사고 ID: {accident_id}")
        return accident_id
    
    except Exception as e:
        print(f"사고 보고 중 오류 발생: {str(e)}")
        return None

def upload_accident_image(accident_id, frame):
    """사고 이미지를 서버에 업로드"""
    try:
        if accident_id is None:
            return
            
        api_url = f"http://localhost:8000/accidents/{accident_id}/image"
        
        # 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # API 호출
        response = requests.post(api_url, json=img_base64)
        response.raise_for_status()
        
        print(f"사고 이미지가 성공적으로 업로드되었습니다. 사고 ID: {accident_id}")
    
    except Exception as e:
        print(f"사고 이미지 업로드 중 오류 발생: {str(e)}")

def detection_worker(cctv_info, on_complete=None):
    """사고 감지 작업을 수행하는 워커 함수"""
    cctv_id = cctv_info["id"]
    cctv_url = cctv_info.get("url", "")
    
    try:
        print(f"CCTV {cctv_id} ({cctv_info.get('name', '')}) 모니터링 시작")
        
        # 비디오 스트림 열기
        cap = None
        
        # CCTV URL이 없거나 연결 실패 시 로컬 비디오 사용
        if not cctv_url:
            print(f"CCTV {cctv_id}의 URL이 없습니다. 로컬 비디오를 사용합니다.")
            # 여기서 DeepSort 코드처럼 직접 경로를 지정
            local_path = settings.LOCAL_VIDEO_PATH
            print(f"로컬 비디오 경로: {local_path}")
            cap = cv2.VideoCapture(local_path)
        else:
            # CCTV 스트림 연결 시도
            try:
                cap = cv2.VideoCapture(cctv_url)
                if not cap.isOpened():
                    raise Exception("스트림에 연결할 수 없습니다.")
            except Exception as e:
                print(f"CCTV 스트림 연결 오류: {str(e)}. 로컬 비디오를 사용합니다.")
                local_path = settings.LOCAL_VIDEO_PATH
                print(f"로컬 비디오 경로: {local_path}")
                cap = cv2.VideoCapture(local_path)
        
        if not cap.isOpened():
            print(f"CCTV {cctv_id}: 비디오 소스를 열 수 없습니다.")
            return
        
        # 사고 감지 모델 초기화
        detection_model = AccidentDetectionModel()
        
        # 프레임 처리 관련 변수
        frame_count = 0
        reported_accidents = set()  # 이미 보고된 사고 추적
        last_accident_check_time = time.time()
        
        print(f"CCTV {cctv_id} 모니터링 시작")
        
        while not stop_event.is_set():
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print(f"CCTV {cctv_id}: 비디오 스트림 종료")
                break
            
            # 프레임 카운트 증가
            frame_count += 1
            
            # 모든 프레임을 처리하지 않고 일부만 처리 (성능 최적화)
            if frame_count % 5 != 0:  # 5프레임마다 처리
                continue
            
            # 차량 감지 및 추적
            detections = detection_model.detect_vehicles(frame)
            active_tracks = detection_model.track_vehicles(frame, detections)
            
            # 사고 감지 (일정 간격으로)
            current_time = time.time()
            if current_time - last_accident_check_time >= settings.DETECTION_INTERVAL:
                accidents = detection_model.detect_accidents(frame, active_tracks)
                last_accident_check_time = current_time
                
                # 감지된 사고 처리
                for accident in accidents:
                    # 사고 신뢰도가 임계값 이상인 경우만 처리
                    if accident["confidence"] >= settings.ACCIDENT_DETECTION_THRESHOLD:
                        # 사고 위치에 근거한 고유 ID 생성 (같은 사고 중복 보고 방지)
                        loc_x, loc_y = accident["location"]
                        accident_key = f"{accident['type']}_{int(loc_x/10)}_{int(loc_y/10)}"
                        
                        # 아직 보고되지 않은 사고만 보고
                        if accident_key not in reported_accidents:
                            # 시각화된 프레임 생성
                            vis_frame = detection_model.visualize_tracking(frame.copy(), active_tracks, [accident])
                            
                            # 사고 보고
                            accident_id = report_accident(accident, cctv_info)
                            
                            # 사고 이미지 업로드
                            if accident_id:
                                upload_accident_image(accident_id, vis_frame)
                                
                            # 보고된 사고 목록에 추가
                            reported_accidents.add(accident_key)
                            
                            print(f"CCTV {cctv_id}에서 사고 감지: {accident['type']}, 심각도: {accident['severity']}")
            
            # 디버그용 시각화 (실제 서버에서는 비활성화 가능)
            if settings.DEBUG_MODE and hasattr(settings, 'HEADLESS_MODE') and not settings.HEADLESS_MODE:
                # 추적 결과 시각화
                vis_frame = detection_model.visualize_tracking(frame.copy(), active_tracks)
                
                # 창 이름에 CCTV ID 포함
                window_name = f"CCTV {cctv_id}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, cv2.resize(vis_frame, (1280, 720)))
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 처리 간격 조절 (CPU 부하 감소)
            time.sleep(0.01)
        
        # 자원 해제
        cap.release()
        if settings.DEBUG_MODE and hasattr(settings, 'HEADLESS_MODE') and not settings.HEADLESS_MODE:
            cv2.destroyAllWindows()
        
        print(f"CCTV {cctv_id} 모니터링 종료")
    
    except Exception as e:
        print(f"CCTV {cctv_id} 모니터링 중 오류 발생: {str(e)}")
    
    finally:
        # 완료 콜백 호출
        if on_complete:
            on_complete()

async def start_detection(cctv_info, on_complete=None):
    """사고 감지 프로세스 시작 (비동기)"""
    cctv_id = cctv_info["id"]
    
    if cctv_id in detection_threads:
        print(f"CCTV {cctv_id}는 이미 모니터링 중입니다.")
        return
    
    # 중지 이벤트 초기화
    stop_event.clear()
    
    # 스레드 생성 및 시작
    thread = threading.Thread(
        target=detection_worker,
        args=(cctv_info, on_complete),
        daemon=True
    )
    detection_threads[cctv_id] = thread
    thread.start()
    
    # 상태 업데이트 대기
    await asyncio.sleep(1)
    
    return {"status": "running", "cctv_id": cctv_id}

def stop_detection(cctv_id):
    """사고 감지 프로세스 중지"""
    if cctv_id not in detection_threads:
        print(f"CCTV {cctv_id}는 현재 모니터링되지 않고 있습니다.")
        return
    
    # 중지 이벤트 설정
    stop_event.set()
    
    # 스레드가 종료될 때까지 대기
    thread = detection_threads[cctv_id]
    if thread.is_alive():
        thread.join(timeout=5)
    
    # 스레드 목록에서 제거
    if cctv_id in detection_threads:
        del detection_threads[cctv_id]
    
    # 중지 이벤트 초기화
    stop_event.clear()
    
    print(f"CCTV {cctv_id} 모니터링이 중지되었습니다.")
    
    return {"status": "stopped", "cctv_id": cctv_id}
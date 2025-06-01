#model.py
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import time
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile
import websocket
import json
import threading


model = YOLO("yolov8m.pt") #아직 학습 완료되지 않은 상태
model.fuse() 
tracker = DeepSort(max_age=60, nn_budget=100, max_cosine_distance=0.3)


previous_positions = defaultdict(lambda: None)
second_last_positions = defaultdict(lambda: None)
previous_speeds = defaultdict(lambda: None)
stopped_frames = defaultdict(int)
suspicion_scores = defaultdict(float)
previous_boxes = defaultdict(lambda: None)
recent_ids = defaultdict(lambda: {'last_seen': 0, 'box': None})


ACCELERATION_THRESHOLD = 30
OVERLAP_IOU_THRESHOLD = 0.4
STATIONARY_DISTANCE = 6
STATIONARY_FRAMES = 30
PERPENDICULAR_MOVEMENT_THRESHOLD = 25
CONFIDENCE_THRESHOLD = 0.25
SUSPICION_SCORE_THRESHOLD = 3.0
SUSPICION_DECAY = 0.005
ID_TRANSFER_IOU_THRESHOLD = 0.65
ID_TRANSFER_TIME_WINDOW = 40
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# 웹소켓 클라이언트 함수 추가
def send_accident_alert(accident_data):
    try:
        # 웹소켓 연결
        ws = websocket.create_connection("ws://localhost:8000/ws")
        
        # 알림 메시지 생성
        message = {
            "type": "accident_detected",
            "data": {
                "id": f"accident_{accident_data['track_id']}_{int(time.time())}",
                "location": "고속도로 사고 지점",
                "cctv_name": "테스트 CCTV",
                "detected_at": datetime.now().isoformat()
            }
        }
        
        # 메시지 전송
        ws.send(json.dumps(message))
        print(f"사고 알림 전송 완료: {message['data']['id']}")
        
        # 연결 종료
        ws.close()
        return True
    except Exception as e:
        print(f"알림 전송 실패: {str(e)}")
        return False

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def smooth_box(prev_box, new_box, alpha=0.4):
    if prev_box is None:
        return new_box
    return [int(prev_box[i] * alpha + new_box[i] * (1 - alpha)) for i in range(4)]

def detect_accident(track_results, frame_count, frame_interval):
    suspicious_ids = set()
    tracks = list(track_results.items())

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            iou_value = iou(tracks[i][1]['bbox'], tracks[j][1]['bbox'])
            if iou_value > OVERLAP_IOU_THRESHOLD:
                suspicion_scores[tracks[i][0]] += 1
                suspicion_scores[tracks[j][0]] += 1

    for tid, data in track_results.items():
        bbox = data['bbox']
        center = np.array(data['center'])
        prev_pos = previous_positions[tid]
        prev_prev_pos = second_last_positions[tid]
        prev_speed = previous_speeds[tid]

        if prev_pos is not None:
            v2 = center - prev_pos
            velocity = np.linalg.norm(v2) / frame_interval
            score = 0

            if prev_prev_pos is not None:
                v1 = prev_pos - prev_prev_pos
                if np.linalg.norm(v1) > 0:
                    unit_dir = v1 / np.linalg.norm(v1)
                    perp = v2 - np.dot(v2, unit_dir) * unit_dir
                    deviation = np.linalg.norm(perp)
                    if deviation > PERPENDICULAR_MOVEMENT_THRESHOLD:
                        score += 1

            if prev_speed is not None:
                delta = velocity - prev_speed
                if delta < -ACCELERATION_THRESHOLD:
                    score += 1

            if velocity < STATIONARY_DISTANCE:
                stopped_frames[tid] += 1
                if stopped_frames[tid] >= STATIONARY_FRAMES:
                    score += 1
            else:
                stopped_frames[tid] = 0

            if score >= 2:
                suspicion_scores[tid] += 0.5

            previous_speeds[tid] = velocity

        second_last_positions[tid] = previous_positions[tid]
        previous_positions[tid] = center

    for tid in list(suspicion_scores.keys()):
        suspicion_scores[tid] = max(0, suspicion_scores[tid] - SUSPICION_DECAY)
        if suspicion_scores[tid] >= SUSPICION_SCORE_THRESHOLD:
            suspicious_ids.add(tid)

    return suspicious_ids

def run(video_input, show=False):
    if isinstance(video_input, str):
        cap = cv2.VideoCapture(video_input)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_input.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps if fps > 0 else 1 / 30
    accident_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
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

        for new_tid in list(track_results.keys()):
            for old_tid, data in recent_ids.items():
                if old_tid == new_tid:
                    continue
                if 0 < frame_count - data['last_seen'] < ID_TRANSFER_TIME_WINDOW:
                    if iou(track_results[new_tid]['bbox'], data['box']) > ID_TRANSFER_IOU_THRESHOLD:
                        suspicion_scores[new_tid] += suspicion_scores.get(old_tid, 0)
                        print(f"[ID 계승] ID {old_tid} → ID {new_tid}")
                        break

        suspicious = detect_accident(track_results, frame_count, frame_interval)

        for tid, data in track_results.items():
            x1, y1, x2, y2 = data['bbox']
            color = (0, 0, 255) if tid in suspicious else (0, 255, 0)
            if show:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tid}"
                if tid in suspicious:
                    label += " accident"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if tid in suspicious:
                accident_log.append({
                    "track_id": tid,
                    "frame": frame_count,
                    "bbox": data['bbox'],
                    "timestamp": time.time()
                })
                
                # 사고 알림 전송
                threading.Thread(
                    target=send_accident_alert, 
                    args=({
                        "track_id": tid,
                        "frame": frame_count,
                        "bbox": data['bbox']
                    },)
                ).start()

        if show:
            cv2.imshow("Accident Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return accident_log

if __name__ == "__main__":
    result = run("C:/Users/user/Desktop/a9.mp4", show=True)
    print("Accident Log:", result)
# test_accident_detection.py
import cv2
from app.services.accident_detection import AccidentDetectionModel
import time

# 테스트 비디오 경로
video_path = "videos/highway.mp4"

# 사고 감지 모델 초기화
model = AccidentDetectionModel()

# 비디오 열기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"비디오를 열 수 없습니다: {video_path}")
    exit()

print(f"비디오가 성공적으로 열렸습니다: {video_path}")

frame_count = 0
last_accident_check_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % 5 != 0:  # 5프레임마다 처리
        continue
        
    # 차량 감지 및 추적
    detections = model.detect_vehicles(frame)
    active_tracks = model.track_vehicles(frame, detections)
    
    # 사고 감지
    current_time = time.time()
    if current_time - last_accident_check_time >= 1:  # 1초마다 확인
        accidents = model.detect_accidents(frame, active_tracks)
        last_accident_check_time = current_time
        
        for accident in accidents:
            print(f"사고 감지: {accident['type']}, 심각도: {accident['severity']}")
    
    # 시각화
    vis_frame = model.visualize_tracking(frame.copy(), active_tracks)
    cv2.imshow("Test Detection", cv2.resize(vis_frame, (1280, 720)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
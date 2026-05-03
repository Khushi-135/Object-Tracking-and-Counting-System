import cv2
import time
import torch
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import threading
from ultralytics import YOLO
import subprocess


# ================================
# AUDIO RECORDING SETUP
# ================================
audio_frames = []
record_audio = True
AUDIO_FS = 44100  # Sample rate

def audio_recorder():
    print("🎤 Audio recording started...")
    while record_audio:
        data = sd.rec(int(AUDIO_FS * 2), samplerate=AUDIO_FS, channels=1, dtype='int16')
        sd.wait()
        audio_frames.append(data)

# Start audio thread
audio_thread = threading.Thread(target=audio_recorder)
audio_thread.start()

# ================================
# Load YOLOv8 Model
# ================================
model = YOLO("yolov8n.pt")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_type}")

# ================================
# COCO Class Names
# ================================
coco_classes = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
    'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

# ================================
# Choose Input Mode
# ================================
print("Choose Input Mode:")
print("1. Live Camera Feed")
print("2. Upload Video File")
choice = input("Enter 1 or 2: ")

if choice == "1":
    cap = cv2.VideoCapture(0)
    output_path = 'output_live_temp.mp4'
elif choice == "2":
    video_path = input("Enter the path of the video file: ")
    cap = cv2.VideoCapture(video_path)
    output_path = 'output_video_temp.mp4'
else:
    print("❌ Invalid choice! Exiting...")
    exit()

# ================================
# Video Writer Setup
# ================================
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
if fps_input == 0:
    fps_input = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

# Tracking variables
unique_track_ids = set()
total_frames = 0
correct_detections = 0
start_time = time.time()
accuracy = 0.0
RESIZE_FACTOR = 0.5

print("⏳ Starting YOLOv8 tracking... Press 'q' to exit.")

# ================================
# Main Loop
# ================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠ No more frames. Exiting...")
        break

    total_frames += 1

    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    results = model.track(small_frame, tracker="bytetrack.yaml", conf=0.4, verbose=False)

    detected = False
    current_track_ids = set()

    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            scale = 1 / RESIZE_FACTOR
            x1, y1, x2, y2 = [int(coord * scale) for coord in (x1, y1, x2, y2)]

            track_id = int(box.id.item()) if box.id is not None else -1
            class_id = int(box.cls[0].item())

            class_name = coco_classes[class_id] if 0 <= class_id < len(coco_classes) else "unknown"

            color = ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 29) % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}-{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if track_id != -1:
                unique_track_ids.add(track_id)
                current_track_ids.add(track_id)
                detected = True

    if detected:
        correct_detections += 1

    accuracy = (correct_detections / total_frames) * 100
    fps = total_frames / (time.time() - start_time)
    total_tracked_objects = len(unique_track_ids)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Currently Tracked: {len(current_track_ids)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Unique Objects: {total_tracked_objects}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopping detection...")
        break

# ================================
# STOP EVERYTHING
# ================================
cap.release()
out.release()
cv2.destroyAllWindows()

# Stop audio thread
record_audio = False
audio_thread.join()

# Save audio file
audio_data = np.concatenate(audio_frames, axis=0)
write("audio_temp.wav", AUDIO_FS, audio_data)

print("🎤 Audio saved as audio_temp.wav")
print("🎥 Video saved as", output_path)
print(f"Final Accuracy: {accuracy:.2f}%")
print(f"Total Unique Objects Tracked: {len(unique_track_ids)}")





print("➤ Includes audio + video + tracking + accuracy + FPS!")

import os
import csv
import cv2
from ultralytics import YOLO

# =========================
# Settings
# =========================
VIDEO_PATH = "test.mp4"            # حط اسم ملف الفيديو عندك هنا (مثلاً: myvideo.mp4)
OUT_DIR = "yolo_out"
OUT_VIDEO = os.path.join(OUT_DIR, "detected.mp4")
OUT_FRAMES_DIR = os.path.join(OUT_DIR, "frames")   # لحفظ فريمات عليها boxes (اختياري)
SAVE_FRAMES = True                 # خلّها True لو تبي صور، False لو تبي فيديو فقط

CONF = 0.35                        # ارفعها لـ 0.5 إذا كثرت الـ false positives
IOU = 0.45

# =========================
# Prepare folders
# =========================
os.makedirs(OUT_DIR, exist_ok=True)
if SAVE_FRAMES:
    os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

# =========================
# Load model (light + fast)
# =========================
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")  # nano model: fast and good enough for most demos

# =========================
# Open video
# =========================
print("Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}\n"
                            f"Make sure the file exists in the project folder, or change VIDEO_PATH.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

# =========================
# CSV logging
# =========================
csv_path = os.path.join(OUT_DIR, "labels.csv")
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])

print(f"Video info: {w}x{h} | FPS={fps:.2f} | Frames={total}")
print("Detecting...")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # YOLO inference
    results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
    r = results[0]

    # Draw detections
    annotated = frame.copy()
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_name = model.names.get(cls_id, str(cls_id))

            # green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # log to CSV
            csv_writer.writerow([frame_idx, cls_id, cls_name, f"{conf:.4f}", x1, y1, x2, y2])

    writer.write(annotated)

    if SAVE_FRAMES and frame_idx < 200:  # نحفظ أول 200 فريم فقط عشان ما يثقل
        out_img = os.path.join(OUT_FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_img, annotated)

    if frame_idx % 50 == 0:
        print(f"Processed frame {frame_idx}/{total}")

    frame_idx += 1

cap.release()
writer.release()
csv_file.close()

print("DONE 🎉")
print(f"Saved video: {OUT_VIDEO}")
print(f"Saved CSV:   {csv_path}")
if SAVE_FRAMES:
    print(f"Saved frames: {OUT_FRAMES_DIR}")

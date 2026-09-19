import cv2
import os

video_path = "test.mp4"

print("Checking video...")

if not os.path.exists(video_path):
    print("❌ ما لقيت الفيديو")
    exit()

print("Video found ✔")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ماقدر يفتح الفيديو")
    exit()

print("Video opened ✔")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

os.makedirs("frames", exist_ok=True)

indices = [0, frame_count//4, frame_count//2, (3*frame_count)//4, frame_count-1]

for i, idx in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()

    if ok:
        path = f"frames/frame_{i}.jpg"
        cv2.imwrite(path, frame)
        print("Saved:", path)

cap.release()

print("DONE 🎉")

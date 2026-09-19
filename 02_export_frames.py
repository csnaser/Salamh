import os
import cv2
from datasets import load_dataset
import requests

OUT_DIR = "sample_frames"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset...")
ds = load_dataset("raiyaanabdullah/isafety-bench", split="train", decode=False)

sample = ds[12]

video_url = sample["video"]
label = sample["label"]

print("Label:", label)
print("Video URL:", video_url)

# تحويل رابط hf إلى رابط تحميل مباشر
if video_url.startswith("hf://"):
    video_url = video_url.replace(
        "hf://datasets/",
        "https://huggingface.co/datasets/"
    ).replace("@", "/resolve/")

print("Download URL:", video_url)

# تحميل الفيديو
local_path = "sample.mp4"

if not os.path.exists(local_path):
    print("Downloading video...")
    r = requests.get(video_url, stream=True)
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Video saved locally")

# فتح الفيديو
cap = cv2.VideoCapture(local_path)

if not cap.isOpened():
    raise RuntimeError("Failed to open video")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

indices = [0, frame_count//4, frame_count//2, (3*frame_count)//4, frame_count-1]

for i, idx in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()

    if ok:
        path = os.path.join(OUT_DIR, f"frame_{i}.jpg")
        cv2.imwrite(path, frame)
        print("Saved:", path)

cap.release()
print("Done.")

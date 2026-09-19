import cv2
import os

folder = "frames"

# HOG person detector (built-in)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

print("Found images:", len(imgs))

for name in imgs:
    path = os.path.join(folder, name)
    img = cv2.imread(path)

    if img is None:
        print(" Can't read:", path)
        continue

    # detection
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))

    # draw
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(f"Detected - {name} (press any key)", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("DONE ")

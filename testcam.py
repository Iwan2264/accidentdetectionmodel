import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Webcam opened. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    results = model(frame, imgsz=416)  

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Live Detection (Fast)", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

model = YOLO("best.pt") 

image = cv2.imread("test4.jpg")  

results = model(image)

annotated = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

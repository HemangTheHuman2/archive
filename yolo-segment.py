from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-seg.pt")
video_path = 'sample_vid/test3.mp4'  
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=0.3, verbose=False)[0]
    cv2.imshow("Road Segmentation", result.plot())

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Load a COCO-pretrained YOLOv8n model
# model = YOLO("yolov8n.pt")
from ultralytics import RTDETR
import cv2

# Load the pretrained RT-DETR model
model = RTDETR("rtdetr-l.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Run RT-DETR model on the current frame
    results = model.predict(source=frame, stream=False, verbose=False)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("RT-DETR Real-Time Detection", annotated_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")

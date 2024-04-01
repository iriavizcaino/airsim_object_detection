import cv2
from ultralytics import YOLO

# Load the pretrained model
# model = YOLO('yolov8n.pt')
model = YOLO('extinguisher.pt')

# Define a video capture object
cap = cv2.VideoCapture("rtsp://192.168.21.82:8554/title")


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        cv2.waitKey(1)
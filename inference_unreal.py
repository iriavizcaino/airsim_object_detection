import cv2
import airsim
from ultralytics import YOLO

client = airsim.VehicleClient()
client.confirmConnection()

camera_name = "0"
image_type = airsim.ImageType.Scene

# Load the YOLOv8 model
model = YOLO('extinguisher.pt')


# Loop through the video frames
while True:

    rawImage = client.simGetImage(camera_name, image_type)
    if not rawImage:
        exit()

    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)

    # Run YOLOv8 inference on the frame
    results = model(png[:,:,:3])

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(1)

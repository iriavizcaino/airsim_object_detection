# Airsim Object Detection
In this repository you will find an object detection datasets generator (YOLO format) and two ways to get inference results.

## Dependencies

- Install Unreal Engine.  
- Python packages: `pip3 install -r requirements.txt`

## Usage
Steps:

- Run Unreal with AirSim plugin enabled.  
- Run `get_files.py` on a terminal and wait until the desired number of files are generated.  
- Train a pretrained model using the following command in the Dataset directory terminal:  
`yolo detect train data=dataset.yaml model=extinguisher.pt epochs=100 imgsz=640`  
- Run any of following scripts to get inference results:
    - `inference_unreal.py`: runs an object detection model on video coming from airsim.  
    - `inference_webcam.py`: runs an object detection model on video coming from your webcam or any live video application.  

### Add backgrounds
If you want to add or change backgrounds, download some HDRIs with **.ext** extension and save them on backgrounds directory
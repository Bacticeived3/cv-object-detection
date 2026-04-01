# Computer Vision Object Detection

This project implements a real-time object detection system using the YOLOv8 architecture. It demonstrates how to train a custom object detection model, optimize it for performance, and deploy it for various applications, including edge computing.

## Features
- Custom dataset preparation
- YOLOv8 model training and validation
- Model optimization techniques (e.g., quantization)
- Real-time inference scripts
- Performance benchmarking

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
# Example usage of the object detection model
from detect import detect_objects

image_path = "./data/sample_image.jpg"
detected_objects = detect_objects(image_path)
print(f"Detected objects: {detected_objects}")
```

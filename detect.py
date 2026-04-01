from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_objects(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                detections.append({"box": [x1, y1, x2, y2], "confidence": conf, "class_id": cls})
        return detections

if __name__ == "__main__":
    detector = ObjectDetector()
    # Create a dummy image for testing
    dummy_image_path = "./data/sample_image.jpg"
    import numpy as np
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(dummy_image_path, dummy_image)
    
    detections = detector.detect_objects(dummy_image_path)
    print(f"Detected: {detections}")

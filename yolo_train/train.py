from ultralytics import YOLO
model = YOLO("yolo11.pt")
resaults = model(data = "data.yaml",epochs = 100,imgsz = 640)
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(data="./datasets/rock-paper-scissors/data.yaml", epochs=1, batch=8)
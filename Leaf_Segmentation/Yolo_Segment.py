from ultralytics import YOLO
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

model = YOLO("yolo11n-seg.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)

tracker.stop()
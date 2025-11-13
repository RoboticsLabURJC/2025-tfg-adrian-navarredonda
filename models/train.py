from ultralytics import YOLO

model = YOLO("yolo11n.pt")  

# Entrenamiento
model.train(
    data="data.v4i.yolov11/data.yaml", 
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,
    evolve=True
)

model.export(format="onnx")
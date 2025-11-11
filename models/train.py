from ultralytics import YOLO

# Modelo base pre-entrenado (tiny o small)
model = YOLO("yolo11n.pt")  

# Entrenamiento
model.train(
    data="data.v4i.yolov11/data.yaml",   # tu archivo con paths y clases
    epochs=100,                 # puedes empezar con 50-100 y ajustar
    imgsz=640,                  # tamaño de imagen para entrenamiento
    batch=16,                   # tu GPU puede manejar más si quieres
    device=0,                   # usa la RTX 4090
    name="yolov11_cones"
)

model.export(format="onnx")
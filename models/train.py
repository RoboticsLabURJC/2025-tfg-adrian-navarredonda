from ultralytics import YOLO

model = YOLO("yolo11n.pt")  

# Entrenamiento
model.train(
    data="data.v4i.yolov11/data.yaml", 
    epochs=200,
    imgsz=640,
    batch=32,
    device=0,
    lr0=0.009350086700765918,
    lrf= 0.10837673249548656,
    cos_lr = True,
    momentum = 0.8350094791321736,
    weight_decay = 1.0360928219562484e-05,
    hsv_h= 0.003937675020288892,
    hsv_s= 0.29254498954113756,
    hsv_v= 0.30346865258176065,
    degrees= 6.608869336213175,
    translate= 0.098740674206747,
    scale= 0.21329565421105243,
    shear= 3.160880172833644,
    fliplr= 0.47167796586344474
)

model.export(format="onnx")

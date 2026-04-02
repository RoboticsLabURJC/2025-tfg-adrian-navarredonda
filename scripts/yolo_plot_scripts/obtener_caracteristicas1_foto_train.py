import time
import csv
from ultralytics import YOLO
import os

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
MODEL_PATH = "../../models/runs/detect/Run_con_parametros/weights/best.pt"
IMAGE_PATH = "/home/adrian/Escritorio/image1.png"   # 🔥 imagen única
CSV_OUTPUT = "benchmark_imagen_unica.csv"

NUM_RUNS = 8000  # número de inferencias

# -------------------------------
# CARGAR MODELO
# -------------------------------
model = YOLO(MODEL_PATH)
model.to("cuda")

# -------------------------------
# CREAR CSV
# -------------------------------
with open(CSV_OUTPUT, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "imagen",
        "run",
        "tiempo_inferencia_ms",
        "clase",
        "confidencia",
        "x1", "y1", "x2", "y2"
    ])

    img_name = os.path.basename(IMAGE_PATH)

    for run in range(NUM_RUNS):

        inicio = time.time()
        results = model(IMAGE_PATH)
        fin = time.time()

        tiempo_ms = (fin - inicio) * 1000
        detections = results[0].boxes

        if detections is None or len(detections) == 0:
            writer.writerow([
                img_name,
                run,
                tiempo_ms,
                "none",
                0,
                0, 0, 0, 0
            ])
            print(f"{img_name} | run {run}: {tiempo_ms:.2f} ms (sin detecciones)")
            continue

        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            writer.writerow([
                img_name,
                run,
                tiempo_ms,
                cls,
                conf,
                x1, y1, x2, y2
            ])

        print(f"{img_name} | run {run}: {tiempo_ms:.2f} ms ({len(detections)} detecciones)")

print(f"\n📁 Resultados guardados en: {CSV_OUTPUT}")
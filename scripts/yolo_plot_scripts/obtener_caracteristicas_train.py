import time
import csv
from ultralytics import YOLO
import glob
import os

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
MODEL_PATH = "../../models/runs/detect/Run_con_parametros/weights/best.pt"
# IMAGES_FOLDER = "/home/adrian/Descargas/data.v4i.yolov11/train/images/" # portatil
IMAGES_FOLDER = "../../models/data.v4i.yolov11/train/images"
CSV_OUTPUT = "caracteristicas_train_servidor.csv"

# Cargar modelo
model = YOLO(MODEL_PATH)

# Obtener imágenes
image_paths = glob.glob(os.path.join(IMAGES_FOLDER, "*.*"))

if not image_paths:
    raise ValueError("❌ No se encontraron imágenes en la carpeta especificada.")

# -------------------------------
# CREAR CSV
# -------------------------------
with open(CSV_OUTPUT, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Cabeceras del CSV
    writer.writerow([
        "imagen",
        "tiempo_inferencia_ms",
        "clase",
        "confidencia",
        "x1", "y1", "x2", "y2"
    ])

    # Procesar imágenes
    for img_path in image_paths:

        # Registrar tiempo de cada inferencia
        inicio = time.time()
        results = model(img_path)  
        fin = time.time()

        tiempo_ms = (fin - inicio) * 1000

        # Extraer detecciones
        detections = results[0].boxes

        if detections is None or len(detections) == 0:
            # Guardar fila indicando sin detecciones
            writer.writerow([
                os.path.basename(img_path),
                tiempo_ms,
                "none",
                0,
                0, 0, 0, 0
            ])
            print(f"{os.path.basename(img_path)}: {tiempo_ms:.2f} ms (sin detecciones)")
            continue

        # Para cada bounding box detectada
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            writer.writerow([
                os.path.basename(img_path),
                tiempo_ms,
                cls,
                conf,
                x1, y1, x2, y2
            ])

        print(f"{os.path.basename(img_path)}: {tiempo_ms:.2f} ms ({len(detections)} detecciones)")

print(f"\n📁 Resultados guardados en: {CSV_OUTPUT}")

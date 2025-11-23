import time
import csv
from ultralytics import YOLO
import glob
import os

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
MODEL_PATH = "../../models/runs/detect/Run_con_parametros/weights/best.pt"        # Pon aquí tu modelo YOLOv11
IMAGES_FOLDER = "/home/adrian/Descargas/data.v4i.yolov11/train/images/"         # Carpeta con imágenes de prueba
CSV_OUTPUT = "tiempos_inferencia.csv"      # Archivo CSV de salida
N_RUNS = 1                                  # Número de veces que quieres repetir cada inferencia

# -------------------------------
# CARGA DEL MODELO
# -------------------------------
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
    writer.writerow(["imagen", "tiempo_inferencia_ms"])

    # Procesar cada imagen
    for img_path in image_paths:
        for _ in range(N_RUNS):
            inicio = time.time()
            _ = model(img_path)  # inferencia
            fin = time.time()

            tiempo_ms = (fin - inicio) * 1000  # pasar a milisegundos

            writer.writerow([os.path.basename(img_path), tiempo_ms])
            print(f"{os.path.basename(img_path)}: {tiempo_ms:.2f} ms")

print(f"\n📁 Resultados guardados en: {CSV_OUTPUT}")




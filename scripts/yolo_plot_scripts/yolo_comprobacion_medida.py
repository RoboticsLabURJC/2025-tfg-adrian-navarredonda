import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
CSV_YOLO = "caracteristicas_train_portatil.csv"  # CSV con detecciones YOLO
IMAGES_FOLDER = "/home/adrian/Descargas/data.v4i.yolov11/train/images"

# Rangos HSV para colores
COLOR_RANGES = {
    "amarillo": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    "azul":     (np.array([100, 150, 0]), np.array([140, 255, 255]))
}

# Cargar CSV YOLO
df_yolo = pd.read_csv(CSV_YOLO)

# Contadores
totales = {"amarillo":0, "azul":0}
correctas = {"amarillo":0, "azul":0}

# -------------------------------
# PROCESAR IMÁGENES
# -------------------------------
for img_name in df_yolo['imagen'].unique():
    img_path = os.path.join(IMAGES_FOLDER, img_name)
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detections = df_yolo[df_yolo['imagen'] == img_name]

    for _, row in detections.iterrows():
        yolo_clase = row['clase']
        if yolo_clase not in COLOR_RANGES:
            continue

        totales[yolo_clase] += 1

        # Recortar bbox
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        roi = img_hsv[y1:y2, x1:x2]

        # Máscara de color
        lower, upper = COLOR_RANGES[yolo_clase]
        mask = cv2.inRange(roi, lower, upper)
        if cv2.countNonZero(mask) > 0:
            correctas[yolo_clase] += 1

# -------------------------------
# CREAR PLOT DE BARRAS
# -------------------------------
colores = ['amarillo', 'azul']
valores = [correctas[c]/totales[c] if totales[c]>0 else 0 for c in colores]

plt.figure(figsize=(6,5))
plt.bar(colores, valores, color=['yellow','blue'])
plt.ylabel("Proporción de detecciones correctas")
plt.ylim(0,1)
plt.title("Precisión de YOLO según color")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#!/usr/bin/env python3
"""
Este script extrae un subconjunto de un dataset filtrando únicamente las muestras 
con dirección extrema (steer = 1.0 o -1.0).

A partir de un CSV que contiene las etiquetas del dataset, el script:
- Calcula estadísticas del dataset completo (total, extremos y porcentaje)
- Selecciona únicamente las filas con steering extremo
- Asume que las imágenes están indexadas por el número de fila del CSV
  (rgb/rgb_XXXXXXXX.png y mask_fs/mask_fs_XXXXXXXX.png)
- Copia dichas imágenes desde el dataset original en /logs hacia un directorio temporal en /tmp

python3 check_percen_max_steer.py --csv_path logs/Track5_2/1777830496303_dataset/dataset.csv 
                                  --logs_base logs/Track5_2/1777830496303_dataset/

"""

import os
import shutil
import pandas as pd
import argparse

TMP_BASE = "/tmp/dataset_extract"

# ================= ARGUMENTOS =================
parser = argparse.ArgumentParser(description="Extraer steering extremo por índice")
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--logs_base", type=str, required=True)

args = parser.parse_args()

# ================= CARGAR CSV =================
df = pd.read_csv(args.csv_path)

total = len(df)

# ================= FILTRO =================
mask_extreme = (df["steer"] == 1.0) | (df["steer"] == -1.0)
df_extreme = df[mask_extreme]

filtered = len(df_extreme)
percentage = (filtered / total) * 100 if total > 0 else 0

print("===== ESTADÍSTICAS STEERING EXTREMO =====")
print(f"Total muestras: {total}")
print(f"Muestras extremas: {filtered}")
print(f"Porcentaje: {percentage:.2f}%")

# ================= PREPARAR /tmp =================
if os.path.exists(TMP_BASE):
    shutil.rmtree(TMP_BASE)

rgb_dst_dir = os.path.join(TMP_BASE, "rgb")
mask_dst_dir = os.path.join(TMP_BASE, "mask_fs")

os.makedirs(rgb_dst_dir, exist_ok=True)
os.makedirs(mask_dst_dir, exist_ok=True)

copied = 0
missing = 0

# ================= COPIAR POR ÍNDICE =================
for idx in df_extreme.index:

    filename = f"{idx:08d}"

    rgb_src = os.path.join(args.logs_base, f"rgb/rgb_{filename}.png")
    mask_src = os.path.join(args.logs_base, f"mask_fs/mask_fs_{filename}.png")

    rgb_dst = os.path.join(rgb_dst_dir, f"rgb_{filename}.png")
    mask_dst = os.path.join(mask_dst_dir, f"mask_fs_{filename}.png")

    if os.path.exists(rgb_src):
        shutil.copy2(rgb_src, rgb_dst)
        copied += 1
    else:
        missing += 1

    if os.path.exists(mask_src):
        shutil.copy2(mask_src, mask_dst)
        copied += 1
    else:
        missing += 1

# ================= RESULTADO =================
print("\n===== EXTRACCIÓN COMPLETADA =====")
print(f"Filas extremas: {filtered}")
print(f"Archivos copiados: {copied}")
print(f"Archivos no encontrados: {missing}")
print(f"Destino: {TMP_BASE}")

#!/usr/bin/env python3
"""
Script para visualizar una imagen guardada en formato .npy.

Soporta:
- Imágenes normales (H, W, 3)
- Imágenes grayscale (H, W)
- One hot encoding (H, W, C)

En caso de one hot encoding:
- Se muestra cada canal por separado
- Todo en una única ventana horizontal
"""

import argparse
import numpy as np
import cv2

# ================= ARGUMENTOS =================

parser = argparse.ArgumentParser(description="Visualizar archivo .npy")
parser.add_argument(
    "--npy_path",
    type=str,
    required=True,
    help="Ruta al archivo .npy"
)

args = parser.parse_args()

# ================= CARGAR =================

data = np.load(args.npy_path)

print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Valores únicos:", np.unique(data))

# ================= UTILIDAD PANEL =================

def add_panel(image, title, pad=20, top_pad=40):

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]

    canvas = np.ones(
        (h + top_pad + pad, w + 2 * pad, 3),
        dtype=np.uint8
    ) * 255

    canvas[top_pad:top_pad+h, pad:pad+w] = image

    cv2.putText(
        canvas,
        title,
        (pad, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return canvas

# ================= VISUALIZACIÓN =================

panels = []

# ===== IMAGE RGB =====

if len(data.shape) == 3 and data.shape[2] == 3:

    img = data.astype(np.uint8)
    panels.append(add_panel(img, "RGB Image"))

# ===== ONE HOT =====

elif len(data.shape) == 3:

    channels = data.shape[2]

    for c in range(channels):

        channel = (data[:, :, c] * 255).astype(np.uint8)

        panels.append(
            add_panel(channel, f"Channel {c}")
        )

# ===== GRAYSCALE =====

elif len(data.shape) == 2:

    img = (data * 255).astype(np.uint8)

    panels.append(add_panel(img, "Grayscale"))

else:
    raise RuntimeError("Formato no soportado")

# ================= CONCATENAR =================

final = cv2.hconcat(panels)

# ================= MOSTRAR =================

cv2.imshow("NPY Viewer", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
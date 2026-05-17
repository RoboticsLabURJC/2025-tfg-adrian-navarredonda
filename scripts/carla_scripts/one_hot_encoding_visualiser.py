"""
Script para visualizar máscaras de conos y su representación en one hot encoding.

El script:
- Carga una imagen del dataset
- Detecta píxeles azules y amarillos
- Genera una representación one hot de 2 canales:
    - Canal 0 -> conos azules
    - Canal 1 -> conos amarillos
- Muestra la imagen original y cada canal por separado

Uso:
python3 script.py --img_path ruta/a/la/imagen.png
"""

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Visualizacion de one hot encoding")
parser.add_argument(
    "--img_path",
    type=str,
    required=True,
    help="Ruta a la imagen del dataset"
)

args = parser.parse_args()

# ================= CARGAR IMAGEN =================
img = cv2.imread(args.img_path)

h, w = img.shape[:2]

# ================= ONE HOT =================
blue_mask = np.all(img == [255, 0, 0], axis=2)
yellow_mask = np.all(img == [0, 255, 255], axis=2)

one_hot = np.zeros((h, w, 2), dtype=np.uint8)
one_hot[:, :, 0][blue_mask] = 1
one_hot[:, :, 1][yellow_mask] = 1

print(np.unique(img.reshape(-1, 3), axis=0))

# ================= CANALES (VISUALIZACIÓN) =================
blue_vis = (one_hot[:, :, 0] * 255).astype(np.uint8)
yellow_vis = (one_hot[:, :, 1] * 255).astype(np.uint8)

blue_vis = cv2.cvtColor(blue_vis, cv2.COLOR_GRAY2BGR)
yellow_vis = cv2.cvtColor(yellow_vis, cv2.COLOR_GRAY2BGR)

orig_vis = img.copy()

# ================= PANELES =================

def add_panel(image, title, pad=10, top_pad=35):
    h, w = image.shape[:2]

    canvas = np.ones((h + top_pad + pad, w + 2*pad, 3), dtype=np.uint8) * 255

    canvas[top_pad:top_pad+h, pad:pad+w] = image

    cv2.putText(
        canvas,
        title,
        (pad + 5, int(top_pad * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    return canvas

# ================= CREAR VISTAS =================

p_orig = add_panel(orig_vis, "Original Mask (ROI)")
p_blue = add_panel(blue_vis, "C0 (blue channel)")
p_yellow = add_panel(yellow_vis, "C1 (yellow channel)")

# ================= CONCATENAR =================

final = cv2.hconcat([p_orig, p_blue, p_yellow])

# ================= MOSTRAR =================

cv2.imshow("Dataset Visualization", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
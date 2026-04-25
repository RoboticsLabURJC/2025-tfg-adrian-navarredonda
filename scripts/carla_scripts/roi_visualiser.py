"""
This script is used for the visualization of an image and its ROI.
This is useful to select the ROI we want
"""

import cv2

image = cv2.imread("logs/Track3/1776881146906_dataset/rgb/rgb_00000141.png")

# 1. ESCALAR IMAGEN 
scale = 0.25  # ajusta según necesites

new_width = int(image.shape[1] * scale)
new_height = int(image.shape[0] * scale)

image_scaled = cv2.resize(
    image,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)

# 2. DEFINIR ROI 
x, y = 0, 75   # ahora estos valores son SOBRE la imagen escalada
w, h = 200, 66    # tamaño de ROI

# Copia para dibujar
image_with_rect = image_scaled.copy()

# Dibujar rectángulo rojo
cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 3. RECORTAR ROI 
roi = image_scaled[y:y+h, x:x+w]

# (Opcional) ampliar ROI para verlo mejor
roi_resized = cv2.resize(roi, (w*3, h*3), interpolation=cv2.INTER_NEAREST)

# 4. IGUALAR ALTURAS
h1 = image_with_rect.shape[0]
h2 = roi_resized.shape[0]

if h1 != h2:
    scale_match = h1 / h2
    roi_resized = cv2.resize(
        roi_resized,
        (int(roi_resized.shape[1] * scale_match), h1),
        interpolation=cv2.INTER_NEAREST
    )

# MOSTRAR 
combined = cv2.hconcat([image_with_rect, roi_resized])

cv2.imshow("Scaled Image + ROI", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
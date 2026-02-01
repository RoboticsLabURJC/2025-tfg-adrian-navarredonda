import cv2
import numpy as np

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
IMAGE_PATH = "/home/adrian/Escritorio/tfg/scripts/amz_00032_jpg.rf.eec8b84528d851488e5ce766afe6ca71.jpg"

# Rangos HSV para colores
COLOR_RANGES = {
    # "amarillo": (np.array([20, 50, 100]), np.array([90, 200, 255])),
    # "azul":     (np.array([100, 150, 0]), np.array([200, 255, 255]))

    "amarillo": (np.array([15, 70, 100]), np.array([100, 255, 255])),
    "azul":     (np.array([90, 60, 50]), np.array([140, 255, 255]))
}

# -------------------------------
# CARGAR IMAGEN
# -------------------------------
img = cv2.imread(IMAGE_PATH)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -------------------------------
# FILTROS
# -------------------------------
mask_amarillo = cv2.inRange(img_hsv, COLOR_RANGES['amarillo'][0], COLOR_RANGES['amarillo'][1])
mask_azul = cv2.inRange(img_hsv, COLOR_RANGES['azul'][0], COLOR_RANGES['azul'][1])

# Aplicar máscaras para mostrar solo los colores
res_amarillo = cv2.bitwise_and(img, img, mask=mask_amarillo)
res_azul = cv2.bitwise_and(img, img, mask=mask_azul)

# -------------------------------
# MOSTRAR RESULTADOS
# -------------------------------
cv2.imshow("Original", img)
cv2.imshow("Filtro Amarillo", res_amarillo)
cv2.imshow("Filtro Azul", res_azul)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# ================= CARGAR IMAGEN =================
img = cv2.imread("logs/Track3/1777136391344_dataset/mask_fs/mask_fs_0000167.png")

h, w = img.shape[:2]

# ================= ONE HOT =================
blue_mask = np.all(img == [255, 0, 0], axis=2)
yellow_mask = np.all(img == [0, 255, 255], axis=2)

one_hot = np.zeros((h, w, 2), dtype=np.uint8)
one_hot[:, :, 0][blue_mask] = 1
one_hot[:, :, 1][yellow_mask] = 1

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

final = cv2.vconcat([p_orig, p_blue, p_yellow])

# ================= MOSTRAR =================

cv2.imshow("Dataset Visualization", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
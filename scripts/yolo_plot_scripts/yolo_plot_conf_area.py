import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV con datos
df = pd.read_csv("caracteristicas_train_servidor.csv")

# Calcualr area del bounding box
df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

# Filtrar detecciones válidas
df = df[df["area"] > 0].copy()
df = df[df["confidencia"] > 0].copy()

# Normalizar area
df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
df["conf_norm"] = (df["confidencia"] - df["confidencia"].min()) / (df["confidencia"].max() - df["confidencia"].min())

# Hacer plot
plt.figure(figsize=(8, 6))
plt.scatter(df["area_norm"], df["conf_norm"], s=10, alpha=0.7)
plt.xlabel("Área normalizada del bounding box")
plt.ylabel("Confidencia normalizada")
plt.title("Relación entre Área del Bounding Box y Confidencia")
plt.grid(True)
plt.show()

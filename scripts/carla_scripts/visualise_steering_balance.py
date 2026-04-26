import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= CARGAR CSV =================
csv_path = "logs/Track3/1777136391344_dataset/dataset.csv"
df = pd.read_csv(csv_path)

steer = df["steer"]

# ================= CLASIFICACIÓN =================
left = steer[steer < -0.1]
straight = steer[(steer >= -0.1) & (steer <= 0.1)]
right = steer[steer > 0.1]

# ================= BINS GLOBALES =================
# 30 bins uniformes en todo el rango [-1, 1]
bins = np.linspace(-1.0, 1.0, 51)

# ================= HISTOGRAMAS =================
hist_left, _ = np.histogram(left, bins=bins)
hist_straight, _ = np.histogram(straight, bins=bins)
hist_right, _ = np.histogram(right, bins=bins)

# mismo eje Y para todos
y_max = max(hist_left.max(), hist_straight.max(), hist_right.max())

# ================= PRINT INFO =================
print("Máxima frecuencia por categoría:")
print(f"Izquierda: {hist_left.max()}")
print(f"Recto: {hist_straight.max()}")
print(f"Derecha: {hist_right.max()}")

# ================= PLOT =================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# ===== IZQUIERDA =====
axes[0].hist(left, bins=bins, color="blue", alpha=0.7)
axes[0].set_title("Giros izquierda")
axes[0].set_xlim(-1.0, -0.1)
axes[0].set_ylim(0, y_max)
axes[0].set_xlabel("Steer")
axes[0].set_ylabel("Frecuencia")

# ===== RECTO =====
axes[1].hist(straight, bins=bins, color="green", alpha=0.7)
axes[1].set_title("Recto")
axes[1].set_xlim(-0.1, 0.1)
axes[1].set_ylim(0, y_max)
axes[1].set_xlabel("Steer")

# ===== DERECHA =====
axes[2].hist(right, bins=bins, color="red", alpha=0.7)
axes[2].set_title("Giros derecha")
axes[2].set_xlim(0.1, 1.0)
axes[2].set_ylim(0, y_max)
axes[2].set_xlabel("Steer")

# ================= MOSTRAR EJE Y EN TODAS =================
for ax in axes:
    ax.tick_params(labelleft=True)

# ================= ESPACIADO =================
plt.subplots_adjust(wspace=0.3)

plt.show()
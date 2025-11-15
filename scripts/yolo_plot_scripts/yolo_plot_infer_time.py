import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("inference_times.csv")

# Ignorar primer frame si es un outlier extremo
df_plot = df[df["frame_id"] != 1]

# Suavizado con media móvil
window_size = 10
df_plot["smoothed"] = df_plot["inference_time_ms"].rolling(window=window_size, min_periods=1).mean()

# Crear figura
plt.figure(figsize=(12, 6))

# Línea original (opcional, muy fina)
plt.plot(df_plot["frame_id"], df_plot["inference_time_ms"], color="lightblue", linewidth=0.8, alpha=0.5, label="Medida original")

# Línea suavizada
plt.plot(df_plot["frame_id"], df_plot["smoothed"], color="blue", linewidth=2, label=f"Media suavizada ({window_size} frames)")

# Etiquetas y título
plt.xlabel("Frame ID", fontsize=12)
plt.ylabel("Tiempo de inferencia (ms)", fontsize=12)
plt.title("Evolución del tiempo de inferencia por frame (suavizado)", fontsize=14)

# Leyenda
plt.legend()

# Cuadrícula
plt.grid(True, alpha=0.3)

# Limitar eje Y para eliminar outliers
max_y = 45
plt.ylim(0, max_y * 1.1)

plt.tight_layout()
plt.show()




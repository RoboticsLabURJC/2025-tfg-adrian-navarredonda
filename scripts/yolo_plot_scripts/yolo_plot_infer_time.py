import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("caracteristicas_train_portatil.csv")
df['tiempo_inferencia_ms'] = df['tiempo_inferencia_ms'].astype(float)

# Filtrar valores <= 100 ms
df_filtered = df[df['tiempo_inferencia_ms'] <= 100].copy()

# Quitar filas repetidas
tiempos_unicos = df_filtered.groupby('imagen')['tiempo_inferencia_ms'].first().reset_index()
tiempos_unicos = tiempos_unicos.sort_index()  # Orden original

# Calcular media movil
window_size = 50
tiempos_unicos['media_movil'] = tiempos_unicos['tiempo_inferencia_ms'].rolling(window=window_size, min_periods=1).mean()

# plot datos individuales claros + media movil
plt.figure(figsize=(12,6))

# Datos individuales
plt.plot(tiempos_unicos['tiempo_inferencia_ms'], linestyle='-', color='blue', alpha=0.3, markersize=4, label='Datos individuales')

# Media movil
plt.plot(tiempos_unicos['media_movil'], color='red', linestyle='-', linewidth=2, label=f'Media movil ({window_size})')

plt.xlabel("Índice de imagen")
plt.ylabel("Tiempo de inferencia (ms)")
plt.title(f"Tiempo de inferencia suavizado (media movil de {window_size} medidas)")
plt.grid(True)
plt.legend()
plt.show()

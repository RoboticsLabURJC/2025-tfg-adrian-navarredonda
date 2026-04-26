import pandas as pd
import matplotlib.pyplot as plt

# ================= CARGAR CSV =================
csv_path = "logs/Track3/1777136391344_dataset/dataset.csv"
df = pd.read_csv(csv_path)

steer = df["steer"]

# ================= CLASIFICACIÓN =================
left_count = (steer < -0.1).sum()
straight_count = ((steer >= -0.1) & (steer <= 0.1)).sum()
right_count = (steer > 0.1).sum()

# ================= DATOS =================
categories = ["Izquierda", "Recto", "Derecha"]
counts = [left_count, straight_count, right_count]

# ================= PLOT =================
plt.figure(figsize=(7, 5))
plt.bar(categories, counts)

plt.title("Balance del dataset (steering)")
plt.ylabel("Número de muestras")

# Mostrar valores encima de las barras
for i, v in enumerate(counts):
    plt.text(i, v + 5, str(v), ha="center")

plt.grid(axis="y", alpha=0.3)
plt.show()

# ================= PRINT DEBUG =================
print("Izquierda:", left_count)
print("Recto:", straight_count)
print("Derecha:", right_count)
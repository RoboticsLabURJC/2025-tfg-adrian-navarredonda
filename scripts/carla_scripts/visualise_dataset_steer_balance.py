import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Visualización de distribución de steering")
parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help="Ruta al archivo CSV del dataset"
)

args = parser.parse_args()
# ================= CARGAR CSV =================
df = pd.read_csv(args.csv_path)

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
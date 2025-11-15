import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("confidences.csv")

# Crear figura
plt.figure(figsize=(10, 6))

# Colores por clase
colors = {
    0: "blue",
    1: "red",
    2: "orange",
    3: "gray",
    4: "yellow"
}

# Nombres personalizados de la leyenda
class_labels = {
    0: "Cono azul",
    1: "Cono naranja grande",
    2: "Cono naranja",
    3: "clase desconocida",
    4: "Cono amarillo"
}

# Plot de histogramas separados por clase
for cls_id, color in colors.items():
    subset = df[df["class"] == cls_id]["confidence"]
    if len(subset) > 0:
        plt.hist(
            subset,
            bins=50,
            alpha=0.5,
            color=color,
            label=class_labels[cls_id]  #  aquí cambiamos el texto de la leyenda
        )

# Etiquetas
plt.xlabel("Confidence", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Confidence Distribution per Class", fontsize=14)

# Leyenda
plt.legend(title="Clases detectadas")

# Mostrar
plt.show()

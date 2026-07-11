"""
Prepara los datasets de circuitos para entrenar PilotNet.

Lee todos los CSVs de los circuitos indicados, hace un split 70/15/15
y genera tres carpetas con la estructura que espera PilotNetDataset:

    output_dir/
    ├── Deepracer_BaseMap_train/
    │   └── dataset.csv   (columnas: mask_path, steer, throttle)
    ├── Deepracer_BaseMap_val/
    │   └── dataset.csv
    └── Deepracer_BaseMap_test/
        └── dataset.csv

Las rutas de imagen en los CSV generados son absolutas, por lo que
no es necesario copiar ningún archivo: PilotNetDataset las carga
directamente desde su ubicación original.

Uso:
----
    python prepare_dataset.py \\
        --circuit_dirs /ruta/circuito1 /ruta/circuito2 ... \\
        --image_col rgb_path \\
        --output_dir ./datasets \\
        --train_ratio 0.70 \\
        --val_ratio   0.15 \\
        --seed        42

Columnas de imagen disponibles según tu CSV:
    rgb_path        → imagen RGB original
    cones_ohe       → conos con one-hot encoding
    bb_box_ohe      → bounding boxes con one-hot encoding
    spline_ohe      → splines con one-hot encoding
"""

import os
import csv
import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepara splits train/val/test para PilotNet")

    parser.add_argument(
        "--circuit_dirs", nargs="+", required=True,
        help="Rutas a las carpetas de cada circuito (cada una debe tener un dataset.csv)"
    )
    parser.add_argument(
        "--image_col", type=str, default="rgb_path",
        choices=["rgb_path", "cones_ohe", "bb_box_ohe", "spline_ohe"],
        help="Columna del CSV original a usar como imagen de entrada (default: rgb_path)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./datasets",
        help="Carpeta base donde se crearán train/val/test (default: ./datasets)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.70,
        help="Proporción de datos para entrenamiento (default: 0.70)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Proporción de datos para validación (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para reproducibilidad del shuffle (default: 42)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Mezclar los datos antes de hacer el split (default: True)"
    )
    parser.add_argument(
        "--per_circuit", action="store_true", default=False,
        help="Si se activa, hace el split 70/15/15 por circuito en lugar de "
             "mezclar todos los circuitos juntos antes de splitear. "
             "Recomendado si los circuitos tienen distribuciones muy distintas."
    )

    return parser.parse_args()


def load_circuit(circuit_dir, image_col):
    """
    Lee el dataset.csv de un circuito y devuelve una lista de dicts:
        [{"img_abs": <ruta absoluta>, "steer": float, "throttle": float}, ...]

    Las rutas de imagen en el CSV original son relativas (empiezan con /).
    Se convierten a absolutas usando circuit_dir como raíz.
    """
    csv_path = os.path.join(circuit_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] No se encontró dataset.csv en: {circuit_dir}")
        return []

    rows = []
    skipped = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        # Verificar columnas necesarias
        required = {image_col, "steer", "throttle"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"[ERROR] Faltan columnas {missing} en {csv_path}")
            return []

        for row in reader:
            img_rel = row[image_col].lstrip("/")
            img_abs = os.path.abspath(os.path.join(circuit_dir, img_rel))

            if not os.path.isfile(img_abs):
                skipped += 1
                continue

            try:
                steer    = float(row["steer"])
                throttle = float(row["throttle"])
            except ValueError:
                skipped += 1
                continue

            rows.append({
                "img_abs":  img_abs,
                "steer":    steer,
                "throttle": throttle,
            })

    if skipped:
        print(f"[WARN] {circuit_dir}: {skipped} filas ignoradas (imagen no encontrada o valor inválido)")

    print(f"  Cargadas {len(rows):>6} muestras  ←  {circuit_dir}")
    return rows


def steer_category(steer):
    """Categoriza el steer en tres clases."""
    if steer < -0.1:
        return "left"
    elif steer > 0.1:
        return "right"
    else:
        return "straight"


def split_rows(rows, train_ratio, val_ratio, seed, shuffle):
    """
    Split ESTRATIFICADO por categoría de steer (left / straight / right).

    Cada categoría se divide en 70/15/15 por separado y luego se unen,
    garantizando que los tres splits tengan la misma distribución de giros
    sin eliminar ningún dato.
    """
    # Separar por categoría
    buckets = {"left": [], "straight": [], "right": []}
    for row in rows:
        buckets[steer_category(row["steer"])].append(row)

    # Mostrar distribución original
    total = len(rows)
    print("\n  Distribución de steer:")
    for cat, cat_rows in buckets.items():
        pct = 100 * len(cat_rows) / total if total else 0
        print(f"    {cat:>8}: {len(cat_rows):>6} muestras ({pct:.1f}%)")

    train, val, test = [], [], []

    for cat, cat_rows in buckets.items():
        if shuffle:
            random.seed(seed)
            random.shuffle(cat_rows)

        n       = len(cat_rows)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train.extend(cat_rows[:n_train])
        val.extend(cat_rows[n_train:n_train + n_val])
        test.extend(cat_rows[n_train + n_val:])

    # Mezclar de nuevo para que los splits no queden ordenados por categoría
    if shuffle:
        random.seed(seed)
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

    return train, val, test


def write_split_csv(rows, out_dir, split_name):
    """
    Escribe el dataset.csv en out_dir/<split_name>/
    con las columnas que espera PilotNetDataset: mask_path, steer, throttle.

    Se usa 'mask_path' como nombre de columna porque es el que lee
    PilotNetDataset, independientemente de qué columna de imagen usamos.
    La ruta escrita es absoluta para no tener que copiar imágenes.
    """
    folder = os.path.join(out_dir, f"Deepracer_BaseMap_{split_name}")
    os.makedirs(folder, exist_ok=True)

    csv_path = os.path.join(folder, "dataset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_path", "steer", "throttle"])
        for row in rows:
            writer.writerow([row["img_abs"], row["steer"], row["throttle"]])

    print(f"  [{split_name:>5}]  {len(rows):>6} muestras  →  {csv_path}")
    return folder


def main():
    args = parse_args()

    test_ratio = round(1.0 - args.train_ratio - args.val_ratio, 6)
    assert test_ratio > 0, "train_ratio + val_ratio debe ser < 1.0"

    print(f"\nSplit: train={args.train_ratio:.0%}  val={args.val_ratio:.0%}  test={test_ratio:.0%}")
    print(f"Columna de imagen: {args.image_col}")
    print(f"Salida: {os.path.abspath(args.output_dir)}\n")

    # ── Cargar todos los circuitos ───────────────────────────────────────────
    all_rows = []
    for circuit_dir in args.circuit_dirs:
        rows = load_circuit(circuit_dir, args.image_col)
        all_rows.extend(rows)

    if not all_rows:
        print("\n[ERROR] No se cargó ninguna muestra. Revisa las rutas y columnas.")
        return

    print(f"\nTotal muestras cargadas: {len(all_rows)}")

    # ── Split ────────────────────────────────────────────────────────────────
    if args.per_circuit:
        # Split independiente por circuito para mantener distribución
        print("\nModo: split POR CIRCUITO")
        train_rows, val_rows, test_rows = [], [], []
        for circuit_dir in args.circuit_dirs:
            rows = load_circuit(circuit_dir, args.image_col)
            if not rows:
                continue
            tr, va, te = split_rows(rows, args.train_ratio, args.val_ratio,
                                     args.seed, args.shuffle)
            train_rows.extend(tr)
            val_rows.extend(va)
            test_rows.extend(te)
    else:
        # Mezclar todos los circuitos juntos y luego splitear
        print("\nModo: split GLOBAL (todos los circuitos mezclados)")
        train_rows, val_rows, test_rows = split_rows(
            all_rows, args.train_ratio, args.val_ratio,
            args.seed, args.shuffle
        )

    # ── Escribir CSVs ────────────────────────────────────────────────────────
    print("\nEscribiendo splits:")
    train_folder = write_split_csv(train_rows, args.output_dir, "train")
    val_folder   = write_split_csv(val_rows,   os.path.join(args.output_dir, "validation"), "val")
    test_folder  = write_split_csv(test_rows,  os.path.join(args.output_dir, "test"),       "test")

    print(f"\n✓ Listo. Para entrenar ejecuta:")
    print(f"""
  bash train.sh

  O directamente:

  python train_final.py \\
    --data_dir {train_folder} \\
    --val_dir  {val_folder} \\
    --test_dir {test_folder} \\
    --num_epochs 100 --batch_size 128 --lr 3e-4 \\
    --base_dir mi_experimento --print_terminal
""")


if __name__ == "__main__":
    main()
"""
Convierte el CSV de conos (ID,color,x,y,z,ScaleX,ScaleY,ScaleZ) que usa el
Blueprint de colocacion en un reftrack.csv valido para
TUMFTM/global_racetrajectory_optimization: [x_m, y_m, w_tr_right_m, w_tr_left_m]

No resamplea ni suaviza la linea: el propio main_globaltraj.py ya hace
interpolacion/spline regression internamente (prep_track.py), asi que aqui
solo generamos los puntos ordenados y emparejados, con espaciado irregular
si hace falta.
"""

import numpy as np
import pandas as pd

# --- CONFIG: ajusta esto a tu CSV -----------------------------------------
INPUT_CSV = "../../Carla_uassets/maps/map_csvs/track9random_track.csv"
OUTPUT_CSV = "reftrack.csv"
COLOR_LEFT = "blue"      # valor exacto de la columna 'color' para el lado izquierdo
COLOR_RIGHT = "yellow"   # valor exacto de la columna 'color' para el lado derecho
PLOT = True              # pon a False para no mostrar la grafica de verificacion
# ---------------------------------------------------------------------------


def order_loop(points: np.ndarray) -> np.ndarray:
    """Ordena un conjunto de puntos 2D en un lazo cerrado por vecino mas cercano.

    Limitacion conocida: si hay huecos grandes e irregulares entre conos,
    la cadena de vecino mas cercano puede "saltar" por el hueco. Revisa
    la grafica de verificacion antes de fiarte del resultado.
    """
    remaining = list(range(len(points)))
    order = [remaining.pop(0)]
    while remaining:
        last = points[order[-1]]
        dists = [np.hypot(*(points[i] - last)) for i in remaining]
        nxt = remaining[int(np.argmin(dists))]
        order.append(nxt)
        remaining.remove(nxt)
    return points[order]


def main():
    df = pd.read_csv(INPUT_CSV)

    print("Valores de color encontrados:", df["color"].unique())

    left = df[df["color"] == COLOR_LEFT][["x", "y"]].to_numpy()
    right = df[df["color"] == COLOR_RIGHT][["x", "y"]].to_numpy()
    print(f"Conos izquierda: {len(left)}, conos derecha: {len(right)}")

    if len(left) == 0 or len(right) == 0:
        raise ValueError(
            "No se ha encontrado ningun cono para uno de los dos lados: "
            "revisa COLOR_LEFT / COLOR_RIGHT contra los valores impresos arriba."
        )

    # ordena solo el lado izquierdo a lo largo del circuito
    left_ordered = order_loop(left)

    # para cada punto ya ordenado de la izquierda, busca el cono derecho mas
    # cercano en el espacio (evita el problema de que las dos cadenas de
    # vecino mas cercano vayan en sentidos o puntos de partida distintos)
    right_matched = np.array(
        [right[np.argmin(np.hypot(*(right - p).T))] for p in left_ordered]
    )

    centerline = (left_ordered + right_matched) / 2.0
    w_tr_left = np.hypot(*(left_ordered - centerline).T)
    w_tr_right = np.hypot(*(right_matched - centerline).T)

    reftrack = np.column_stack(
        [centerline[:, 0], centerline[:, 1], w_tr_right, w_tr_left]
    )

    np.savetxt(
        OUTPUT_CSV,
        reftrack,
        delimiter=",",
        header="x_m,y_m,w_tr_right_m,w_tr_left_m",
        comments="# ",
        fmt="%.4f",
    )
    print(f"Guardado {OUTPUT_CSV} con {len(reftrack)} puntos")

    if PLOT:
        import matplotlib.pyplot as plt

        plt.plot(left[:, 0], left[:, 1], "o", color="tab:blue", label="conos izquierda")
        plt.plot(right[:, 0], right[:, 1], "o", color="tab:orange", label="conos derecha")
        plt.plot(centerline[:, 0], centerline[:, 1], "-", color="black", label="centerline")
        plt.axis("equal")
        plt.legend()
        plt.title("Verificacion visual del reftrack generado")
        plt.show()


if __name__ == "__main__":
    main()
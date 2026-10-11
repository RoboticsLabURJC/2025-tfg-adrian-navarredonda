
#!/usr/bin/env python3
"""
Convert a CARLA cone CSV into a TUMFTM reftrack.csv:
[x_m, y_m, w_tr_right_m, w_tr_left_m].

Blue, yellow and large orange cones contribute to the track boundaries.
Coordinates are converted from cm to m when CM_TO_M=True.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --- CONFIG ---------------------------------------------------------------
INPUT_CSV = "../../../Carla_uassets/maps/modify_tracks/track1_modify.csv"
OUTPUT_CSV = "reftrack_1.csv"
COLOR_LEFT = "blue"
COLOR_RIGHT = "yellow"
PLOT = True
CM_TO_M = True
DEDUP_TOL = 0.02  # Minimum distance between duplicate boundary points, in metres
# -------------------------------------------------------------------------


def order_loop(points: np.ndarray) -> np.ndarray:
    """Order boundary points using nearest-neighbour traversal."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("Each boundary must contain at least 3 points.")

    # Remove near-duplicate points to avoid zero-length jumps.
    unique = [points[0]]
    for p in points[1:]:
        if min(np.linalg.norm(p - q) for q in unique) > DEDUP_TOL:
            unique.append(p)
    points = np.asarray(unique)

    if len(points) < 3:
        raise ValueError("Too few unique points remain on a boundary.")

    remaining = list(range(1, len(points)))
    order = [0]
    while remaining:
        last = points[order[-1]]
        distances = np.linalg.norm(points[remaining] - last, axis=1)
        j = int(np.argmin(distances))
        order.append(remaining.pop(j))

    return points[order]


def nearest_distances(points, reference):
    """Distance from each point to its nearest point in a reference boundary."""
    return np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=2).min(axis=1)


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)
    required = {"color", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(df.columns))}")

    colors = df["color"].astype(str).str.strip().str.lower()
    print("Colors found:", df["color"].unique())

    blue = df.loc[colors == COLOR_LEFT, ["x", "y"]].to_numpy(dtype=float)
    yellow = df.loc[colors == COLOR_RIGHT, ["x", "y"]].to_numpy(dtype=float)
    orange_mask = colors.str.contains("orange", na=False)
    orange = df.loc[orange_mask, ["x", "y"]].to_numpy(dtype=float)

    if CM_TO_M:
        blue, yellow, orange = blue / 100.0, yellow / 100.0, orange / 100.0

    if len(blue) < 3 or len(yellow) < 3:
        raise ValueError(
            f"Need at least 3 blue and yellow cones; found blue={len(blue)}, yellow={len(yellow)}. "
            f"Check COLOR_LEFT/COLOR_RIGHT against the printed colors."
        )

    # Assign each orange cone to the nearest original boundary.
    if len(orange):
        d_blue = nearest_distances(orange, blue)
        d_yellow = nearest_distances(orange, yellow)
        orange_left = orange[d_blue <= d_yellow]
        orange_right = orange[d_blue > d_yellow]
        blue_all = np.vstack((blue, orange_left))
        yellow_all = np.vstack((yellow, orange_right))
    else:
        orange_left = orange_right = np.empty((0, 2))
        blue_all, yellow_all = blue.copy(), yellow.copy()

    print(f"Blue boundary cones: {len(blue)}")
    print(f"Yellow boundary cones: {len(yellow)}")
    print(f"Orange cones assigned to blue boundary: {len(orange_left)}")
    print(f"Orange cones assigned to yellow boundary: {len(orange_right)}")

    # Include the orange cones as actual boundary points.
    left_ordered = order_loop(blue_all)
    right_ordered = order_loop(yellow_all)

    # Match each left-boundary point to the nearest right-boundary point.
    right_matched = right_ordered[
        np.argmin(
            np.linalg.norm(left_ordered[:, None, :] - right_ordered[None, :, :], axis=2),
            axis=1
        )
    ]

    centerline = (left_ordered + right_matched) / 2.0
    distances_left = np.linalg.norm(left_ordered - centerline, axis=1)
    distances_right = np.linalg.norm(right_matched - centerline, axis=1)

    # TUMFTM expects right width first, then left width.
    reftrack = np.column_stack((
        centerline[:, 0], centerline[:, 1], distances_right, distances_left
    ))

    np.savetxt(
        OUTPUT_CSV, reftrack, delimiter=",",
        header="x_m,y_m,w_tr_right_m,w_tr_left_m",
        comments="# ", fmt="%.4f"
    )
    print(f"Saved {OUTPUT_CSV}: {len(reftrack)} reference points")

    if PLOT:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(blue[:, 0], blue[:, 1], "o", color="tab:blue",
                markersize=4, label="Blue cones")
        ax.plot(yellow[:, 0], yellow[:, 1], "o", color="gold",
                markersize=4, label="Yellow cones")
        if len(orange):
            ax.scatter(orange[:, 0], orange[:, 1], marker="s", s=75,
                       facecolors="none", edgecolors="red", label="Original orange cones")
        if len(orange_left):
            ax.scatter(orange_left[:, 0], orange_left[:, 1], marker="s",
                       s=45, color="deepskyblue", label="Orange assigned to blue")
        if len(orange_right):
            ax.scatter(orange_right[:, 0], orange_right[:, 1], marker="s",
                       s=45, color="darkorange", label="Orange assigned to yellow")

        ax.plot(left_ordered[:, 0], left_ordered[:, 1], "--",
                color="blue", alpha=0.65, label="Ordered blue boundary")
        ax.plot(right_ordered[:, 0], right_ordered[:, 1], "--",
                color="orange", alpha=0.65, label="Ordered yellow boundary")
        ax.plot(centerline[:, 0], centerline[:, 1], "-k",
                linewidth=1.2, label="Generated centerline")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Reftrack verification, including orange cones")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

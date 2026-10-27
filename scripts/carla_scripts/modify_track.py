
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resample_closed(points, spacing=None, n=None):
    """Resample a closed polyline uniformly by arc length."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("Each track boundary needs at least 3 points.")

    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    valid = lengths > 1e-9
    closed = np.vstack((closed[0], closed[1:][valid]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    perimeter = cumulative[-1]

    if n is None:
        n = max(3, int(round(perimeter / spacing)))
    distances = np.linspace(0.0, perimeter, n, endpoint=False)
    return np.column_stack((np.interp(distances, cumulative, closed[:, 0]),
                            np.interp(distances, cumulative, closed[:, 1])))


def offset_boundaries(center, lane_width):
    """Create two boundaries at half the lane width from the centerline."""
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm < 1e-9] = 1.0
    tangent /= norm[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return center + lane_width / 2.0 * normal, center - lane_width / 2.0 * normal


def project_orange_cones(orange_df, blue_original, yellow_original, blue_new, yellow_new):
    """Place each large orange cone on its nearest original boundary's modified counterpart."""
    if orange_df.empty:
        return orange_df.copy()

    orange_new = orange_df.copy()
    blue_ref = resample_closed(blue_original, n=10000)
    yellow_ref = resample_closed(yellow_original, n=10000)
    blue_target = resample_closed(blue_new, n=10000)
    yellow_target = resample_closed(yellow_new, n=10000)

    for idx, row in orange_df.iterrows():
        point = row[["x", "y"]].to_numpy(dtype=float)
        db = np.sum((blue_ref - point) ** 2, axis=1)
        dy = np.sum((yellow_ref - point) ** 2, axis=1)

        if db.min() <= dy.min():
            ref, target = blue_ref, blue_target
            nearest = int(np.argmin(db))
        else:
            ref, target = yellow_ref, yellow_target
            nearest = int(np.argmin(dy))

        # Transfer the normalized position along the original boundary.
        fraction = nearest / len(ref)
        target_idx = int(round(fraction * len(target))) % len(target)
        orange_new.loc[idx, ["x", "y"]] = target[target_idx]

    return orange_new


def main():
    parser = argparse.ArgumentParser(
        description="Modify track width and cone spacing while keeping a fixed centerline."
    )
    parser.add_argument("--input", default="track1random_track.csv", help="Input CSV file.")
    parser.add_argument("--output", default="track_modified.csv", help="Output CSV file.")
    parser.add_argument("--width", type=float, default=400.0, help="Distance between blue and yellow rows, in cm.")
    parser.add_argument("--spacing", type=float, default=800.0, help="Target distance between cones, in cm.")
    parser.add_argument("--plot", default="track_comparison.png", help="Output comparison plot.")
    args = parser.parse_args()

    if args.width <= 0 or args.spacing <= 0:
        parser.error("--width and --spacing must be positive.")

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"ID", "color", "x", "y", "z", "ScaleX", "ScaleY", "ScaleZ"}
    if not required.issubset(df.columns):
        parser.error(f"Missing CSV columns: {sorted(required - set(df.columns))}")

    colors = df["color"].astype(str).str.strip().str.lower()
    blue_df = df[colors == "blue"].copy()
    yellow_df = df[colors == "yellow"].copy()
    orange_mask = colors.str.contains("orange", na=False)
    orange_df = df[orange_mask].copy()

    if len(blue_df) < 3 or len(yellow_df) < 3:
        parser.error("The CSV must contain at least 3 blue and 3 yellow cones.")

    blue_original = blue_df[["x", "y"]].to_numpy(dtype=float)
    yellow_original = yellow_df[["x", "y"]].to_numpy(dtype=float)

    # Reconstruct the centerline independently of the requested width and spacing.
    n_center = max(1000, 4 * max(len(blue_original), len(yellow_original)))
    blue_aligned = resample_closed(blue_original, n=n_center)
    yellow_aligned = resample_closed(yellow_original, n=n_center)
    center = (blue_aligned + yellow_aligned) / 2.0

    perimeter = np.linalg.norm(np.diff(np.vstack((center, center[0])), axis=0), axis=1).sum()
    n_cones = max(3, int(round(perimeter / args.spacing)))
    center_samples = resample_closed(center, n=n_cones)
    blue_new, yellow_new = offset_boundaries(center_samples, args.width)

    # Generate boundary cones while preserving their original per-color properties.
    blue_rows, yellow_rows = [], []
    for b, y in zip(blue_new, yellow_new):
        rb, ry = blue_df.iloc[0].copy(), yellow_df.iloc[0].copy()
        rb[["x", "y"]] = b
        ry[["x", "y"]] = y
        blue_rows.append(rb)
        yellow_rows.append(ry)

    # Large orange cones remain in the CSV and retain their own scales and z.
    orange_new = project_orange_cones(
        orange_df, blue_original, yellow_original, blue_new, yellow_new
    )

    # Preserve all objects that are neither boundary cones nor orange cones.
    other = df[~(colors.isin(["blue", "yellow"]) | orange_mask)].copy()
    result = pd.concat(
        [pd.DataFrame(blue_rows), pd.DataFrame(yellow_rows), orange_new, other],
        ignore_index=True
    )
    result["ID"] = np.arange(1, len(result) + 1)
    result = result[df.columns]
    result.to_csv(args.output, index=False)

    # Plot original and modified track.
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(blue_original[:, 0], blue_original[:, 1], s=18, c="blue",
               label="Original blue", alpha=0.65)
    ax.scatter(yellow_original[:, 0], yellow_original[:, 1], s=18, c="gold",
               label="Original yellow", alpha=0.65)
    ax.scatter(blue_new[:, 0], blue_new[:, 1], s=15, c="deepskyblue",
               marker="x", label="Modified blue")
    ax.scatter(yellow_new[:, 0], yellow_new[:, 1], s=15, c="orange",
               marker="x", label="Modified yellow")

    if not orange_df.empty:
        ax.scatter(orange_df["x"], orange_df["y"], s=75, facecolors="none",
                   edgecolors="red", linewidths=1.5, label="Original large orange")
        ax.scatter(orange_new["x"], orange_new["y"], s=75, c="red",
                   marker="s", label="Modified large orange")

    ax.plot(center[:, 0], center[:, 1], "k--", linewidth=0.8,
            alpha=0.45, label="Reconstructed centerline")
    ax.set_title(f"Track comparison — width: {args.width:g} cm, spacing: {args.spacing:g} cm")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=180)
    plt.show()

    print(f"Original cones: {len(df)}")
    print(f"Modified blue cones: {len(blue_new)}")
    print(f"Modified yellow cones: {len(yellow_new)}")
    print(f"Large orange cones repositioned: {len(orange_new)}")
    print(f"Centerline length: {perimeter:.2f} cm ({perimeter / 100:.2f} m)")
    print(f"Approximate spacing: {perimeter / n_cones:.2f} cm")
    print(f"Saved CSV: {args.output}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()

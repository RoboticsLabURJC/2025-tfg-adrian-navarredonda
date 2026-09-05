import carla
import csv
import math
import time
import os
import argparse
import numpy as np
import pygame
import random

# ============================================================
# CONFIG
# ============================================================
VEHICLE_MODEL = "vehicle.kart.kart"
CONTROL_HZ = 30.0
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720

# ============================================================
# STEERING CONTROLLER
# ============================================================
STEERING_KP = 1.5
MAX_STEER = 1.0

# ============================================================
# ALINEACION TRAYECTORIA <-> CIRCUITO
# ============================================================
TRACK = "Track 4"

# (rotacion_deg, espejo_x, espejo_y, offset_x, offset_y)
TRACK_ALIGNMENTS = {
    "Track 1": (-90.0, False, True, 33.5, -33.5),
    "Track 2": (-90.0, False, True, 33.5, -33.5),
    "Track 3": (-90.0, False, True, 33.5, -33.5),
    "Track 4": (-90.0, False, True, 69.0, -70.0),
    "Track 5": (-90.0, False, True, 33.5, -33.5),
    "Track 6": (-90.0, False, True, 33.5, -33.5),
    "Track 7": (-90.0, False, True, 33.5, -33.5),
    "Track 8": (-90.0, False, True, 33.5, -33.5),
    "Track 9": (-90.0, False, True, 33.5, -33.5),
}

if TRACK not in TRACK_ALIGNMENTS:
    raise ValueError(f"TRACK='{TRACK}' no definido.")

(
    ALIGN_ROTATION_DEG,
    ALIGN_MIRROR_X,
    ALIGN_MIRROR_Y,
    ALIGN_OFFSET_X,
    ALIGN_OFFSET_Y
) = TRACK_ALIGNMENTS[TRACK]

# ============================================================
# LOOKAHEAD
# ============================================================
LOOKAHEAD_MIN = 3.0
LOOKAHEAD_SPEED_GAIN = 0.30
LOOKAHEAD_MAX = 8.0

# ============================================================
# SPEED CONTROLLER
# ============================================================
SPEED_KP = 0.25
SPEED_KI = 0.02
SPEED_KD = 0.02

MAX_THROTTLE = 0.5
MAX_BRAKE = 1.0

SPEED_BRAKE_THRESHOLD = 0.5

# ============================================================
# CAMERAS
# ============================================================
CAMERA_FOV = 90.0
TOP_CAMERA_HEIGHT = 80.0

KART_CAMERA_X = 1.2
KART_CAMERA_Z = 1.4
KART_CAMERA_PITCH = -5.0
KART_CAMERA_FOV = 100.0


# ============================================================
# UTILIDADES
# ============================================================
def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def align_point(x, y, psi):
    theta = math.radians(ALIGN_ROTATION_DEG)

    c = math.cos(theta)
    s = math.sin(theta)

    x, y, psi = (
        x * c - y * s,
        x * s + y * c,
        psi + theta
    )

    if ALIGN_MIRROR_X:
        x = -x
        psi = math.pi - psi

    if ALIGN_MIRROR_Y:
        y = -y
        psi = -psi

    return (
        x + ALIGN_OFFSET_X,
        y + ALIGN_OFFSET_Y,
        normalize_angle(psi)
    )


def load_trajectory(filename):
    trajectory = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(";")]

            if len(parts) < 7:
                continue

            try:
                s, x, y, psi, kappa, vx, ax = map(
                    float,
                    parts[:7]
                )
            except ValueError:
                continue

            x, y, psi = align_point(x, y, psi)

            trajectory.append({
                "s": s,
                "x": x,
                "y": y,
                "psi": psi,
                "kappa": kappa,
                "vx": vx,
                "ax": ax
            })

    if len(trajectory) < 3:
        raise RuntimeError(
            "La trayectoria contiene menos de 3 puntos."
        )

    print(
        f"[TRAJECTORY] {len(trajectory)} puntos cargados"
    )

    print(
        f"[ALIGN] rot={ALIGN_ROTATION_DEG}° "
        f"mirror_x={ALIGN_MIRROR_X} "
        f"mirror_y={ALIGN_MIRROR_Y} "
        f"offset=({ALIGN_OFFSET_X},{ALIGN_OFFSET_Y})"
    )

    return trajectory


def distance_xy(x1, y1, x2, y2):
    return math.hypot(
        x2 - x1,
        y2 - y1
    )


def get_vehicle_speed(vehicle):
    v = vehicle.get_velocity()

    return math.sqrt(
        v.x ** 2 +
        v.y ** 2 +
        v.z ** 2
    )


# ============================================================
# TRAJECTORY TRACKER
# ============================================================
class TrajectoryTracker:

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.n = len(trajectory)

        self.current_index = 0

        self.speed_integral = 0.0
        self.previous_speed_error = 0.0

    # --------------------------------------------------------
    # Find closest trajectory point
    # --------------------------------------------------------
    def find_nearest_index(self, x, y):

        best_index = self.current_index
        best_distance = float("inf")

        for offset in range(-5, 31):

            idx = (self.current_index + offset) % self.n

            p = self.trajectory[idx]

            d = distance_xy(
                x,
                y,
                p["x"],
                p["y"]
            )

            if d < best_distance:
                best_distance = d
                best_index = idx

        self.current_index = best_index

        return best_index, best_distance

    # --------------------------------------------------------
    # Get target point ahead of the kart
    # --------------------------------------------------------
    def get_target_index(self, nearest_index, lookahead):

        accumulated = 0.0
        idx = nearest_index

        while accumulated < lookahead:

            nxt = (idx + 1) % self.n

            p1 = self.trajectory[idx]
            p2 = self.trajectory[nxt]

            accumulated += distance_xy(
                p1["x"],
                p1["y"],
                p2["x"],
                p2["y"]
            )

            idx = nxt

            if idx == nearest_index:
                break

        return idx

    # --------------------------------------------------------
    # SIMPLE P STEERING CONTROLLER
    # --------------------------------------------------------
    def calculate_steering(
        self,
        vehicle,
        target_index
    ):

        transform = vehicle.get_transform()

        kart_x = transform.location.x
        kart_y = transform.location.y

        kart_yaw = math.radians(
            transform.rotation.yaw
        )

        target = self.trajectory[target_index]

        target_x = target["x"]
        target_y = target["y"]

        # ----------------------------------------------------
        # Vector from kart to target
        # ----------------------------------------------------
        vector_x = target_x - kart_x
        vector_y = target_y - kart_y

        # ----------------------------------------------------
        # Direction of the vector in world coordinates
        # ----------------------------------------------------
        target_angle = math.atan2(
            vector_y,
            vector_x
        )

        # ----------------------------------------------------
        # Angle between kart direction and target direction
        # ----------------------------------------------------
        angle_error = normalize_angle(
            target_angle - kart_yaw
        )

        # ----------------------------------------------------
        # Pure proportional controller
        # ----------------------------------------------------
        steer = STEERING_KP * angle_error

        # ----------------------------------------------------
        # Normalize from radians to CARLA steering [-1, 1]
        # ----------------------------------------------------
        steer = float(
            np.clip(
                steer / math.radians(45.0),
                -MAX_STEER,
                MAX_STEER
            )
        )

        return steer, angle_error

    # --------------------------------------------------------
    # Speed controller
    # --------------------------------------------------------
    def calculate_speed_control(
        self,
        vehicle,
        target_speed,
        dt
    ):

        current_speed = get_vehicle_speed(vehicle)

        error = target_speed - current_speed

        self.speed_integral = float(
            np.clip(
                self.speed_integral + error * dt,
                -20.0,
                20.0
            )
        )

        derivative = (
            (error - self.previous_speed_error) / dt
            if dt > 0
            else 0.0
        )

        self.previous_speed_error = error

        output = (
            SPEED_KP * error +
            SPEED_KI * self.speed_integral +
            SPEED_KD * derivative
        )

        if error >= 0:

            throttle = float(
                np.clip(
                    output,
                    0.0,
                    MAX_THROTTLE
                )
            )

            brake = 0.0

        else:

            throttle = 0.0

            brake = (
                float(
                    np.clip(
                        -output,
                        0.0,
                        MAX_BRAKE
                    )
                )
                if error < -SPEED_BRAKE_THRESHOLD
                else 0.0
            )

        return throttle, brake, current_speed


# ============================================================
# LOGGING
# ============================================================
def create_log(log_path):

    os.makedirs(
        log_path,
        exist_ok=True
    )

    fh = open(
        os.path.join(
            log_path,
            "trajectory_data.csv"
        ),
        "w",
        newline=""
    )

    writer = csv.writer(fh)

    writer.writerow([
        "sim_time",
        "x",
        "y",
        "yaw",
        "speed_mps",
        "target_speed_mps",
        "nearest_index",
        "nearest_distance",
        "angle_error",
        "steer",
        "throttle",
        "brake"
    ])

    return fh, writer


# ============================================================
# VISUALIZER
# ============================================================
class TrajectoryVisualizer:

    def __init__(
        self,
        world,
        vehicle,
        trajectory
    ):

        self.world = world
        self.vehicle = vehicle
        self.trajectory = trajectory

        self.top_camera = None
        self.kart_camera = None

        self.top_image = None
        self.kart_image = None

        self.running = True

        pygame.init()

        self.display = pygame.display.set_mode(
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )

        pygame.display.set_caption(
            "CARLA - Simple P Trajectory Controller"
        )

        self.font = pygame.font.SysFont(
            "Arial",
            17
        )

        xs = [
            p["x"]
            for p in trajectory
        ]

        ys = [
            p["y"]
            for p in trajectory
        ]

        self.center_x = (
            min(xs) + max(xs)
        ) / 2

        self.center_y = (
            min(ys) + max(ys)
        ) / 2

        self.create_cameras()

    # --------------------------------------------------------
    # Camera blueprint
    # --------------------------------------------------------
    def create_camera_bp(self):

        bp = self.world.get_blueprint_library().find(
            "sensor.camera.rgb"
        )

        bp.set_attribute(
            "image_size_x",
            str(DISPLAY_WIDTH // 2)
        )

        bp.set_attribute(
            "image_size_y",
            str(DISPLAY_HEIGHT)
        )

        return bp

    # --------------------------------------------------------
    # Cameras
    # --------------------------------------------------------
    def create_cameras(self):

        bp = self.create_camera_bp()

        self.top_camera = self.world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(
                    x=self.center_x,
                    y=self.center_y,
                    z=TOP_CAMERA_HEIGHT
                ),
                carla.Rotation(
                    pitch=-90.0
                )
            )
        )

        self.top_camera.listen(
            self._top_callback
        )

        bp2 = self.create_camera_bp()

        bp2.set_attribute(
            "fov",
            str(KART_CAMERA_FOV)
        )

        self.kart_camera = self.world.spawn_actor(
            bp2,
            carla.Transform(
                carla.Location(
                    x=0,
                    y=-0.65,
                    z=1.4
                )
            ),
            attach_to=self.vehicle
        )

        self.kart_camera.listen(
            self._kart_callback
        )

        print(
            "[CAMERA] Cenital + camara embarcada creadas."
        )

    # --------------------------------------------------------
    # Top camera callback
    # --------------------------------------------------------
    def _top_callback(self, image):

        a = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        ).reshape(
            image.height,
            image.width,
            4
        )

        self.top_image = (
            a[:, :, :3][:, :, ::-1]
        )

    # --------------------------------------------------------
    # Kart camera callback
    # --------------------------------------------------------
    def _kart_callback(self, image):

        a = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        ).reshape(
            image.height,
            image.width,
            4
        )

        self.kart_image = (
            a[:, :, :3][:, :, ::-1]
        )

    # --------------------------------------------------------
    # World -> top camera
    # --------------------------------------------------------
    def world_to_top(self, x, y):

        z = TOP_CAMERA_HEIGHT

        visible_width = (
            2 *
            z *
            math.tan(
                math.radians(CAMERA_FOV) / 2
            )
        )

        visible_height = (
            visible_width /
            ((DISPLAY_WIDTH / 2) / DISPLAY_HEIGHT)
        )

        return (
            int(
                (DISPLAY_WIDTH / 2) / 4 +
                (x - self.center_x) *
                ((DISPLAY_WIDTH / 2) / 2) /
                visible_width
            ),

            int(
                DISPLAY_HEIGHT / 2 -
                (y - self.center_y) *
                DISPLAY_HEIGHT /
                visible_height
            )
        )

    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------
    def draw_top_overlay(self, target):

        for p in self.trajectory:

            x, y = self.world_to_top(
                p["x"],
                p["y"]
            )

            pygame.draw.circle(
                self.display,
                (0, 255, 0),
                (x, y),
                2
            )

        # ----------------------------------------------------
        # Kart position
        # ----------------------------------------------------
        vt = self.vehicle.get_transform()

        x, y = self.world_to_top(
            vt.location.x,
            vt.location.y
        )

        pygame.draw.circle(
            self.display,
            (255, 0, 0),
            (x, y),
            7
        )

        # ----------------------------------------------------
        # Kart heading
        # ----------------------------------------------------
        yaw = math.radians(
            vt.rotation.yaw
        )

        end = (
            int(
                x +
                math.cos(yaw) * 18
            ),
            int(
                y -
                math.sin(yaw) * 18
            )
        )

        pygame.draw.line(
            self.display,
            (255, 255, 255),
            (x, y),
            end,
            3
        )

        # ----------------------------------------------------
        # Target point
        # ----------------------------------------------------
        tx, ty = self.world_to_top(
            target["x"],
            target["y"]
        )

        pygame.draw.circle(
            self.display,
            (255, 255, 0),
            (tx, ty),
            6
        )

        pygame.draw.line(
            self.display,
            (255, 255, 0),
            (x, y),
            (tx, ty),
            2
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------
    def text(self, text, x, y):

        self.display.blit(
            self.font.render(
                text,
                True,
                (255, 255, 255)
            ),
            (x, y)
        )

    # --------------------------------------------------------
    # Update display
    # --------------------------------------------------------
    def update(self, target, data):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.running = False

            elif (
                event.type == pygame.KEYDOWN and
                event.key == pygame.K_ESCAPE
            ):

                self.running = False

        self.display.fill(
            (0, 0, 0)
        )

        # ----------------------------------------------------
        # Top camera
        # ----------------------------------------------------
        if self.top_image is not None:

            img = pygame.surfarray.make_surface(
                self.top_image.swapaxes(0, 1)
            )

            self.display.blit(
                img,
                (0, 0)
            )

        self.draw_top_overlay(
            target
        )

        # ----------------------------------------------------
        # Kart camera
        # ----------------------------------------------------
        if self.kart_image is not None:

            img = pygame.surfarray.make_surface(
                self.kart_image.swapaxes(0, 1)
            )

            self.display.blit(
                img,
                (
                    DISPLAY_WIDTH // 2,
                    0
                )
            )

        pygame.draw.line(
            self.display,
            (255, 255, 255),
            (
                DISPLAY_WIDTH // 2,
                0
            ),
            (
                DISPLAY_WIDTH // 2,
                DISPLAY_HEIGHT
            ),
            2
        )

        self.text(
            "CENITAL",
            15,
            15
        )

        self.text(
            "CAMARA KART",
            DISPLAY_WIDTH // 2 + 15,
            15
        )

        self.text(
            f"idx={data['nearest_index']}  "
            f"target={data['target_index']}  "
            f"dist={data['nearest_distance']:.2f} m",
            15,
            DISPLAY_HEIGHT - 75
        )

        self.text(
            f"angle={math.degrees(data['angle_error']):+.1f}°  "
            f"steer={data['steer']:+.2f}",
            15,
            DISPLAY_HEIGHT - 50
        )

        self.text(
            f"v={data['speed']:.2f} m/s  "
            f"target_v={data['target_speed']:.2f} m/s",
            15,
            DISPLAY_HEIGHT - 25
        )

        pygame.display.flip()

    # --------------------------------------------------------
    # Destroy
    # --------------------------------------------------------
    def destroy(self):

        self.running = False

        for camera in (
            self.top_camera,
            self.kart_camera
        ):

            if camera is not None:

                try:
                    camera.stop()
                    camera.destroy()
                except Exception:
                    pass

        pygame.quit()

        print(
            "[CAMERA] Camaras cerradas."
        )


# ============================================================
# SPAWN KART
# ============================================================
def spawn_kart(world):

    bp = world.get_blueprint_library().find(
        VEHICLE_MODEL
    )

    spawn = random.choice(
        world.get_map().get_spawn_points()
    )

    vehicle = world.try_spawn_actor(
        bp,
        spawn
    )

    if vehicle is None:

        raise RuntimeError(
            "No se pudo crear el kart."
        )

    print(
        f"[VEHICLE] spawn={spawn}"
    )

    return vehicle


# ============================================================
# MAIN LOOP
# ============================================================
def game_loop(args):

    trajectory = load_trajectory(
        args.trajectory
    )

    client = carla.Client(
        "localhost",
        args.port
    )

    client.set_timeout(
        10.0
    )

    world = client.get_world()

    if args.town:

        print(
            f"[CARLA] Cargando mapa {args.town}"
        )

        world = client.load_world(
            args.town
        )

    log_path = os.path.join(
        args.log_path,
        str(int(time.time())) +
        "_" +
        args.town
    )

    os.makedirs(
        log_path,
        exist_ok=True
    )

    vehicle = None
    visualizer = None
    log_file = None

    try:

        # ----------------------------------------------------
        # Spawn
        # ----------------------------------------------------
        vehicle = spawn_kart(
            world
        )

        time.sleep(1.0)

        # ----------------------------------------------------
        # Tracker
        # ----------------------------------------------------
        tracker = TrajectoryTracker(
            trajectory
        )

        # ----------------------------------------------------
        # Visualizer
        # ----------------------------------------------------
        visualizer = TrajectoryVisualizer(
            world,
            vehicle,
            trajectory
        )

        # ----------------------------------------------------
        # Initial control
        # ----------------------------------------------------
        control = carla.VehicleControl()

        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        log_file, log_writer = create_log(
            log_path
        )

        previous_time = (
            world.get_snapshot()
            .timestamp
            .elapsed_seconds
        )

        print("\nCONTROL INICIADO")
        print(
            f"Trayectoria: {args.trajectory}"
        )
        print(
            f"Mapa: {args.town}"
        )
        print(
            f"Logs: {log_path}"
        )
        print(
            f"STEERING_KP: {STEERING_KP}"
        )
        print(
            "ESC para detener.\n"
        )

        # ====================================================
        # CONTROL LOOP
        # ====================================================
        while visualizer.running:

            snapshot = world.get_snapshot()

            current_time = (
                snapshot.timestamp
                .elapsed_seconds
            )

            dt = (
                min(
                    current_time -
                    previous_time,
                    0.1
                )
                if current_time > previous_time
                else 1.0 / CONTROL_HZ
            )

            previous_time = current_time

            # ------------------------------------------------
            # Kart position
            # ------------------------------------------------
            transform = vehicle.get_transform()

            x = transform.location.x
            y = transform.location.y

            # ------------------------------------------------
            # Nearest trajectory point
            # ------------------------------------------------
            (
                nearest_index,
                nearest_distance
            ) = tracker.find_nearest_index(
                x,
                y
            )

            # ------------------------------------------------
            # Lookahead
            # ------------------------------------------------
            speed = get_vehicle_speed(
                vehicle
            )

            lookahead = float(
                np.clip(
                    LOOKAHEAD_MIN +
                    LOOKAHEAD_SPEED_GAIN *
                    speed,
                    LOOKAHEAD_MIN,
                    LOOKAHEAD_MAX
                )
            )

            # ------------------------------------------------
            # Target point
            # ------------------------------------------------
            target_index = (
                tracker.get_target_index(
                    nearest_index,
                    lookahead
                )
            )

            target = trajectory[
                target_index
            ]

            # ------------------------------------------------
            # SIMPLE P STEERING
            # ------------------------------------------------
            (
                steer,
                angle_error
            ) = tracker.calculate_steering(
                vehicle,
                target_index
            )

            # ------------------------------------------------
            # Speed control
            # ------------------------------------------------
            target_speed = max(
                0.0,
                target["vx"]
            )

            (
                throttle,
                brake,
                speed
            ) = tracker.calculate_speed_control(
                vehicle,
                target_speed,
                dt
            )

            # ------------------------------------------------
            # Apply control
            # ------------------------------------------------
            control.steer = steer
            control.throttle = throttle
            control.brake = brake

            vehicle.apply_control(
                control
            )

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------
            yaw = math.radians(
                transform.rotation.yaw
            )

            log_writer.writerow([
                f"{current_time:.6f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{yaw:.6f}",
                f"{speed:.6f}",
                f"{target_speed:.6f}",
                nearest_index,
                f"{nearest_distance:.6f}",
                f"{angle_error:.6f}",
                f"{steer:.6f}",
                f"{throttle:.6f}",
                f"{brake:.6f}"
            ])

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------
            visualizer.update(
                target,
                {
                    "nearest_index": nearest_index,
                    "target_index": target_index,
                    "nearest_distance": nearest_distance,
                    "angle_error": angle_error,
                    "steer": steer,
                    "speed": speed,
                    "target_speed": target_speed
                }
            )

            # ------------------------------------------------
            # Console
            # ------------------------------------------------
            print(
                f"\ridx={nearest_index:4d} "
                f"target={target_index:4d} | "
                f"v={speed:5.2f}/"
                f"{target_speed:5.2f} | "
                f"angle="
                f"{math.degrees(angle_error):+6.1f}° | "
                f"steer={steer:+.3f}",
                end=""
            )

            time.sleep(
                1.0 / CONTROL_HZ
            )

    except KeyboardInterrupt:

        print(
            "\n\n[STOP] Interrumpido."
        )

    finally:

        # ----------------------------------------------------
        # Stop vehicle
        # ----------------------------------------------------
        if vehicle is not None:

            try:

                control = carla.VehicleControl()

                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0

                vehicle.apply_control(
                    control
                )

                time.sleep(0.2)

                vehicle.destroy()

                print(
                    "\n[VEHICLE] Kart destruido."
                )

            except Exception as e:

                print(
                    f"[VEHICLE] Error: {e}"
                )

        # ----------------------------------------------------
        # Close log
        # ----------------------------------------------------
        if log_file is not None:

            log_file.flush()
            log_file.close()

            print(
                f"[LOG] Guardado en {log_path}"
            )

        # ----------------------------------------------------
        # Destroy cameras
        # ----------------------------------------------------
        if visualizer is not None:

            visualizer.destroy()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Kart con trayectoria y "
            "controlador P simple"
        )
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        default=(
            "../TUMFTM/Output/"
            "Track_4_trayectory_opt.csv"
        )
    )

    parser.add_argument(
        "--town",
        "--carla-town",
        type=str,
        default="Track4"
    )

    parser.add_argument(
        "--port",
        "--carla-port",
        type=int,
        default=3010
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default=(
            os.getcwd() +
            "/logs/"
        )
    )

    args = parser.parse_args()

    game_loop(args)

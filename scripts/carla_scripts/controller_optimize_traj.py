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
# CORRECCION GEOMETRIA DEL KART
# ============================================================
# El origen (0,0) del actor no esta centrado en la anchura del kart:
# el cuerpo se genera hacia la izquierda del origen. 
KART_WIDTH_M = 0.64
KART_CENTER_OFFSET_M = KART_WIDTH_M / 2.0  # 0.32 m

# Desplazamiento lateral real usado por get_kart_center() (hacia la izquierda
# del vector "right"). Se usa tambien al hacer spawn para que el CENTRO del
# kart caiga justo sobre el punto de la trayectoria.
KART_ORIGIN_OFFSET_M = 0.70

# ============================================================
# SPAWN POINTS
# ============================================================
# Se eligen NUM_SPAWN_POINTS puntos repartidos por la vuelta (uno por tramo,
# con posicion aleatoria dentro del tramo) y se generan dos spawns por punto:
# sentido normal y sentido inverso -> 2 * NUM_SPAWN_POINTS spawns en total.
# La semilla es el nombre del mapa, asi que los puntos son siempre los mismos
# para un mismo circuito.
NUM_SPAWN_POINTS = 5
SPAWN_HEADING_LOOKAHEAD_PTS = 3   # puntos hacia delante para calcular la orientacion
SPAWN_Z_OFFSET = 0.3              # m sobre el suelo
SPAWN_FALLBACK_Z = 0.5            # m, si no se puede proyectar al suelo

# En sentido inverso se recorre la misma trayectoria al reves, pero el perfil
# de velocidad (vx) fue calculado para el sentido normal: las frenadas quedan
# donde no tocan. Se reduce para compensarlo.
REVERSE_SPEED_SCALE = 0.8

# ============================================================
# STEERING CONTROLLER
# ============================================================
STEERING_KP = 2.5
MAX_STEER = 1.0

# ============================================================
# ALINEACION TRAYECTORIA <-> CIRCUITO
# ============================================================
TRACK = "Track 4"

# (rotacion_deg, espejo_x, espejo_y, offset_x, offset_y)
TRACK_ALIGNMENTS = {
    "Track 1": (0.0, False, False, 0.0, 0.0),
    "Track 2": (0.0, False, False, 0.0, 0.0),
    "Track 3": (0.0, False, False, 0.0, 0.0),
    "Track 4": (0.0, False, False, 0.0, 0.0),
    "Track 5": (0.0, False, False, 0.0, 0.0),
    "Track 6": (0.0, False, False, 0.0, 0.0),
    "Track 7": (0.0, False, False, 0.0, 0.0),
    "Track 8": (0.0, False, False, 0.0, 0.0),
    "Track 9": (0.0, False, False, 0.0, 0.0),
}

if TRACK not in TRACK_ALIGNMENTS:
    raise ValueError(f"TRACK='{TRACK}' no definido.")

ALIGN_ROTATION_DEG, ALIGN_MIRROR_X, ALIGN_MIRROR_Y, ALIGN_OFFSET_X, ALIGN_OFFSET_Y = TRACK_ALIGNMENTS[TRACK]

# ============================================================
# LOOKAHEAD
# ============================================================
LOOKAHEAD_MIN = 1.0
LOOKAHEAD_SPEED_GAIN = 0.30
LOOKAHEAD_MAX = 4.0

# ============================================================
# SPEED CONTROLLER
# ============================================================
SPEED_KP = 0.35
SPEED_KI = 0.02
SPEED_KD = 0.02
MAX_THROTTLE = 1.0
MAX_BRAKE = 1.0
SPEED_BRAKE_THRESHOLD = 2.5

# ============================================================
# TOP CAMERA
# ============================================================
TOP_CAMERA_HEIGHT = 40.0
TOP_CAMERA_FOV = 90.0

# ============================================================
# KART CAMERA
# ============================================================
KART_CAMERA_FOV = 100.0

def debug_world_actors(world):
    actors = world.get_actors()
    print(f"\n[DEBUG] Total de actores detectados por CARLA: {len(actors)}")

    type_counts = {}
    for actor in actors:
        type_id = actor.type_id
        type_counts[type_id] = type_counts.get(type_id, 0) + 1

    print("\n[DEBUG] Tipos de actores presentes en el mundo:")
    for type_id, count in sorted(type_counts.items()):
        print(f"  {type_id}: {count}")

    print("\n[DEBUG] Actores cuyo tipo o atributos podrían corresponder a conos:")
    keywords = ("cone", "cono", "blue", "yellow", "orange", "azul", "amarillo", "naranja", "prop")

    matches = 0
    for actor in actors:
        attributes = getattr(actor, "attributes", {})
        description = f"{actor.type_id} {attributes}".lower()

        if any(keyword in description for keyword in keywords):
            location = actor.get_location()
            print(f"  ID={actor.id} | tipo={actor.type_id} | posición=({location.x:.2f}, {location.y:.2f}, {location.z:.2f}) | atributos={attributes}")
            matches += 1

    print(f"\n[DEBUG] Actores coincidentes: {matches}")

def get_kart_center(vehicle):
    """
    Centro real del kart en coordenadas de mundo, corrigiendo el
    desplazamiento lateral del origen del actor.

    Si tras probarlo el kart se va hacia el lado contrario al esperado,
    cambia el signo (usa -right.x, -right.y) en vez de invertir la constante.
    """
    transform = vehicle.get_transform()
    right = transform.get_right_vector()  # ya normalizado, en coords de mundo

    x = transform.location.x - right.x * KART_ORIGIN_OFFSET_M
    y = transform.location.y - right.y * KART_ORIGIN_OFFSET_M

    return x, y

# ============================================================
# UTILITIES
# ============================================================
def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def align_point(x, y, psi):
    theta = math.radians(ALIGN_ROTATION_DEG)
    c, s = math.cos(theta), math.sin(theta)
    x, y, psi = x * c - y * s, x * s + y * c, psi + theta

    if ALIGN_MIRROR_X:
        x, psi = -x, math.pi - psi

    if ALIGN_MIRROR_Y:
        y, psi = -y, -psi

    return x + ALIGN_OFFSET_X, y + ALIGN_OFFSET_Y, normalize_angle(psi)


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
                s, x, y, psi, kappa, vx, ax = map(float, parts[:7])
            except ValueError:
                continue

            #x, y, psi = align_point(x, y, psi)
            trajectory.append({"s": s, "x": x, "y": y, "psi": psi, "kappa": kappa, "vx": vx, "ax": ax})

    if len(trajectory) < 3:
        raise RuntimeError("La trayectoria contiene menos de 3 puntos.")

    print(f"[TRAJECTORY] {len(trajectory)} puntos cargados")
    print(f"[ALIGN] rot={ALIGN_ROTATION_DEG}° mirror_x={ALIGN_MIRROR_X} mirror_y={ALIGN_MIRROR_Y} offset=({ALIGN_OFFSET_X},{ALIGN_OFFSET_Y})")

    return trajectory


def distance_xy(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_vehicle_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


# ============================================================
# SPAWN POINTS (calculados a partir de la trayectoria)
# ============================================================
def compute_spawn_candidates(trajectory, num_points, seed):
    """
    Devuelve 2*num_points spawns sobre la trayectoria:
      - los primeros num_points en el sentido normal,
      - los siguientes num_points en las MISMAS posiciones pero girados 180°
        (sentido inverso). El spawn k y el k+num_points comparten posicion.

    La orientacion (yaw, en grados CARLA) sale de la tangente de la propia
    trayectoria, en el mismo sistema de coordenadas que usa el controlador.
    """
    n = len(trajectory)
    rng = random.Random(seed)
    segment = n / num_points
    margin = int(segment * 0.2)  # evita puntos pegados al borde de cada tramo

    indices = []
    for k in range(num_points):
        lo = int(k * segment) + margin
        hi = max(lo, int((k + 1) * segment) - 1 - margin)
        indices.append(rng.randint(lo, hi) % n)

    forward, reverse = [], []
    for idx in indices:
        p0 = trajectory[idx]
        p1 = trajectory[(idx + SPAWN_HEADING_LOOKAHEAD_PTS) % n]
        yaw = math.degrees(math.atan2(p1["y"] - p0["y"], p1["x"] - p0["x"]))
        yaw_rev = math.degrees(normalize_angle(math.radians(yaw + 180.0)))

        forward.append({"index": idx, "x": p0["x"], "y": p0["y"], "yaw": yaw, "reverse": False})
        reverse.append({"index": idx, "x": p0["x"], "y": p0["y"], "yaw": yaw_rev, "reverse": True})

    return forward + reverse


def reverse_trajectory(trajectory):
    """Misma trayectoria recorrida al reves (para los spawns en sentido inverso)."""
    return [
        dict(
            p,
            psi=normalize_angle(p["psi"] + math.pi),
            kappa=-p["kappa"],
            vx=p["vx"] * REVERSE_SPEED_SCALE,
        )
        for p in reversed(trajectory)
    ]


def get_ground_z(world, x, y):
    """Altura del suelo en (x, y) si la version de CARLA lo permite."""
    if hasattr(world, "ground_projection"):
        try:
            hit = world.ground_projection(carla.Location(x=x, y=y, z=50.0), 100.0)
            if hit is not None:
                return hit.location.z
        except Exception:
            pass
    return SPAWN_FALLBACK_Z - SPAWN_Z_OFFSET


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

    # ========================================================
    # START AT THE POINT CLOSEST TO THE KART (ANY PLACE IN THE LAP)
    # ========================================================
    def reset_to_nearest(self, x, y):
        self.current_index = min(
            range(self.n),
            key=lambda i: distance_xy(x, y, self.trajectory[i]["x"], self.trajectory[i]["y"])
        )

    # ========================================================
    # FIND NEAREST POINT IN FRONT OF THE KART
    # ========================================================
    def find_nearest_index(self, x, y, kart_yaw):
        best_index, best_distance = self.current_index, float("inf")

        for offset in range(-5, 200):
            idx = (self.current_index + offset) % self.n
            p = self.trajectory[idx]
            d = distance_xy(x, y, p["x"], p["y"])

            if d < best_distance:
                best_distance, best_index = d, idx

        self.current_index = best_index
        return best_index, best_distance

    def get_target_index(self, nearest_index, lookahead):
        accumulated, idx = 0.0, nearest_index

        while accumulated < lookahead:
            nxt = (idx + 1) % self.n
            p1, p2 = self.trajectory[idx], self.trajectory[nxt]
            accumulated += distance_xy(p1["x"], p1["y"], p2["x"], p2["y"])
            idx = nxt

            if idx == nearest_index:
                break

        return idx

    # ========================================================
    # SIMPLE P STEERING
    # ========================================================
    def calculate_steering(self, vehicle, target_index):
        kart_x, kart_y = get_kart_center(vehicle)
        kart_yaw = math.radians(vehicle.get_transform().rotation.yaw)

        target = self.trajectory[target_index]

        vector_x = target["x"] - kart_x
        vector_y = target["y"] - kart_y

        target_angle = math.atan2(vector_y, vector_x)
        angle_error = normalize_angle(target_angle - kart_yaw)

        steer = STEERING_KP * angle_error
        steer = float(np.clip(steer / math.radians(45.0), -MAX_STEER, MAX_STEER))

        return steer, angle_error

    # ========================================================
    # SPEED CONTROLLER
    # ========================================================
    def calculate_speed_control(self, vehicle, target_speed, dt):
        current_speed = get_vehicle_speed(vehicle)
        error = target_speed - current_speed

        self.speed_integral = float(np.clip(self.speed_integral + error * dt, -20.0, 20.0))

        derivative = (error - self.previous_speed_error) / dt if dt > 0 else 0.0
        self.previous_speed_error = error

        output = SPEED_KP * error + SPEED_KI * self.speed_integral + SPEED_KD * derivative

        if error >= 0:
            throttle = float(np.clip(output, 0.0, MAX_THROTTLE))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-output, 0.0, MAX_BRAKE)) if error < -SPEED_BRAKE_THRESHOLD else 0.0

        return throttle, brake, current_speed


# ============================================================
# SPEED CSV LOGGING
# ============================================================
def create_speed_log(log_path):
    csv_file_path = log_path + "/data.csv"
    csv_fh = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["sim_time", "speed_m_s"])
    return csv_fh, csv_writer


# ============================================================
# VISUALIZER
# ============================================================
class TrajectoryVisualizer:

    def __init__(self, world, vehicle, trajectory):
        self.world = world
        self.vehicle = vehicle
        self.trajectory = trajectory

        self.top_camera = None
        self.kart_camera = None
        self.top_image = None
        self.kart_image = None
        self.running = True

        pygame.init()
        self.display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("CARLA - Trajectory Debug / Kart Camera")
        self.font = pygame.font.SysFont("Arial", 17)

        self.top_width = DISPLAY_WIDTH // 2
        self.top_height = DISPLAY_HEIGHT

        xs = [p["x"] for p in trajectory]
        ys = [p["y"] for p in trajectory]

        self.center_x = (min(xs) + max(xs)) / 2
        self.center_y = (min(ys) + max(ys)) / 2

        self.create_cameras()

    # ========================================================
    # CAMERA BLUEPRINT
    # ========================================================
    def create_camera_bp(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.top_width))
        bp.set_attribute("image_size_y", str(self.top_height))
        return bp

    # ========================================================
    # CREATE CAMERAS
    # ========================================================
    def create_cameras(self):
        bp = self.create_camera_bp()
        bp.set_attribute("fov", str(TOP_CAMERA_FOV))

        self.top_camera = self.world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=self.center_x, y=self.center_y, z=TOP_CAMERA_HEIGHT),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            )
        )

        self.top_camera.listen(self._top_callback)

        bp2 = self.create_camera_bp()
        bp2.set_attribute("fov", str(KART_CAMERA_FOV))

        self.kart_camera = self.world.spawn_actor(
            bp2,
            carla.Transform(carla.Location(x=0.0, y=-0.65, z=1.4)),
            attach_to=self.vehicle
        )

        self.kart_camera.listen(self._kart_callback)

        print(f"[CAMERA] Cenital creada: {self.top_width}x{self.top_height}, FOV={TOP_CAMERA_FOV}°, Z={TOP_CAMERA_HEIGHT} m")
        print("[CAMERA] Camara embarcada creada.")

    # ========================================================
    # CAMERA CALLBACKS
    # ========================================================
    def _top_callback(self, image):
        a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        self.top_image = a[:, :, :3][:, :, ::-1]

    def _kart_callback(self, image):
        a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        self.kart_image = a[:, :, :3][:, :, ::-1]

    # ========================================================
    # WORLD -> TOP CAMERA PIXEL
    # ========================================================
    def world_to_top(self, x, y, z=0.0):
        camera_transform = self.top_camera.get_transform()
        point_camera = camera_transform.inverse_transform(carla.Location(x=x, y=y, z=z))

        depth = point_camera.x

        if depth <= 0.001:
            return None

        fx = self.top_width / (2.0 * math.tan(math.radians(TOP_CAMERA_FOV) / 2.0))
        fy = fx

        cx = self.top_width / 2.0
        cy = self.top_height / 2.0

        pixel_x = cx + fx * (point_camera.y / depth)
        pixel_y = cy - fy * (point_camera.z / depth)

        return int(round(pixel_x)), int(round(pixel_y))

    # ========================================================
    # WORLD -> KART CAMERA
    # ========================================================
    def world_to_kart(self, x, y, z=0.0):
        camera_transform = self.kart_camera.get_transform()
        point_camera = camera_transform.inverse_transform(carla.Location(x=x, y=y, z=z))

        depth = point_camera.x
        if depth <= 0.001:
            return None  # detrás de la cámara

        fx = self.top_width / (2.0 * math.tan(math.radians(KART_CAMERA_FOV) / 2.0))
        fy = fx

        cx = self.top_width / 2.0
        cy = self.top_height / 2.0

        pixel_x = cx + fx * (point_camera.y / depth)
        pixel_y = cy - fy * (point_camera.z / depth)

        return int(round(pixel_x)), int(round(pixel_y))

    def draw_kart_overlay(self, target):
        offset_x = DISPLAY_WIDTH // 2

        for p in self.trajectory:
            point = self.world_to_kart(p["x"], p["y"], 0.0)
            if point is not None:
                px, py = point
                if 0 <= px < self.top_width and 0 <= py < self.top_height:
                    pygame.draw.circle(self.display, (0, 255, 0), (px + offset_x, py), 3)

        target_point = self.world_to_kart(target["x"], target["y"], 0.0)
        if target_point is not None:
            tx, ty = target_point
            if 0 <= tx < self.top_width and 0 <= ty < self.top_height:
                pygame.draw.circle(self.display, (255, 255, 0), (tx + offset_x, ty), 7)

    # ========================================================
    # DRAW TOP OVERLAY
    # ========================================================
    def draw_top_overlay(self, target):
        for p in self.trajectory:
            point = self.world_to_top(p["x"], p["y"], 0.0)

            if point is not None:
                pygame.draw.circle(self.display, (0, 255, 0), point, 2)

        vt = self.vehicle.get_transform()
        x, y = get_kart_center(self.vehicle)
        kart_point = self.world_to_top(x, y, vt.location.z)

        if kart_point is not None:
            x, y = kart_point
            pygame.draw.circle(self.display, (255, 0, 0), (x, y), 7)

            yaw = math.radians(vt.rotation.yaw)
            heading_length = 18

            end = (
                int(x + math.cos(yaw) * heading_length),
                int(y - math.sin(yaw) * heading_length)
            )

            pygame.draw.line(self.display, (255, 255, 255), (x, y), end, 3)

            target_point = self.world_to_top(target["x"], target["y"], 0.0)

            if target_point is not None:
                tx, ty = target_point
                pygame.draw.circle(self.display, (255, 255, 0), (tx, ty), 6)
                pygame.draw.line(self.display, (255, 255, 0), (x, y), (tx, ty), 2)

    # ========================================================
    # TEXT
    # ========================================================
    def text(self, text, x, y):
        self.display.blit(self.font.render(text, True, (255, 255, 255)), (x, y))

    # ========================================================
    # UPDATE DISPLAY
    # ========================================================
    def update(self, target, data):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

        self.display.fill((0, 0, 0))

        # Draw trayectory in top image
        if self.top_image is not None:
            img = pygame.surfarray.make_surface(self.top_image.swapaxes(0, 1))
            self.display.blit(img, (0, 0))
        self.draw_top_overlay(target)

        # Draw trayectory in kart image
        if self.kart_image is not None:
            img = pygame.surfarray.make_surface(self.kart_image.swapaxes(0, 1))
            self.display.blit(img, (DISPLAY_WIDTH // 2, 0))
        self.draw_kart_overlay(target)

        # if self.kart_image is not None:
        #     img = pygame.surfarray.make_surface(self.kart_image.swapaxes(0, 1))
        #     self.display.blit(img, (DISPLAY_WIDTH // 2, 0))

        pygame.draw.line(self.display, (255, 255, 255), (DISPLAY_WIDTH // 2, 0), (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT), 2)

        self.text("CENITAL", 15, 15)
        self.text("CAMARA KART", DISPLAY_WIDTH // 2 + 15, 15)

        self.text(
            f"idx={data['nearest_index']}  target={data['target_index']}  dist={data['nearest_distance']:.2f} m",
            15, DISPLAY_HEIGHT - 75
        )

        self.text(
            f"angle={math.degrees(data['angle_error']):+.1f}°  steer={data['steer']:+.2f}",
            15, DISPLAY_HEIGHT - 50
        )

        self.text(
            f"v={data['speed']:.2f} m/s  target_v={data['target_speed']:.2f} m/s",
            15, DISPLAY_HEIGHT - 25
        )

        pygame.display.flip()

    # ========================================================
    # DESTROY
    # ========================================================
    def destroy(self):
        self.running = False

        for camera in (self.top_camera, self.kart_camera):
            if camera is not None:
                try:
                    camera.stop()
                    camera.destroy()
                except Exception:
                    pass

        pygame.quit()
        print("[CAMERA] Camaras cerradas.")


# ============================================================
# SPAWN KART
# ============================================================
def spawn_kart(world, spawn):
    """
    Crea el kart de modo que su CENTRO (no el origen del actor) quede sobre
    el punto `spawn` de la trayectoria, mirando en la direccion `spawn["yaw"]`.
    """
    bp = world.get_blueprint_library().find(VEHICLE_MODEL)

    yaw = spawn["yaw"]
    yaw_rad = math.radians(yaw)

    # vector "right" del kart para ese yaw; el origen del actor esta
    # desplazado KART_ORIGIN_OFFSET_M hacia ese lado respecto al centro.
    right_x, right_y = -math.sin(yaw_rad), math.cos(yaw_rad)
    origin_x = spawn["x"] + right_x * KART_ORIGIN_OFFSET_M
    origin_y = spawn["y"] + right_y * KART_ORIGIN_OFFSET_M
    origin_z = get_ground_z(world, origin_x, origin_y) + SPAWN_Z_OFFSET

    transform = carla.Transform(
        carla.Location(x=origin_x, y=origin_y, z=origin_z),
        carla.Rotation(yaw=yaw)
    )

    vehicle = world.try_spawn_actor(bp, transform)

    if vehicle is None:
        raise RuntimeError(
            f"No se pudo crear el kart en {transform}. Hay algo ocupando ese punto "
            f"(cono, otro actor) o esta fuera del suelo. Prueba con otro --spawn."
        )

    box = vehicle.bounding_box
    print(f"\nKart ID: {vehicle.id} | Blueprint: {vehicle.type_id}")
    print(f"Largo:  {2 * box.extent.x:.3f} m")
    print(f"Ancho:  {2 * box.extent.y:.3f} m")
    print(f"Alto:   {2 * box.extent.z:.3f} m")
    print(f"Escala del actor: {vehicle.get_transform()}")

    debug_world_actors(world)

    print(f"[VEHICLE] spawn={transform}")
    return vehicle


# ============================================================
# MAIN LOOP
# ============================================================
def game_loop(args):
    trajectory = load_trajectory(args.trajectory)

    client = carla.Client("localhost", args.port)
    client.set_timeout(10.0)

    world = client.get_world()

    if args.town:
        print(f"[CARLA] Cargando mapa {args.town}")
        world = client.load_world(args.town)

    # ========================================================
    # SPAWN POINTS (5 puntos del circuito + los mismos 5 al reves)
    # ========================================================
    candidates = compute_spawn_candidates(trajectory, NUM_SPAWN_POINTS, seed=args.town)

    print(f"\n[SPAWN] {len(candidates)} spawns disponibles ({NUM_SPAWN_POINTS} puntos x 2 sentidos):")
    for i, c in enumerate(candidates):
        way = "INVERSO" if c["reverse"] else "normal "
        print(f"  [{i}] {way} idx={c['index']:4d} x={c['x']:8.2f} y={c['y']:8.2f} yaw={c['yaw']:+7.1f}°")

    if args.spawn is None:
        spawn_id = random.randrange(len(candidates))
    else:
        if not 0 <= args.spawn < len(candidates):
            raise ValueError(f"--spawn debe estar entre 0 y {len(candidates) - 1}")
        spawn_id = args.spawn

    spawn = candidates[spawn_id]
    print(f"[SPAWN] Elegido: {spawn_id} ({'sentido inverso' if spawn['reverse'] else 'sentido normal'})")

    # En sentido inverso el kart sigue la trayectoria recorrida al reves
    if spawn["reverse"]:
        trajectory = reverse_trajectory(trajectory)

    # ========================================================
    # LOG DIRECTORY
    # ========================================================
    log_path = args.log_path + "/" + str(int(time.time())) + "_" + args.town + "/"
    os.makedirs(log_path, exist_ok=True)

    # ========================================================
    # SPEED CSV LOG
    # ========================================================
    csv_fh, csv_writer = create_speed_log(log_path)

    # ========================================================
    # CARLA RECORDER
    # ========================================================
    log_filename = log_path + args.town + ".log"
    print(log_filename)
    client.start_recorder(log_filename, True)

    vehicle = None
    visualizer = None

    try:
        vehicle = spawn_kart(world, spawn)
        time.sleep(1.0)

        tracker = TrajectoryTracker(trajectory)
        tracker.reset_to_nearest(*get_kart_center(vehicle))
        visualizer = TrajectoryVisualizer(world, vehicle, trajectory)

        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0

        # Start time at 0 for the speed CSV
        snapshot = world.get_snapshot()
        t0 = snapshot.timestamp.elapsed_seconds
        previous_time = t0

        print("\nCONTROL INICIADO")
        print(f"Trayectoria: {args.trajectory}")
        print(f"Mapa: {args.town}")
        print(f"Logs: {log_path}")
        print(f"STEERING_KP: {STEERING_KP}")
        print("ESC para detener.\n")

        while visualizer.running:
            snapshot = world.get_snapshot()
            current_time = snapshot.timestamp.elapsed_seconds
            rel_time = current_time - t0

            dt = min(current_time - previous_time, 0.1) if current_time > previous_time else 1.0 / CONTROL_HZ
            previous_time = current_time

            transform = vehicle.get_transform()
            x, y = get_kart_center(vehicle)
            yaw = transform.rotation.yaw

            nearest_index, nearest_distance = tracker.find_nearest_index(x, y, yaw)

            speed = get_vehicle_speed(vehicle)
            lookahead = float(np.clip(LOOKAHEAD_MIN + LOOKAHEAD_SPEED_GAIN * speed, LOOKAHEAD_MIN, LOOKAHEAD_MAX))

            target_index = tracker.get_target_index(nearest_index, lookahead)
            target = trajectory[target_index]

            # =================================================
            # SIMPLE P STEERING
            # =================================================
            steer, angle_error = tracker.calculate_steering(vehicle, target_index)

            # =================================================
            # SPEED CONTROL
            # =================================================
            target_speed = max(0.0, target["vx"])
            throttle, brake, speed = tracker.calculate_speed_control(vehicle, target_speed, dt)

            control.steer = steer
            control.throttle = throttle
            control.brake = brake

            vehicle.apply_control(control)

            # =================================================
            # SPEED CSV LOGGING
            # =================================================
            csv_writer.writerow([f"{rel_time:.6f}", f"{speed:.6f}"])

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

            print(
                f"\ridx={nearest_index:4d} target={target_index:4d} | "
                f"v={speed:5.2f}/{target_speed:5.2f} | "
                f"angle={math.degrees(angle_error):+6.1f}° | "
                f"steer={steer:+.3f}",
                end=""
            )

            time.sleep(1.0 / CONTROL_HZ)

    except KeyboardInterrupt:
        print("\n\n[STOP] Interrumpido.")

    finally:
        # ====================================================
        # STOP VEHICLE
        # ====================================================
        if vehicle is not None:
            try:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0

                vehicle.apply_control(control)
                time.sleep(0.2)
                vehicle.destroy()

                print("\n[VEHICLE] Kart destruido.")

            except Exception as e:
                print(f"[VEHICLE] Error: {e}")

        # ====================================================
        # STOP CARLA RECORDER
        # ====================================================
        try:
            client.stop_recorder()
            print(f"[RECORDER] Guardado en {log_filename}")
        except Exception as e:
            print(f"[RECORDER] Error al detener recorder: {e}")

        # ====================================================
        # CLOSE SPEED CSV
        # ====================================================
        csv_fh.flush()
        csv_fh.close()
        print(f"[LOG] Velocidad guardada en {log_path}/data.csv")

        # ====================================================
        # DESTROY CAMERAS
        # ====================================================
        if visualizer is not None:
            visualizer.destroy()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kart con trayectoria y controlador P simple")

    parser.add_argument(
        "--trajectory",
        type=str,
        default="../TUMFTM/Output/Track9/Track_9_trayectory_opt.csv"
    )

    parser.add_argument(
        "--town",
        "--carla-town",
        type=str,
        default="Track9"
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
        default=os.getcwd() + "/logs/"
    )

    parser.add_argument(
        "--spawn",
        type=int,
        default=None,
        help="Indice de spawn (0-4 sentido normal, 5-9 los mismos puntos en sentido inverso). "
             "Si se omite, se elige uno al azar."
    )

    args = parser.parse_args()
    game_loop(args)
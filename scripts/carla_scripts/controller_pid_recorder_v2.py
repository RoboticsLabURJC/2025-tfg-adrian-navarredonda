"""
Este script ejecuta una simulación en CARLA donde un vehículo autónomo es capaz
de seguir un carril delimitado por conos utilizando visión artificial.

Se emplea un modelo de YOLO para identificar conos en la imagen de una cámara RGB montada en el vehículo. 
A partir de las detecciones, se calcula una línea central del carril y se aplica un controlador
PID para ajustar la dirección del vehículo en tiempo real.

Además:
- Se registran datos de velocidad en un archivo CSV
- Se graba la simulación en un archivo .log de CARLA
- Se visualiza la imagen procesada con anotaciones

-------------------------------------------------------------
Uso:
----
Ejecutar desde terminal:

    python controller_pid_recorder_v2.py [opciones]

Opciones disponibles:
---------------------
--log_path       Ruta donde se guardarán los logs (por defecto ./logs/)
--town           Mapa de CARLA a cargar (por defecto "Track3")
--port           Puerto del servidor CARLA (por defecto 3010)
--tport          Puerto del Traffic Manager (por defecto 3020)

Ejemplo:

    python controller_pid_recorder_v2.py --town Track3 --port 3010
"""

import carla
import time
import pygame
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.interpolate import splprep, splev
import csv
import os
import argparse
import random
import sys

# ================= MODEL =================
MODEL_PATH = "../../yolo_model/runs/detect/Run_con_parametros/weights/best.pt"
device = "cuda:1"
model = YOLO(MODEL_PATH).to()

# ================= CONFIG =================
WIDTH, HEIGHT = 1000, 800
RATE_CONTROL_LOOP = 30
VEHICLE_MODEL = "vehicle.kart.kart"

CSV_FILENAME = "data.csv"

CLASSES = [
    "blue_cone",
    "large_orange_cone",
    "orange_cone",
    "unknown_cone",
    "yellow_cone"
]

LANE_WIDTH_PX = 600

# Punto fijo en el centro inferior de la imagen (posición del kart).
# Siempre se añade a la centerline para que el spline esté anclado
# en un punto cercano fiable independientemente de lo que detecte YOLO.
ANCHOR_POINT = (int(WIDTH // 2), int(0.8 * HEIGHT))

# ================= STATE =================
control = carla.VehicleControl()
control.throttle = 0.5

camera_image = None
frame_id = 0

prev_centerline = None
last_time = time.time()
lost_frames = 0
MAX_LOST_FRAMES = 100

SCAN_LINES_Y = [
    int(HEIGHT * 0.875),
    int(HEIGHT * 0.85),
    int(HEIGHT * 0.825),
    int(HEIGHT * 0.80),
    int(HEIGHT * 0.775),
    int(HEIGHT * 0.75),
    int(HEIGHT * 0.725),
    int(HEIGHT * 0.70),
    int(HEIGHT * 0.675),
    int(HEIGHT * 0.65),
    int(HEIGHT * 0.625),
    int(HEIGHT * 0.60),
    int(HEIGHT * 0.575),
    int(HEIGHT * 0.55),
    int(HEIGHT * 0.525),
    int(HEIGHT * 0.50),
]

# PID constants
kp = 0.0045
ki = 0.00007
kd = 0.0005
prev_error = 0.0
integral = 0.0
prev_target_x = 0


# ================= HELPERS =================

# Filtros de seguridad para process_detections:
# - MIN_BOX_AREA: descarta conos con bbox muy pequeño (lejanos o falsos positivos)
# - BORDER_MARGIN: descarta conos cuyo centro esté cerca del borde de la imagen
#   (los conos en los extremos suelen ser parcialmente visibles y poco fiables)
MIN_BOX_AREA  = 200  # px² — bbox más pequeño que esto se ignora
BORDER_MARGIN = 20   # px — margen desde cada borde (izq, der, arriba)

def process_detections(results):
    left, right = [], []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.7:
            continue

        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Filtro 1: tamaño mínimo del bounding box
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Filtro 2: descartar conos en los bordes de la imagen
        # (izquierdo, derecho y superior — el inferior es donde está el kart)
        if cx < BORDER_MARGIN or cx > WIDTH - BORDER_MARGIN:
            continue
        if cy < BORDER_MARGIN:
            continue

        name = CLASSES[cls]

        if name == "blue_cone":
            left.append((cx, cy))
        elif name == "yellow_cone":
            right.append((cx, cy))

    # Ordenar de más cercano (mayor Y) a más lejano (menor Y)
    # y quedarse solo con los 3 más cercanos por lado.
    left  = sorted(left,  key=lambda p: p[1], reverse=True)[:3]
    right = sorted(right, key=lambda p: p[1], reverse=True)[:3]

    return left, right

def estimate_missing_side(known_points, side, lane_width):
    """
    Si falta un lado completo, estima su posición desplazando LANE_WIDTH_PX.
    side='left'  desplazar a la derecha; side='right'  a la izquierda.
    """
    points = known_points[1:] if len(known_points) > 1 else known_points
    dx = lane_width if side == 'left' else -lane_width
    estimated = [(x + dx, y) for (x, y) in points]
    return estimated

def find_central_spline(left, right):
    """
    Empareja conos azules (left) con amarillos (right) por proximidad en Y.
    Genera puntos centrales para pares con diferencia en Y < MAX_Y_DIFF.
    Si no hay ningún par válido, devuelve None para que el ancla tome el control.
    """
    # Subido de 30 a 60 px: con 30 los conos en curva raramente forman
    # pares válidos y find_central_spline devuelve None constantemente,
    # lo que hace que el kart dependa solo del ancla sin información real.
    MAX_Y_DIFF = 30

    centerline = []
    used_right = set()

    for lx, ly in left:
        best_idx = None
        best_dy = float('inf')

        for i, (rx, ry) in enumerate(right):
            if i in used_right:
                continue
            dy = abs(ly - ry)
            if dy < best_dy:
                best_dy = dy
                best_idx = i

        if best_idx is not None and best_dy < MAX_Y_DIFF:
            rx, ry = right[best_idx]
            cx = int((lx + rx) / 2)
            cy = int((ly + ry) / 2)
            centerline.append((cx, cy))
            used_right.add(best_idx)

    pts = sorted(centerline, key=lambda p: p[1], reverse=True)

    return pts if len(pts) >= 1 else None

def draw_yb_lines(annotated, left, right):
    left_sorted = sorted(left, key=lambda p: p[1])
    right_sorted = sorted(right, key=lambda p: p[1])

    for i in range(len(left_sorted) - 1):
        cv2.line(annotated, left_sorted[i], left_sorted[i + 1], (255, 0, 0), 2)

    for i in range(len(right_sorted) - 1):
        cv2.line(annotated, right_sorted[i], right_sorted[i + 1], (0, 255, 255), 2)

def draw_points(image, points, radius=5):
    if points:
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), radius, (0, 0, 255), -1)
    else:
        print("No hay puntos para pintar")
        return

def smooth_centerline(centerline, num_points=20):
    if len(centerline) < 3:
        return centerline

    # Ordenar por Y descendente: splprep interpola en el orden de entrada,
    # si los puntos no son monótonos en Y el spline hace tirabuzones.
    pts = sorted(centerline, key=lambda p: p[1], reverse=True)

    # Eliminar puntos con Y casi idéntica (<5 px): generan nudos en splprep.
    filtered = [pts[0]]
    for p in pts[1:]:
        if abs(p[1] - filtered[-1][1]) >= 5:
            filtered.append(p)

    if len(filtered) < 3:
        return centerline

    xs = [p[0] for p in filtered]
    ys = [p[1] for p in filtered]

    try:
        tck, _ = splprep([xs, ys], s=50, k=min(3, len(filtered)-1))
        u_new = np.linspace(0, 1, num_points)
        xs_smooth, ys_smooth = splev(u_new, tck)
        return [(int(x), int(y)) for x, y in zip(xs_smooth, ys_smooth)]
    except Exception:
        return centerline

def get_pid_target(centerline):
    """
    Para cada línea de control de la 4ª a la 8ª (índices 3 a 7 de
    SCAN_LINES_Y ordenadas de cercana a lejana), busca el punto del
    spline más próximo. Con los X recogidos calcula la mediana y
    devuelve un punto objetivo sintético en esa X y la Y media.

    La mediana suaviza outliers puntuales del spline (un frame con un
    cono mal detectado no desplaza el objetivo bruscamente).
    """
    if not centerline:
        return None

    SCAN_TOLERANCE = 40  # px de margen alrededor de cada línea de control

    # SCAN_LINES_Y ordenadas de mayor Y (cercano) a menor Y (lejano)
    scan_lines_sorted = sorted(SCAN_LINES_Y, reverse=True)

    # Líneas 4ª a 8ª (índices 3 a 7 inclusive)
    target_lines = scan_lines_sorted[3:8]

    xs, ys = [], []
    for scan_y in target_lines:
        candidates = [p for p in centerline if abs(p[1] - scan_y) < SCAN_TOLERANCE]
        if candidates:
            best = min(candidates, key=lambda p: abs(p[1] - scan_y))
            xs.append(best[0])
            ys.append(best[1])

    if not xs:
        # Fallback: punto más lejano disponible del spline
        return min(centerline, key=lambda p: p[1])

    median_x = int(np.median(xs))
    median_y = int(np.median(ys))
    return (median_x, median_y)


def game_loop(args):

    global camera_image, frame_id
    global prev_centerline, lost_frames

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    client = carla.Client("localhost", args.port)
    client.set_timeout(10.0)

    world = client.load_world(args.town)

    log_path = args.log_path + "/" + str(int(time.time())) + "_" + args.town + "/"
    os.makedirs(log_path, exist_ok=True)

    # Speed CSV settings    
    csv_file_path = log_path + CSV_FILENAME
    csv_fh = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["sim_time", "speed_m_s"])

    # Vehicle
    bp = world.get_blueprint_library().find(VEHICLE_MODEL)
    spawn = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn)

    # Logging
    log_filename = log_path + args.town + ".log"
    print(log_filename)
    client.start_recorder(log_filename, True)

    # Camera
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WIDTH))
    cam_bp.set_attribute("image_size_y", str(HEIGHT))

    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=0, y=-0.65, z=1.4)),
        attach_to=vehicle
    )

    # Callback for processing images
    def process_image(image):

        global camera_image, frame_id
        global prev_centerline, lost_frames, last_time
        global kp, ki, kd, prev_error, integral, prev_target_x

        frame_id += 1
        now = time.time()
        dt = now - last_time
        last_time = now

        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        results = model(img)

        # Create left and right lines
        left, right = process_detections(results)

        # Draw bounding box and lines
        annotated = results[0].plot(labels=False, conf=False)

        draw_yb_lines(annotated, left, right)

        # Draw the control lines
        for y in SCAN_LINES_Y:
            cv2.line(annotated, (0, y), (WIDTH, y), (255, 255, 255), 1)

        # Estimate missing side (primer intento: lado completamente vacío)
        if not left and right:
            print("LADO IZQUIERDO INFERIDO")
            left = estimate_missing_side(right, 'right', LANE_WIDTH_PX)
        elif not right and left:
            print("LADO DERECHO INFERIDO")
            right = estimate_missing_side(left, 'left', LANE_WIDTH_PX)

        # Create centerline
        centerline = find_central_spline(left, right)

        # Segundo intento: si find_central_spline no generó ningún par
        # (los conos existen pero no coinciden en Y), inferir el lado
        # con menos conos a partir del lado dominante y recalcular.
        # Esto cubre el caso de 3 azules + 1 amarillo lejano donde
        # no se forma ningún par válido pero hay información en un lado.
        if centerline is None:
            if len(left) >= len(right):
                right = estimate_missing_side(left, 'left', LANE_WIDTH_PX)
            else:
                left = estimate_missing_side(right, 'right', LANE_WIDTH_PX)
            centerline = find_central_spline(left, right)

        draw_points(annotated, left)
        draw_points(annotated, right)

        # Añadir siempre el ancla, haya o no centerline detectada.
        # El else descomentado garantiza que aunque find_central_spline
        # devuelva None siempre tengamos al menos el ancla para el spline.
        if centerline is not None:
            centerline = centerline + [ANCHOR_POINT]
        else:
            centerline = [ANCHOR_POINT]

        # Dibujar el punto ancla en magenta para distinguirlo del resto
        cv2.circle(annotated, ANCHOR_POINT, 8, (255, 0, 255), -1)

        if centerline is not None and len(centerline) >= 3:
            centerline = smooth_centerline(centerline)
            # Garantizar int nativos para cv2 (splev devuelve numpy scalars)
            centerline = [(int(x), int(y)) for x, y in centerline]
            prev_centerline = centerline
            lost_frames = 0

            for i in range(len(centerline) - 1):
                cv2.line(
                    annotated,
                    centerline[i],
                    centerline[i + 1],
                    (0, 255, 0),  # verde BGR
                    2
                )
        else:
            lost_frames += 1
            centerline = prev_centerline

        draw_points(annotated, centerline)

        # Objetivo PID
        if centerline:
            target = get_pid_target(centerline)
            if target:
                tx, ty = target
                error = tx - WIDTH / 2
                integral += error * dt
                derivative = (error - prev_error) / dt if dt > 0 else 0.0
                prev_error = error

                steer = kp * error + ki * integral + kd * derivative
                control.steer = float(np.clip(steer, -1.0, 1.0))

                cv2.circle(annotated, (tx, ty), 6, (0, 255, 255), -1)

        # ================= STOP CONDITION =================
        if lost_frames >= MAX_LOST_FRAMES:
            control.throttle = 0.0
            control.brake = 1.0
            vehicle.apply_control(control)
            print("[STOP] Lane lost")
            return

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        camera_image = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        vehicle.apply_control(control)


    camera.listen(process_image)

    clock = pygame.time.Clock()

    # Measure the execution rate (Hz) in the server
    def on_tick(snapshot):
        fps_server = 1.0 / snapshot.timestamp.delta_seconds
        print(f"Frame {snapshot.frame} | Server ~{fps_server:.1f} Hz  ", end="\r")

    callback_id = world.on_tick(on_tick)

    # Start time at 0 for the speed csv
    snapshot = world.get_snapshot()               
    t0 = snapshot.timestamp.elapsed_seconds

    # ================= LOOP =================
    try:
        while True:
            rel_time = world.get_snapshot().timestamp.elapsed_seconds - t0
            # Get vehicle speed
            raw_vel = vehicle.get_velocity()

            # rate of this control loop
            clock.tick(RATE_CONTROL_LOOP)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            if camera_image:
                screen.blit(camera_image, (0, 0))
            pygame.display.flip()

            speed = float(np.linalg.norm([raw_vel.x, raw_vel.y, raw_vel.z]))
            csv_writer.writerow([f"{rel_time:.6f}", f"{speed:.6f}"])
            print(f"Speed {speed:.2f} m/s | Steer {control.steer:.2f}")
    except KeyboardInterrupt:
        print("Exit...")

    finally:      
        world.remove_on_tick(callback_id)

        client.stop_recorder()
        
        if camera is not None:
            camera.stop()
            camera.destroy()
            
        if vehicle is not None:
            vehicle.destroy()

        csv_fh.flush()
        csv_fh.close()

        pygame.quit()
        sys.exit()

# ================= MAIN =================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="recorder")

    parser.add_argument("--log_path", type=str, default=os.getcwd() + "/logs/",
                        help="Directory where log files will be stored")
    
    parser.add_argument("--town", "--carla-town", type=str, default="Track3",
                        help="Name of the CARLA map to load")
    
    parser.add_argument("--port", "--carla-port", type=int, default=3010,
                        help="Port used to connect to the CARLA simulator")
    
    parser.add_argument("--tport","--carla-traffic-port", type=int, default=3020,
                        help="Port used by the CARLA traffic manager")

    args = parser.parse_args()

    try:
        game_loop(args)
    except SystemExit:
        pass
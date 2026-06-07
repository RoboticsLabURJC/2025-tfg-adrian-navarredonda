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
model = YOLO(MODEL_PATH).to(device)

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

LANE_WIDTH_PX = 400

# ================= STATE =================
control = carla.VehicleControl()
control.throttle = 0.5

camera_image = None
frame_id = 0


prev_centerline = None
last_time = time.time()
lost_frames = 0
MAX_LOST_FRAMES = 10

SCAN_LINES_Y = [
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
kp = 0.002
ki = 0.00007
kd = 0.0005
prev_error = 0.0
integral = 0.0
prev_target_x = 0


# ================= HELPERS =================

def process_detections(results):
    left, right = [], []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.7:
            continue

        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        name = CLASSES[cls]

        if name == "blue_cone":
            left.append((cx, cy))
        elif name == "yellow_cone":
            right.append((cx, cy))

    # Ordenar puntos de mas cercano a mas lejano, (cada punto es un cono)
    left = sorted(left, key=lambda p: p[1], reverse=True)
    right = sorted(right, key=lambda p: p[1], reverse=True)

    return left, right

def estimate_missing_side(known_points, side, lane_width):
    """
    Si falta un lado completo, estima su posición desplazando LANE_WIDTH_PX.
    side='left'  desplazar a la derecha; side='right'  a la izquierda.
    """
    estimated = []
    dx = lane_width if side == 'left' else -lane_width
    for (x, y) in known_points:
        estimated.append((x + dx, y))
    return estimated

def find_central_spline(left, right):
    """
    Empareja conos azules (left) con amarillos (right) por proximidad en Y.
    Genera puntos centrales solo para pares con Y < MAX_Y_DIFF.
    """
    MAX_Y_DIFF = 30 
    
    centerline = []
    used_right = set()

    for lx, ly in left:
        # Buscar el cono amarillo más cercano en Y que no esté ya emparejado
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

    if len(centerline) <= 2:
        return centerline

    pts = sorted(centerline, key=lambda p: p[1], reverse=True)

    return pts if len(pts) >= 3 else None

def draw_yb_lines(annotated, left, right):
    left_sorted = sorted(left, key=lambda p: p[1])
    right_sorted = sorted(right, key=lambda p: p[1])

    # línea azul (izquierda)
    for i in range(len(left_sorted) - 1):
        cv2.line(
            annotated,
            left_sorted[i],
            left_sorted[i + 1],
            (255, 0, 0),  # azul BGR
            2
        )

    # línea amarilla (derecha)
    for i in range(len(right_sorted) - 1):
        cv2.line(
            annotated,
            right_sorted[i],
            right_sorted[i + 1],
            (0, 255, 255),  # amarillo BGR
            2
        )

def draw_points(image, points, radius=5):
    """
    Dibuja una lista de puntos sobre una imagen.

    Args:
        image: imagen OpenCV (BGR).
        points: lista de tuplas [(x, y), ...].
        radius: radio del punto.
    """
    if points:
        for x, y in points:
            cv2.circle(
                image,
                (int(x), int(y)),
                radius,
                (0, 0, 255),  # rojo en BGR
                -1            # relleno
            )
    else:
        print("No hay puntos para pintar")
        return

def smooth_centerline(centerline, num_points=20):
    if len(centerline) < 3:
        return centerline
    
    xs = [p[0] for p in centerline]
    ys = [p[1] for p in centerline]
    
    try:
        tck, _ = splprep([xs, ys], s=50, k=min(3, len(centerline)-1))
        u_new = np.linspace(0, 1, num_points)
        xs_smooth, ys_smooth = splev(u_new, tck)
        return list(zip(xs_smooth.astype(int), ys_smooth.astype(int)))
    except Exception:
        return centerline
    
LOOKAHEAD_Y = int(HEIGHT * 0.60)  # 60% desde arriba = ~adelante del kart

def get_pid_target(centerline):
    """
    Devuelve el punto de la línea central más cercano al lookahead_y.
    Más estable que el punto más cercano al vehículo.
    """
    if not centerline:
        return None
    
    target = min(centerline, key=lambda p: abs(p[1] - LOOKAHEAD_Y))
    return target


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

        # Estimate missing side
        if not left and right:
            left = estimate_missing_side(right, 'right', LANE_WIDTH_PX)
        elif not right and left:
            right = estimate_missing_side(left, 'left', LANE_WIDTH_PX)

        # Create centerline
        centerline = find_central_spline(left, right)

        draw_points(annotated, left)
        draw_points(annotated, right)
        draw_points(annotated, centerline)

        if centerline is not None:
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

        if centerline and len(centerline) >= 3:
            centerline = smooth_centerline(centerline)
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

                cv2.circle(annotated, (tx, ty), 6, (0, 0, 255), -1)

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
    
    # parser.add_argument("--extra_actor", "--carla-extra-actor", action="store_true",
    #                     help="Spawn an additional actor in front of the ego-vehicle")

    args = parser.parse_args()

    try:
        game_loop(args)
    except SystemExit:
        pass
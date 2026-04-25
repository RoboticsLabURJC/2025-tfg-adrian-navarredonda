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

    python script.py [opciones]

Opciones disponibles:
---------------------
--log_path       Ruta donde se guardarán los logs (por defecto ./logs/)
--town           Mapa de CARLA a cargar (por defecto "Track3")
--port           Puerto del servidor CARLA (por defecto 3010)
--tport          Puerto del Traffic Manager (por defecto 3020)

Ejemplo:

    python script.py --town Track3 --port 3010
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

MODEL_PATH = "../../models/runs/detect/Run_con_parametros/weights/best.pt"
model = YOLO(MODEL_PATH)
CSV_FILENAME = "data.csv"
RATE_CONTROL_LOOP = 30
VEHICLE_MODEL = "vehicle.kart.kart"
CLASSES = ["blue_cone", "large_orange_cone", "orange_cone", "unknown_cone", "yellow_cone"]
# ================= CARLA =================
WIDTH, HEIGHT = 1000, 800
LANE_WIDTH_PX = 180


kp, ki, kd = 0.004, 0.0001, 0.001
prev_error = 0.0
integral = 0.0

control = carla.VehicleControl()
control.throttle = 0.5
control.brake = 0.0

camera_image = None
frame_id = 0
last_time = time.time()

def game_loop(args):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA - YOLO + Robust Lane Following")

    client = carla.Client('localhost', args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town)

    world.set_weather(carla.WeatherParameters(
        cloudiness=80.0,
        sun_altitude_angle=90.0
    ))

    log_path = args.log_path + "/" + str(int(time.time())) + "_" + args.town + "/"
    os.makedirs(log_path, exist_ok=True)

    # Speed CSV settings    
    csv_file_path = log_path + CSV_FILENAME
    csv_fh = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["sim_time", "speed_m_s"])

    bp_lib = world.get_blueprint_library()

    #select vehicle
    vehicle_bp = bp_lib.find(VEHICLE_MODEL)

    #Escoger un punto de spawn aleatoriamente
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) 
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    if not vehicle:
        raise RuntimeError("Vehicle not spawned")
    
    # Traffic manager
    try:
        tm = client.get_trafficmanager(args.tport)
    except RuntimeError:
        args.tport += 2
        tm = client.get_trafficmanager(args.tport)

    print("TM port:", tm.get_port())
    tm = client.get_trafficmanager(args.tport)
    tm_port = tm.get_port()  
    
    # Start the log and recording
    # Important! Make sure you start the recording after spawn all your actors
    log_filename = log_path + args.town + ".log"
    print(log_filename)
    client.start_recorder(log_filename, True)

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(WIDTH))
    camera_bp.set_attribute("image_size_y", str(HEIGHT))
    camera_bp.set_attribute("fov", "90")

    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=0, y=-0.65, z=1)),
        attach_to=vehicle
    )

    # ================= PID =================

    def process_image(image):
        global camera_image, frame_id
        global prev_error, integral, last_time, control

        frame_id += 1
        now = time.time()
        dt = now - last_time
        last_time = now

        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        start = time.time()
        results = model(img)
        infer_ms = (time.time() - start) * 1000.0
        #writer_time.writerow([frame_id, infer_ms])

        left_cones = []
        right_cones = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            #writer.writerow([frame_id, cls, conf])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cls_name = CLASSES[cls]

            if cls_name == "blue_cone":
                left_cones.append((cx, cy))
            elif cls_name == "yellow_cone":
                right_cones.append((cx, cy))

        annotated = results[0].plot(labels=False, conf=False)

        # ===== SOLO LOS 4 CONOS MÁS CERCANOS =====
        left_cones = sorted(left_cones, key=lambda p: p[1], reverse=True)[:4]
        right_cones = sorted(right_cones, key=lambda p: p[1], reverse=True)[:4]

        left_cones.sort(key=lambda p: p[1])
        right_cones.sort(key=lambda p: p[1])

        # ===== POLILÍNEAS LATERALES =====
        for i in range(len(left_cones) - 1):
            cv2.line(annotated, left_cones[i], left_cones[i + 1], (255, 0, 0), 2)

        for i in range(len(right_cones) - 1):
            cv2.line(annotated, right_cones[i], right_cones[i + 1], (0, 255, 255), 2)

        # ===== CONSTRUCCIÓN ROBUSTA DEL CENTRO =====
        centerline = []
        have_left = len(left_cones) >= 2
        have_right = len(right_cones) >= 2

        # Ambos lados visibles → promedio normal
        if have_left and have_right:
            for l, r in zip(left_cones, right_cones):
                centerline.append((
                    int((l[0] + r[0]) / 2),
                    int((l[1] + r[1]) / 2)
                ))

        elif not have_left and have_right:
            for x, y in left_cones:
                cx = int(x + LANE_WIDTH_PX)
                cy = int(y)
                centerline.append((cx, cy))

        elif not have_right and have_left:
            for x, y in right_cones:
                cx = int(x - LANE_WIDTH_PX)
                cy = int(y)
                centerline.append((cx, cy))

        # ===== SPLINE CENTRAL + CONTROL =====
        if len(centerline) >= 3:
            pts = np.array(centerline)
            k = min(2, len(centerline) - 1)
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=5, k=k)
            u = np.linspace(0, 1, 50)
            xs, ys = splev(u, tck)

            spline = list(zip(xs.astype(int), ys.astype(int)))

            for i in range(len(spline) - 1):
                cv2.line(annotated, spline[i], spline[i + 1], (0, 255, 0), 2)

            TARGET_Y = int(HEIGHT * 0.6)
            tx, ty = spline[-1]
            for x, y in spline:
                if y > TARGET_Y:
                    tx, ty = x, y
                    break

            error = tx - WIDTH / 2
            integral += error * dt
            derivative = (error - prev_error) / dt if dt > 0 else 0.0
            prev_error = error

            steer = kp * error + ki * integral + kd * derivative
            control.steer = float(np.clip(steer, -1.0, 1.0))

            cv2.circle(annotated, (tx, ty), 6, (0, 0, 255), -1)

        vehicle.apply_control(control)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        camera_image = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

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

if __name__ == '__main__':

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

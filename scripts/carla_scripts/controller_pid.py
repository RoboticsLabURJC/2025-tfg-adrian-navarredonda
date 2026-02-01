import carla
import time
import pygame
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.interpolate import splprep, splev
import csv
import os

# ================= YOLO =================
MODEL_PATH = "../../models/runs/detect/Run_con_parametros/weights/best.pt"
model = YOLO(MODEL_PATH)

CLASSES = ["blue_cone", "large_orange_cone", 'orange_cone', 'unknown_cone', 'yellow_cone']

# ================= CSV =================
csv_path = "../yolo_plot_scripts/confidences.csv"
write_header = not os.path.exists(csv_path)
f = open(csv_path, "a", newline="")
writer = csv.writer(f)
if write_header:
    writer.writerow(["frame_id", "class", "confidence"])

csv_time_path = "../yolo_plot_scripts/inference_times.csv"
write_time_header = not os.path.exists(csv_time_path)
f_time = open(csv_time_path, "a", newline="")
writer_time = csv.writer(f_time)
if write_time_header:
    writer_time.writerow(["frame_id", "inference_time_ms"])

# ================= CARLA =================
WIDTH, HEIGHT = 1000, 800
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARLA - YOLO + PID Lane Following")

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
client.load_world("Track5")
world = client.get_world()

world.set_weather(carla.WeatherParameters(
    cloudiness=80.0,
    sun_altitude_angle=90.0
))

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.kart.kart')

spawn = carla.Transform(
    carla.Location(x=-7, y=-15, z=0.5),
    carla.Rotation(yaw=-15)
)

vehicle = world.try_spawn_actor(vehicle_bp, spawn)
if not vehicle:
    raise RuntimeError("Vehicle not spawned")

camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '90')

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=-1, y=-0.5, z=1)),
    attach_to=vehicle
)

# ================= PID =================
kp, ki, kd = 0.005, 0.0001, 0.001
prev_error = 0.0
integral = 0.0

control = carla.VehicleControl()
control.throttle = 0.3
control.brake = 0.0

camera_image = None
frame_id = 0
last_time = time.time()

def process_image(image):
    global camera_image, frame_id
    global prev_error, integral, last_time
    global control

    frame_id += 1
    now = time.time()
    dt = now - last_time
    last_time = now

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]

    start = time.time()
    results = model(img)
    infer_ms = (time.time() - start) * 1000.0
    writer_time.writerow([frame_id, infer_ms])

    left_cones = []
    right_cones = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        writer.writerow([frame_id, cls, conf])

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cls_name = CLASSES[cls]

        if cls_name == "blue_cone":
            left_cones.append((cx, cy))

        elif cls_name == "yellow_cone":
            right_cones.append((cx, cy))

    annotated = results[0].plot()

    left_cones.sort(key=lambda p: p[1])
    right_cones.sort(key=lambda p: p[1])

    centerline = []
    for l, r in zip(left_cones, right_cones):
        centerline.append((
            int((l[0] + r[0]) / 2),
            int((l[1] + r[1]) / 2)
        ))

    if len(centerline) >= 4:
        pts = np.array(centerline)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=20)
        u = np.linspace(0, 1, 100)
        xs, ys = splev(u, tck)

        spline = list(zip(xs.astype(int), ys.astype(int)))

        for i in range(len(spline) - 1):
            cv2.line(annotated, spline[i], spline[i + 1], (0, 255, 0), 2)

        lookahead = 30
        tx, ty = spline[min(lookahead, len(spline) - 1)]

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

# ================= LOOP =================
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if camera_image:
        screen.blit(camera_image, (0, 0))
    pygame.display.flip()

    vel = vehicle.get_velocity()
    speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
    print(f"Speed {speed:.2f} m/s | Steer {control.steer:.2f}")

# ================= CLEANUP =================
f.close()
f_time.close()
camera.destroy()
vehicle.destroy()
pygame.quit()
print("Finished")

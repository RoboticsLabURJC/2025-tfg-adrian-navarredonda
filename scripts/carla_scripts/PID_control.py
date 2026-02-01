import carla
import pygame
import numpy as np
import cv2
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = "../../models/runs/detect/Run_con_parametros/weights/best.pt"
model = YOLO(MODEL_PATH)

CLASSES = ["blue_cone", "large_orange_cone", "orange_cone", "unknown_cone", "yellow_cone"]

WIDTH, HEIGHT = 1000, 800
KP_STEER = 0.02   # Proporcional en BEV (error en metros)
THROTTLE = 0.6
LOOKAHEAD_Y = 10.0  # metros adelante para referencia
SINGLE_ROW_OFFSET = 1.5  # metros desde fila cuando falta la otra

# ===================== PYGAME =====================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARLA - BEV Lane Following")

# ===================== CARLA =====================
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
client.load_world("Track5")
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.kart.kart')

spawn_point = carla.Transform(
    carla.Location(x=6.09819214, y=0.07350677, z=0.5),
    carla.Rotation(yaw=0)
)

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Unable to spawn vehicle")
    exit()
print("Vehicle spawned")

# ===================== CAMERA =====================
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(
    carla.Location(x=-0.5, y=-0.68, z=1.2)
)

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

camera_image = None
control = carla.VehicleControl()

# ===================== Funcion para proyectar imagen =====================

def project_image(pixel_x, pixel_y, camera_transform, focal=WIDTH/2):
    """
    Proyecta un pixel de la imagen al plano del suelo (z=0)
    usando modelo de cámara pinhole simple.
    """
    # Normalizado de -1 a 1
    nx = (pixel_x - WIDTH / 2) / focal
    ny = (pixel_y - HEIGHT / 2) / focal

    # Supongamos que el plano del suelo z=0
    cam_loc = camera_transform.location
    cam_rot = camera_transform.rotation

    pitch = np.radians(cam_rot.pitch)
    yaw = np.radians(cam_rot.yaw)

    # Vector de dirección de cámara (aprox)
    dir_x = nx
    dir_y = ny*np.cos(pitch) + np.sin(pitch)
    dir_z = -ny*np.sin(pitch) + np.cos(pitch)

    if dir_z == 0:
        dir_z = 1e-6

    t = -cam_loc.z / dir_z
    world_x = cam_loc.x + dir_x * t
    world_y = cam_loc.y + dir_y * t

    return world_x, world_y

def fit_lane(cones):
    if len(cones) < 2:
        return None
    xs = np.array([p[0] for p in cones])
    ys = np.array([p[1] for p in cones])
    a, b = np.polyfit(ys, xs, 1)  # x = a*y + b
    return a, b

# ===================== IMAGE PROCESS =====================
MAX_CONE_DIST = 6.0
def process_image(image):
    global camera_image, control

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    bgr = array.copy()

    results = model(bgr)

    left_cones = []
    right_cones = []

    veh_transform = vehicle.get_transform()
    veh_x = veh_transform.location.x
    veh_y = veh_transform.location.y

    # Convertir cada cono a coordenadas BEV (metros)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        xyxy = box.xyxy[0]

        px = int((xyxy[0] + xyxy[2]) / 2)
        py = int((xyxy[1] + xyxy[3]) / 2)

        wx, wy = project_image(px, py, camera.get_transform())

        # Filtrar conos demasiado lejos
        dist = np.sqrt((wx - veh_x)**2 + (wy - veh_y)**2)
        if dist > MAX_CONE_DIST:
            print("Distancia cono: ", dist)
            continue  # ignoramos este cono

        if CLASSES[cls] == "blue_cone":
            right_cones.append((wx, wy))  # derecha
        elif CLASSES[cls] == "yellow_cone":
            left_cones.append((wx, wy))  # izquierda

    lane_center_x = None

    left_line = fit_lane(left_cones)
    right_line = fit_lane(right_cones)

    if left_line and right_line:
        # Ambos lados visibles
        x_left = left_line[0]*LOOKAHEAD_Y + left_line[1]
        x_right = right_line[0]*LOOKAHEAD_Y + right_line[1]
        lane_center_x = (x_left + x_right) / 2

    elif left_line and not right_line:
        # Solo izquierda -> offset hacia la derecha
        x_left = left_line[0]*LOOKAHEAD_Y + left_line[1]
        a, _ = left_line
        normal = np.array([1, -a])
        normal = normal / np.linalg.norm(normal)
        lane_center_x = x_left + SINGLE_ROW_OFFSET * normal[0]

    elif right_line and not left_line:
        # Solo derecha -> offset hacia la izquierda
        x_right = right_line[0]*LOOKAHEAD_Y + right_line[1]
        a, _ = right_line
        normal = np.array([1, -a])
        normal = normal / np.linalg.norm(normal)
        lane_center_x = x_right - SINGLE_ROW_OFFSET * normal[0]

    else:
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0

    if lane_center_x is not None:
        # Steering basado en error lateral (metros)
        vehicle_x = vehicle.get_transform().location.x
        error = lane_center_x - vehicle_x
        control.steer = KP_STEER * error
        control.throttle = THROTTLE
        control.brake = 0.0

    # Visualización (opcional)
    annotated = results[0].plot()
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    camera_image = pygame.surfarray.make_surface(rgb.swapaxes(0,1))

camera.listen(process_image)

# ===================== MAIN LOOP =====================

running = True
clock = pygame.time.Clock()

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        vehicle.apply_control(control)

        if camera_image:
            screen.blit(camera_image, (0, 0))

        pygame.display.flip()
        clock.tick(30)

except KeyboardInterrupt:
    print("Interrupted")

# ===================== CLEANUP =====================
camera.stop()
camera.destroy()
vehicle.destroy()
pygame.quit()

print("Session ended cleanly")

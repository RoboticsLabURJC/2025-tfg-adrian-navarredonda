import carla
import time
import pygame
import numpy as np
import socket

# Socket config
HOST = 'localhost'
PORT = 1977

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print(f"Waiting for connection {HOST}:{PORT}...")
conn, addr = s.accept()
print(f"Connected from {addr}")

# Carla config
WIDTH, HEIGHT = 1000, 800
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - Remote control")

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)

client.load_world("Track5")
world = client.get_world()

weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.kart.kart')

spawn_point = carla.Transform(
    carla.Location(x=-7, y=-15, z=0.5),
    carla.Rotation(yaw=-15)
)

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Unable to spawn vehicle")
    exit()
print("Vehicle spawned correctly")

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '90')
#camera_transform = carla.Transform(carla.Location(x=-1, y=-0.5, z=1))
camera_transform = carla.Transform(carla.Location(x=1, y=-2.5, z=0.5), carla.Rotation(yaw=90))

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

camera_image = None
def process_image(image):
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))
camera.listen(process_image)

# Control vars
control = carla.VehicleControl()
current_steer = 0.0
current_throttle = 0.0
current_brake = 0.0

running = True
while running:
    try:
        buffer = conn.recv(1024).decode(errors='ignore').strip()
        if not buffer:
            continue

        lines = buffer.splitlines()
        for line in lines:
            if "[ABS_X]" in line and "[R2]" in line and "[L2]" in line:
                try:
                    parts = line.strip().split("[ABS_X]")
                    if len(parts) > 1:
                        vals = parts[1].split("[R2]")
                        if len(vals) == 2:
                            steer_val = int(vals[0].strip())
                            rest = vals[1].split("[L2]")
                            if len(rest) == 2:
                                r2_val = int(rest[0].strip())
                                l2_val = int(rest[1].strip())

                                # Escalar steer de 0-255 a -1 a 1
                                current_steer = (steer_val - 127) / 128.0
                                current_steer = max(-1.0, min(1.0, current_steer))

                                # Escalar R2 (aceleración) de 0-255 a 0.0-1.0
                                current_throttle = max(0.0, min(1, r2_val / 255.0))

                                # Escalar L2 (freno) de 255-0 a 0.001-0.1
                                brake_normalized = 1.0 - (l2_val / 255.0)
                                current_brake =  max(0.0, min(1.0, l2_val / 255.0))
                except Exception as e:
                    print("Error parsing line:", e)
                    continue

        # Apply control
        control.steer = current_steer
        control.throttle = current_throttle
        control.brake = current_brake
        vehicle.apply_control(control)

        # Show camera
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if camera_image:
            screen.blit(camera_image, (0, 0))
        pygame.display.flip()

        # Speed display
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        print(f"Speed: {speed:.2f} m/s | Steer: {control.steer:.2f} | Throttle: {control.throttle:.2f} | Brake: {control.brake:.3f}")

    except KeyboardInterrupt:
        print("Interrupted")
        break

# Cleanup
camera.destroy()
vehicle.destroy()
pygame.quit()
conn.close()
s.close()
print("Session ended.")
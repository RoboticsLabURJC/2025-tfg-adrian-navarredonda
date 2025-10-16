import carla
import random
import time
import numpy as np
import pygame

# Inicializar Pygame
pygame.init()
display_width = 1000
display_height = 800
display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Vista desde la cámara de Carla")

# Conectar al servidor de CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

client.load_world("Track3")
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Vehículo
vehicle_bp = blueprint_library.find('vehicle.kart.kart')
transform = carla.Transform(carla.Location(x=0, y=0, z=1), carla.Rotation(yaw=0))
vehicle = world.try_spawn_actor(vehicle_bp, transform)

# Cámara
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(display_width))
camera_bp.set_attribute('image_size_y', str(display_height))
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-10))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Variable global para el último frame recibido
image_surface = None

def process_image(image):
    global image_surface
    # Convertir datos sin asumir que siempre llega bien formateado
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    expected_size = image.width * image.height * 4
    if array.size != expected_size:
        return  # A veces llegan frames incompletos; los ignoramos

    # Reshape a (height, width, 4)
    array = np.reshape(array, (image.height, image.width, 4))
    # Convertir BGRA -> RGB
    rgb_array = array[:, :, :3][:, :, ::-1]

    # Asegurar que tiene 3 canales y tipo uint8
    if rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
        image_surface = pygame.surfarray.make_surface(np.rot90(rgb_array))


camera.listen(lambda image: process_image(image))

# Bucle principal
try:
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if image_surface is not None:
            display.blit(image_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)

finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()













































# import carla
# import random
# import time
# import cv2
# import numpy as np

# # Conectar al servidor de Carla
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)

# client.load_world("Track1")
# world = client.get_world()
# blueprint_library = world.get_blueprint_library()

# # Vehículo
# vehicle_bp = blueprint_library.find('vehicle.kart.kart')
# transform = carla.Transform(carla.Location(x=0, y=0, z=1), carla.Rotation(yaw=0))
# vehicle = world.try_spawn_actor(vehicle_bp, transform)

# # Cámara
# camera_bp = blueprint_library.find('sensor.camera.rgb')
# camera_bp.set_attribute('image_size_x', '800')
# camera_bp.set_attribute('image_size_y', '600')
# camera_bp.set_attribute('fov', '90')

# camera_transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-10))
# camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# # Callback para procesar imágenes
# def process_image(image):
#     # Convertir el frame de CARLA a un array numpy
#     array = np.frombuffer(image.raw_data, dtype=np.uint8)
#     array = array.reshape((image.height, image.width, 4))  # CARLA da BGRA
#     frame = array[:, :, :3]  # Quitar canal alpha
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    
#     cv2.imshow("Vista tercera persona", frame)
#     cv2.waitKey(1)  # 1ms para refrescar ventana

# camera.listen(lambda image: process_image(image))

# try:
#     while True:
#         time.sleep(0.1)
# finally:
#     camera.stop()
#     camera.destroy()
#     vehicle.destroy()
#     cv2.destroyAllWindows()


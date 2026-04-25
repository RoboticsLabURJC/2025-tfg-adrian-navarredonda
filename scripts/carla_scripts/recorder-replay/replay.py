#!/usr/bin/env python3
#
#
#  Copyright (C) URJC DeepRacer
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/. 
#
#  Author : Roberto Calvo Palomino <roberto.calvo at urjc dot es
#           Sergio Robledo <s.robledo.2021 at alumnos dot urjc dot es>


import carla
import pygame
import numpy as np
import sys
import time
import os
import argparse
import csv
import cv2
from pathlib import Path

import queue
from queue import Queue

from dataset_manager import DatasetSaver

from ultralytics import YOLO

MODEL_PATH = "../../yolo_model/runs/detect/Run_con_parametros/weights/best.pt"
model = YOLO(MODEL_PATH)
CLASSES = ["blue_cone", "large_orange_cone", "orange_cone", "unknown_cone", "yellow_cone"]


RATE_CONTROL_LOOP = 30


def get_log_duration(client, log_file):
    import re           
    info = client.show_recorder_file_info(log_file, False)  
    # Look at for Duration: 12.34 s"
    match = re.search(r"Duration:\s+([0-9.]+)", info)
    if not match:
        raise RuntimeError("Duration time cannot be read!")
    return float(match.group(1))

def replay_loop(args, view="car"):

    pygame.init()
    pygame.display.set_caption(f"CARLA Replay - Replay view {view} ")
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height))

    client = carla.Client('localhost', args.port)  
    client.set_timeout(10.0)

    world = client.get_world()
    
    path = Path(args.log_path)
    logs = list(path.glob("*.log"))    
    if (len(logs) == 0):
        print(f"Error, no log file found in {args.log_path}")
        exit(-1)    

    p = logs[0]
    if not p.is_absolute():
        p = Path.cwd() / p
    log_filename = str(p)
    print(f'Using log file {log_filename}')

    duration = get_log_duration(client, log_filename)
    duration = duration + world.get_snapshot().timestamp.elapsed_seconds
    print(f"Replaying: {log_filename}, duration: {duration:.2f} s")

    dataset = None
    if args.generate_dataset_path is not None:        
        dataset = DatasetSaver(args.generate_dataset_path)
    
    client.replay_file(log_filename, 0, 0, 0)

    blueprint_library = world.get_blueprint_library()
    
    actors = None

    if (view == "car"):
        actors = world.get_actors().filter("vehicle.tesla.model3")
    elif (view == "bike"):
        actors = world.get_actors().filter("vehicle.diamondback.century")
    elif (view == "kart"):
        actors = world.get_actors().filter("vehicle.kart.kart")

    if not actors:
        raise RuntimeError("No vehicles found in the replay")

    vehicle = actors[0]  # first vehicle is ego
    print(f"Using ego con id={vehicle.id}, type={vehicle.type_id}")
    
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(display_width))
    camera_bp.set_attribute("image_size_y", str(display_height))
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    frame_q = Queue(maxsize=1)   # save (rgb, bgr, (w, h))

    def _safe_put(q: Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def process_image(image):
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = np.reshape(bgra, (image.height, image.width, 4))
        bgr  = bgra[:, :, :3].copy()
        rgb  = bgr[:, :, ::-1]
        _safe_put(frame_q, (rgb, bgr, (image.width, image.height)))

    camera.listen(lambda img: process_image(img))

    clock = pygame.time.Clock()

    # Start at a relative time 0.0 to syncronize with speed csv
    t0_sim = 0.0

    try:
        while True:         
            
            clock.tick(RATE_CONTROL_LOOP)

            snapshot = world.get_snapshot()  
            sim_time = snapshot.timestamp.elapsed_seconds                      
            
            if sim_time >= duration:
                print("Replay finished")
                break

            try:
                rgb, bgr, (w, h) = frame_q.get_nowait()
            except queue.Empty:
             
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Get relative time for speedcsv/replay sync
            if t0_sim == 0.0:
                t0_sim = sim_time

            rel_time = sim_time - t0_sim

            
            #if image_surface is not None:
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))

            pygame.display.flip()

            if dataset is not None:
                results = model(bgr)
                # Generate dataset
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                h, w = hsv.shape[:2]

                # máscara global vacía
                mask_cones = np.zeros((h, w), np.uint8)

                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf < 0.5:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                    # Filtro de distancia
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h
                    if box_area < 100:
                        continue

                    # Filtro de separacion hacia los extremos
                    cx = (x1 + x2) // 2
                    center_x = w // 2
                    offset = abs(cx - center_x)
                    if offset > w * 0.45:
                        continue

                    # recorte ROI
                    hsv_roi = hsv[y1:y2, x1:x2]
                    s_roi = hsv_roi[:, :, 1]

                    # ===== BLUE =====
                    if cls == 0: 
                        mask = cv2.inRange(
                            hsv_roi,
                            np.array([10, 50, 70]),
                            np.array([180, 255, 255])
                        )
                        mask_sat = cv2.inRange(s_roi, 70, 255)
                        mask = cv2.bitwise_and(mask, mask_sat)

                        mask_cones[y1:y2, x1:x2][mask > 0] = 1

                    # ===== YELLOW =====
                    elif cls == 4:
                        mask = cv2.inRange(
                            hsv_roi,
                            np.array([18, 50, 120]),
                            np.array([40, 255, 255])
                        )
                        mask_cones[y1:y2, x1:x2][mask > 0] = 2

                    # ===== ORANGE =====
                    elif cls == 2:
                        mask = cv2.inRange(
                            hsv_roi,
                            np.array([5, 100, 100]),
                            np.array([18, 255, 255])
                        )
                        mask_cones[y1:y2, x1:x2][mask > 0] = 3

                mask_cones_rgb = np.zeros_like(bgr)
                mask_cones_rgb[mask_cones == 1] = [0, 0, 255]     # azul
                mask_cones_rgb[mask_cones == 2] = [255, 255, 0]   # amarillo
                mask_cones_rgb[mask_cones == 3] = [255, 165, 0]   # naranja

                mask_y = cv2.inRange(hsv, np.array([18, 50, 150]), np.array([40, 255, 255]))
                mask_w = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))

                mask_c = np.zeros(mask_w.shape, np.uint8)
                mask_c[mask_w > 0] = 1
                mask_c[mask_y > 0] = 2

                mask_rgb = np.zeros_like(bgr)
                mask_rgb[mask_c == 1] = [255, 255, 255]   # blanco
                mask_rgb[mask_c == 2] = [255, 255, 0]     # amarillo

                # ================= Escalado de la imagen con mascaras ===============
                scale = 0.25  # ajusta según necesites

                new_w = int(bgr.shape[1] * scale)
                new_h = int(bgr.shape[0] * scale)

                mask_cones_scaled = cv2.resize(mask_cones_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # ============== ROI  =================
                x, y = 0, 75   
                w_roi, h_roi = 200, 66

                # Asegurar que no nos salimos de la imagen
                x = max(0, min(x, new_w - w_roi))
                y = max(0, min(y, new_h - h_roi))

                # Recorte
                mask_cones_roi = mask_cones_scaled[y:y+h_roi, x:x+w_roi]

                # You can get the controls of the vehicule at each snapshot
                # ctrl = vehicle.get_control()
                # print(ctrl.throttle, ctrl.steer, ctrl.brake)

                # ================ Control ============
                ctrl = vehicle.get_control()
                throttle = float(ctrl.throttle)
                steer    = max(-1.0, min(1.0, float(ctrl.steer)))
                brake    = float(ctrl.brake)
                speed = 0.0
                
                dataset.save_sample(rel_time, bgr, mask_rgb, mask_cones_roi, throttle, steer, brake, speed)


    except KeyboardInterrupt:
        print("Exit...")
    except Exception as e:
        print(e)
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()

        vehicle.destroy()
        
        if dataset is not None:

            # Takes both dataset and speed CSV files and do the matching
            path = Path(args.log_path)
            logs = list(path.glob("*.csv"))    
            if (len(logs) == 0):
                print(f"Error, no data csv file found in {args.log_path}")
                exit(-1) 

            csv_data_filename = str(logs[0])

            dataset.adjust_speed(csv_data_filename)


        pygame.quit()
        sys.exit()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="recorder")

    parser.add_argument("--log_path", type=str, required=True,
                        help="Directory where log files will be loaded")
    
    parser.add_argument("--port", "--carla-port", type=int, default=3010,
                        help="Port used to connect to the CARLA simulator")
    
    parser.add_argument("--tport","--carla-traffic-port", type=int, default=3020,
                        help="Port used by the CARLA traffic manager")
    
    parser.add_argument("--generate_dataset_path",  type=str, default=None,
                        help="Enable dataset generation and set the path to save it")
    
    parser.add_argument(
                        "--dataset_types", "--carla-dataset-types",
                        nargs="+",
                        choices=["rgb", "mask", "segmented", "all"],
                        default=["all"],
                        metavar="TYPE",
                        help=(
                            "Types of frames to export. Options: rgb, mask, segmented, all. "
                            "Example: --dataset_types rgb mask"
                        )
                    )
    args = parser.parse_args()

    # Use "bike" or "car" to choose from where point of view you want to replay de simulation
    replay_loop(args, "kart")

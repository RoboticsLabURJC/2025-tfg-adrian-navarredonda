import socket
from evdev import InputDevice, categorize, ecodes, list_devices
import select

HOST = 'localhost'
PORT = 1977

# Search for device
devices = [InputDevice(path) for path in list_devices()]
joystick = None
for device in devices:
    if "Sony" in device.name or "Wireless Controller" in device.name or "PS4" in device.name:
        joystick = device
        break

if not joystick:
    print("No PS4 controller found")
    exit()

print(f"? Using device: {joystick.path} - {joystick.name}")

# Connect
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("? Connected to receiver")

    abs_x = 0
    abs_z = 0     # L2
    abs_rz = 0    # R2

    for event in joystick.read_loop():
        if event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_X:      # Left joystick (steer)
                abs_x = event.value
            elif event.code == ecodes.ABS_Z:    # L2
                abs_z = event.value
            elif event.code == ecodes.ABS_RZ:   # R2
                abs_rz = event.value

            # Send values
            msg = f"[ABS_X] {abs_x}[R2] {abs_rz}[L2] {abs_z}\n"
            s.sendall(msg.encode())
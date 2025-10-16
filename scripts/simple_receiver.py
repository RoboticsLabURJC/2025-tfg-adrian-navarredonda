import socket

"""
This code simply allows toy to check whether messages have been received or not
"""

HOST = 'localhost'
PORT = 1977

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Waiting for connection on {HOST}:{PORT}...")
    conn, addr = s.accept()
    with conn:
        print(f"Connected from {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            msg = data.decode().strip()
            print(f"{msg}")
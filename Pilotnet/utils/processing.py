import os

def check_path(path):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

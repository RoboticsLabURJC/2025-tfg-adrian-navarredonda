from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import os, csv

class PilotNetDataset(Dataset):
    def __init__(self, folder_paths, mirrored=False, transform=None, preprocessing=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        self.mirrored = mirrored
        self.preprocessing = preprocessing

        for folder in folder_paths:
            csv_path = os.path.join(folder, "dataset.csv")
            if not os.path.exists(csv_path):
                print(f"[Warning] No se encontró: {csv_path}")
                continue

            with open(csv_path, "r") as f:
                for row in csv.DictReader(f):
                    img_rel = row["mask_path"].lstrip("/")
                    img_abs = os.path.join(folder, img_rel)
                    if not os.path.isfile(img_abs):
                        print(f"[Warning] Falta imagen: {img_abs}")
                        continue
                    steer = float(row["steer"]); throttle = float(row["throttle"])
                    self.image_paths.append(img_abs); self.labels.append([steer, throttle])
                    if self.mirrored:
                        self.image_paths.append((img_abs, "mirror"))
                        self.labels.append([-steer, throttle])

        self.image_shape = (66, 200, 3) # tamano de imagen por 3 canales
        self.num_labels = 2

        # Transform por defecto si no pasan uno
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        entry = self.image_paths[idx]
        mirrored = isinstance(entry, tuple) and entry[1] == "mirror"
        img_path = entry[0] if isinstance(entry, tuple) else entry

        img = Image.open(img_path).convert("RGB")

        if mirrored:
            img = ImageOps.mirror(img)

        img = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, y

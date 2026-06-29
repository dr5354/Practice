import os
import cv2
import torch
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=128):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size

        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        print(f"[DATASET] Loaded: {len(self.files)} images")

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # Оставляем нормализацию 0..1
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img

    def _load_bbox(self, label_path):
        if not os.path.exists(label_path):
            return torch.zeros(4, dtype=torch.float32)
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        if not lines:
            return torch.zeros(4, dtype=torch.float32)
            
        line = lines[0].strip().split()

        if len(line) < 5:
            return torch.zeros(4, dtype=torch.float32)

        _, xc, yc, w, h = map(float, line)

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        txt_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(self.label_dir, txt_name)

        x = self._load_image(img_path)
        bbox = self._load_bbox(label_path)

        return x, bbox
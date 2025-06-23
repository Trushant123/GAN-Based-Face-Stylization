import os
import random
import json
from PIL import Image
from torch.utils.data import Dataset


class FaceStyleDataset(Dataset):
    def __init__(self, content_root, style_root, style_map_path, transform=None):
        self.content_root = content_root
        self.style_root = style_root
        self.transform = transform

        # Load all face image paths
        self.face_images = [os.path.join(content_root, f) for f in os.listdir(content_root) if f.endswith(('.jpg', '.png'))]

        # Load style mapping: {"anime": 0, "cartoon": 1, ...}
        with open(style_map_path, 'r') as f:
            self.style_map = json.load(f)

        # Build list of style images per class
        self.style_images = {}
        for style_class in self.style_map:
            class_dir = os.path.join(style_root, style_class)
            self.style_images[style_class] = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]

        self.num_styles = len(self.style_map)

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        # Load face image
        face_path = self.face_images[idx]
        face_img = Image.open(face_path).convert("RGB")

        # Randomly pick a style class and image
        style_class = random.choice(list(self.style_map.keys()))
        style_id = self.style_map[style_class]
        style_path = random.choice(self.style_images[style_class])
        style_img = Image.open(style_path).convert("RGB")

        if self.transform:
            face_img = self.transform(face_img)
            style_img = self.transform(style_img)

        return face_img, style_img, style_id

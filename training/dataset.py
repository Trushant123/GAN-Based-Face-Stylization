import os
import random
from PIL import Image
from torch.utils.data import Dataset

class StylizationDataset(Dataset):
    def __init__(self, face_dir, style_dir, transform=None):
        self.face_paths = [os.path.join(face_dir, fname) for fname in os.listdir(face_dir) if fname.endswith(('.jpg', '.png'))]
        self.style_paths = [os.path.join(style_dir, fname) for fname in os.listdir(style_dir) if fname.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        face_path = self.face_paths[idx]
        style_path = random.choice(self.style_paths)

        face_img = Image.open(face_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")

        if self.transform:
            face_img = self.transform(face_img)
            style_img = self.transform(style_img)

        return face_img, style_img
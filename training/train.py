import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from model import StylizerNet
from losses import VGGFeatures, perceptual_loss, style_loss, IdentityLoss

# --------- Configs ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_dir = "../data/faces"
style_dir = "../data/styles"
batch_size = 8
epochs = 20
lr = 1e-4

# --------- Transforms ---------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --------- Dataset ---------
class StylizationDataset(torch.utils.data.Dataset):
    def __init__(self, face_folder, style_folder, transform):
        self.face_paths = [os.path.join(face_folder, x) for x in os.listdir(face_folder)]
        self.style_paths = [os.path.join(style_folder, x) for x in os.listdir(style_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        face_img = self.transform(Image.open(self.face_paths[idx]).convert("RGB"))
        style_img = self.transform(Image.open(random.choice(self.style_paths)).convert("RGB"))
        return face_img, style_img

# --------- Initialize ---------
from PIL import Image
import random

model = StylizerNet().to(device)
vgg = VGGFeatures().to(device)
id_loss_fn = IdentityLoss(device=device)
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = StylizationDataset(face_dir, style_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --------- Training Loop ---------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for face_img, style_img in train_loader:
        face_img = face_img.to(device)
        style_img = style_img.to(device)

        optimizer.zero_grad()
        stylized = model(face_img, style_img)

        loss_c = perceptual_loss(vgg, stylized, face_img)
        loss_s = style_loss(vgg, stylized, style_img)
        loss_i = id_loss_fn(stylized, face_img)
        
        loss = loss_c + loss_s + loss_i
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# --------- Save Model ---------
torch.save(model.state_dict(), "../training/checkpoints/stylizer_net.pth")

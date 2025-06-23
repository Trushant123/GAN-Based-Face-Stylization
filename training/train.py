import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FaceStyleDataset
from model import StylizerNet
from losses import PerceptualLoss, StyleLoss, IdentityLoss
from tqdm import tqdm

# ---------- Load Config ----------
with open("training/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# ---------- Style ID Mapping ----------
with open(config["style_classes_path"], 'r') as f:
    style_to_id = json.load(f)
num_styles = len(style_to_id)

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor()
])

# ---------- Dataset & Loader ----------
dataset = FaceStyleDataset(
    content_root=config["content_dir"],
    style_root=config["style_dir"],
    style_map_path=config["style_classes_path"],
    transform=transform
)
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StylizerNet(num_styles=num_styles).to(device)

# ---------- Losses ----------
vgg_loss = PerceptualLoss(weight=1.0).to(device)
style_loss = StyleLoss(weight=10.0).to(device)
identity_loss = IdentityLoss(arcface_model=lambda x: torch.nn.functional.normalize(x.mean(dim=[2, 3]), dim=1), weight=5.0).to(device)  # dummy encoder

# ---------- Optimizer ----------
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# ---------- Training Loop ----------
print("\n[INFO] Starting training...")
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0

    for content_img, style_img, style_id in tqdm(loader):
        content_img = content_img.to(device)
        style_img = style_img.to(device)
        style_id = style_id.to(device)

        output = model(content_img, style_id)

        loss_c = vgg_loss(output, content_img)
        loss_s = style_loss(output, style_img)
        loss_i = identity_loss(output, content_img)

        total_loss = loss_c + loss_s + loss_i

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch {epoch+1}/{config['num_epochs']} | Loss: {running_loss / len(loader):.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["checkpoint_path"])

print("\n[INFO] Training complete.")
import os
import json
import base64
import torch
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms
from model import StylizerNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Style Classes ----
with open("app/style_classes.json", "r") as f:
    style_to_id = json.load(f)
num_styles = len(style_to_id)

# ---- Load Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StylizerNet(num_styles=num_styles).to(device).eval()
model.load_state_dict(torch.load("training/checkpoints/stylizer_net.pth", map_location=device))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

inv_transform = transforms.ToPILImage()

@app.post("/stylize")
async def stylize(face: UploadFile = File(...), style_name: str = Form(...)):
    try:
        if style_name not in style_to_id:
            return JSONResponse(status_code=400, content={"error": "Invalid style name."})

        style_id = torch.tensor([style_to_id[style_name]], dtype=torch.long).to(device)

        face_img = Image.open(BytesIO(await face.read())).convert("RGB")
        face_tensor = transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor, style_id)

        output_img = inv_transform(output.squeeze(0).clamp(0, 1).cpu())
        buf = BytesIO()
        output_img.save(buf, format='JPEG')
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        return JSONResponse(content={"result": img_str})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
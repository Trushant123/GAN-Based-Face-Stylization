from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import StylizerNet
import base64

app = FastAPI()

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = StylizerNet().eval().cuda()
model.load_state_dict(torch.load("training/checkpoints/stylizer_net.pth"))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

inv_transform = transforms.Compose([
    transforms.ToPILImage()
])

@app.post("/stylize")
async def stylize(face: UploadFile = File(...), style: UploadFile = File(...)):
    try:
        face_img = Image.open(BytesIO(await face.read())).convert("RGB")
        style_img = Image.open(BytesIO(await style.read())).convert("RGB")

        face_tensor = transform(face_img).unsqueeze(0).cuda()
        style_tensor = transform(style_img).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(face_tensor, style_tensor)

        output_img = inv_transform(output.squeeze(0).cpu().clamp(0, 1))

        buf = BytesIO()
        output_img.save(buf, format='JPEG')
        return JSONResponse(content={"result": base64.b64encode(buf.getvalue()).decode('utf-8')})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

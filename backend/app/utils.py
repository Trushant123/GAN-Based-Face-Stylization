import torchvision.transforms as transforms
from PIL import Image
import io
import base64

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

def get_inverse_transform():
    return transforms.Compose([
        transforms.ToPILImage()
    ])

def image_bytes_to_tensor(image_bytes, transform):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def tensor_to_base64(tensor):
    image = transforms.ToPILImage()(tensor.squeeze(0).clamp(0, 1))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
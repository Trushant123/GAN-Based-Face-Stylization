import torch
from model import StylizerNet

# Wrapper for loading model for inference
class InferenceModel:
    def __init__(self, weights_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = StylizerNet().to(self.device).eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def predict(self, face_tensor, style_tensor):
        with torch.no_grad():
            output = self.model(face_tensor.to(self.device), style_tensor.to(self.device))
        return output.cpu()

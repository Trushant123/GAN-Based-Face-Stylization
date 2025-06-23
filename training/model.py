import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Content Encoder ---------
class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


# --------- Style Encoder ---------
class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.encoder(x)


# --------- AdaIN (Adaptive Instance Normalization) ---------
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean = style_feat.mean(dim=1, keepdim=True)
        style_std = style_feat.std(dim=1, keepdim=True)
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True)
        normalized = (content_feat - content_mean) / (content_std + 1e-5)
        stylized = normalized * style_std.view(-1, size[1], 1, 1) + style_mean.view(-1, size[1], 1, 1)
        return stylized


# --------- Decoder ---------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decode(x)


# --------- Stylizer Network ---------
class StylizerNet(nn.Module):
    def __init__(self):
        super(StylizerNet, self).__init__()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def forward(self, face_img, style_img):
        content_feat = self.content_encoder(face_img)
        style_feat = self.style_encoder(style_img)
        style_feat_expanded = style_feat.unsqueeze(-1).unsqueeze(-1).expand_as(content_feat)
        fused = self.adain(content_feat, style_feat_expanded)
        out = self.decoder(fused)
        return out

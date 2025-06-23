import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --------- VGG Feature Extractor ---------
class VGGFeatures(nn.Module):
    def __init__(self, layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1")):
        super(VGGFeatures, self).__init__()
        self.layer_names = layers
        vgg = models.vgg19(pretrained=True).features
        self.selected = {
            "relu1_1": 1,
            "relu2_1": 6,
            "relu3_1": 11,
            "relu4_1": 20
        }
        self.model = nn.Sequential(*[vgg[i] for i in range(max(self.selected.values()) + 1)])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer_idx in self.selected.items():
            x = self.model[:layer_idx + 1](x)
            features[name] = x
        return features


# --------- Perceptual Loss ---------
class PerceptualLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatures()
        self.weight = weight

    def forward(self, output, target):
        f_output = self.vgg(output)
        f_target = self.vgg(target)
        loss = 0.0
        for key in f_output.keys():
            loss += F.l1_loss(f_output[key], f_target[key])
        return self.weight * loss


# --------- Style Loss ---------
class StyleLoss(nn.Module):
    def __init__(self, weight=10.0):
        super(StyleLoss, self).__init__()
        self.vgg = VGGFeatures()
        self.weight = weight

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, output, style_target):
        f_output = self.vgg(output)
        f_style = self.vgg(style_target)
        loss = 0.0
        for key in f_output.keys():
            G_o = self.gram_matrix(f_output[key])
            G_s = self.gram_matrix(f_style[key])
            loss += F.l1_loss(G_o, G_s)
        return self.weight * loss


# --------- Identity Loss ---------
class IdentityLoss(nn.Module):
    def __init__(self, arcface_model, weight=5.0):
        super(IdentityLoss, self).__init__()
        self.arcface = arcface_model
        self.weight = weight

    def forward(self, output, input_face):
        feat_out = self.arcface(output)
        feat_in = self.arcface(input_face)
        return self.weight * (1 - F.cosine_similarity(feat_out, feat_in).mean())
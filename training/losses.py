import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from facenet_pytorch import InceptionResnetV1

# --------- VGG Perceptual and Style Loss ---------
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*vgg[:4])   # relu1_1
        self.slice2 = nn.Sequential(*vgg[4:9])  # relu2_1
        self.slice3 = nn.Sequential(*vgg[9:16]) # relu3_1
        self.slice4 = nn.Sequential(*vgg[16:23])# relu4_1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


def gram_matrix(feature):
    (b, c, h, w) = feature.size()
    features = feature.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)


def perceptual_loss(vgg, output, target):
    output_feats = vgg(output)
    target_feats = vgg(target)
    loss = 0.0
    for o, t in zip(output_feats, target_feats):
        loss += F.l1_loss(o, t)
    return loss


def style_loss(vgg, output, style_img):
    output_feats = vgg(output)
    style_feats = vgg(style_img)
    loss = 0.0
    for o, s in zip(output_feats, style_feats):
        gram_o = gram_matrix(o)
        gram_s = gram_matrix(s)
        loss += F.l1_loss(gram_o, gram_s)
    return loss


# --------- Identity Loss using ArcFace ---------
class IdentityLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(IdentityLoss, self).__init__()
        self.arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        for param in self.arcface.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        out_emb = self.arcface(F.interpolate(output, size=(160,160)))
        tgt_emb = self.arcface(F.interpolate(target, size=(160,160)))
        cosine_sim = F.cosine_similarity(out_emb, tgt_emb, dim=-1)
        return 1 - cosine_sim.mean()
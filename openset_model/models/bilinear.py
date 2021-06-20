import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .registry import register_model

class BilinearOneLayer(nn.Module):
    def __init__(self, net_dim=[2048, 2048], embed_dim=1024, num_classes=414, **kwargs):
        super(BilinearOneLayer, self).__init__()
        embed = []
        for ndim in net_dim:
            fc = nn.Sequential(
                nn.Linear(ndim, embed_dim),
                nn.BatchNorm1d(embed_dim, affine=True),
                nn.ReLU(inplace=True),
                #nn.Dropout()
            )
            embed.append(fc)
        
        self.embed = nn.ModuleList(embed)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, feats):
        feats_arr = []
        for i, feat in enumerate(feats):
            feat = self.embed[i](feat)
            feats_arr.append(feat.unsqueeze(-1))

        feats_arr = torch.cat(feats_arr, dim=-1)
        feats_prod = torch.prod(feats_arr, -1)
        res = self.fc(feats_prod)
        return res

class BilinearTwoLayer(nn.Module):
    def __init__(self, net_dim=[2048, 2048], embed_dim=1024, num_classes=414, **kwargs):
        super(BilinearTwoLayer, self).__init__() 
        embed1 = []
        embed2 = []
        for ndim in net_dim:
            fc1 = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(ndim, embed_dim),
                nn.BatchNorm1d(embed_dim, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )
            fc2 = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim, affine=True),
                nn.ReLU(inplace=True),
                #nn.Dropout()
            )
            embed1.append(fc1)
            embed2.append(fc2)

        self.embed1 = nn.ModuleList(embed1)
        self.embed2 = nn.ModuleList(embed2)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, feats):
        feats_arr = []
        for i, feat in enumerate(feats):
            feat = self.embed1[i](feat)
            feat = self.embed2[i](feat)
            feats_arr.append(feat.unsqueeze(-1))

        feats_arr = torch.cat(feats_arr, dim=-1)
        feats_prod = torch.prod(feats_arr, -1)
        res = self.fc(feats_prod)
        return res

@register_model
def bilinear_onelayer(net_dim=[2048, 2048], embed_dim=1024, num_classes=414, **kwargs):
    return BilinearOneLayer(net_dim, embed_dim, num_classes)

@register_model
def bilinear_twolayer(net_dim=[2048, 2048], embed_dim=1024, num_classes=414, **kwargs):
    return BilinearTwoLayer(net_dim, embed_dim, num_classes)
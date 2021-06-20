# Unofficial pytorch implementation for Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
# https://arxiv.org/pdf/1805.07836.pdf

import torch
from torch import nn
import torch.nn.functional as F


#class GeneralEntropyLoss(nn.Module):
#    def __init__(self, k=0.5, q=0.7, k_ukn=0.0, total_epoch=100):
#        super(GeneralEntropyLoss, self).__init__()
#        self.k = k
#        self.q = q
#        self.k_ukn = k_ukn
#        self.total_epoch = total_epoch
#        self.trunc = (1 - pow(self.k, self.q)) / self.q
#
#    def forward(self, x, y, epoch):
#        prob = F.softmax(x, dim=1)
#        y_labels = y.unsqueeze(-1)
#        ukn_idns = (y_labels[:, 0] == 0)
#        P = torch.gather(prob, 1, y_labels)
#        mask = (P <= self.k).float()
#        mask[ukn_idns] = (P[ukn_idns] <= self.k_ukn).float()
#
#        loss = ((1 - torch.pow(P, self.q)) / self.q) * (1 - mask) + self.trunc * mask
#        return loss.mean()

class GeneralEntropyLoss(nn.Module):
    def __init__(self, k=0., q=0.7):
        super(GeneralEntropyLoss, self).__init__()
        self.k = k #cfg.LOSSES.GXENT_K
        self.q = q #cfg.LOSSES.GXENT_Q
        self.trunc = (1 - pow(self.k, self.q)) / self.q

    def forward(self, x, y):
        prob = F.softmax(x, dim=1)
        y_labels = y.unsqueeze(-1)
        P = torch.gather(prob, 1, y_labels)
        mask = (P <= self.k).type(torch.cuda.FloatTensor)

        loss = ((1 - torch.pow(P, self.q)) / self.q) * (1 - mask) + self.trunc * mask
        loss = loss.mean()
        return loss
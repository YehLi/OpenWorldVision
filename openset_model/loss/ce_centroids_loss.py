import torch
import torch.nn as nn
import torch.nn.functional as F

from .disc_centroids_loss import DiscCentroidsLoss


class CECentroidsLoss(nn.Module):
    """
    CE Loss + Centroids Loss
    """
    def __init__(self, num_classes, feat_dim, lambda_centr, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(CECentroidsLoss, self).__init__()
        self.centroids_loss = DiscCentroidsLoss(num_classes, feat_dim)
        self.lambda_centr = lambda_centr
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def cuda(self):
        super().cuda()
        self.centroids_loss.cuda()

    def forward(self, x, feat, target):
        # cross entropy loss
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss_ce = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()

        # centroids loss
        loss_centr = self.centroids_loss(feat, target)

        return loss_ce + self.lambda_centr * loss_centr, loss_ce, self.lambda_centr * loss_centr
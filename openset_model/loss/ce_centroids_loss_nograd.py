import torch
import torch.nn as nn
import torch.nn.functional as F

from .disc_centroids_loss_nograd import DiscCentroidsLossNoGrad


class CECentroidsLossNoGrad(nn.Module):
    """
    CE Loss + Centroids Loss
    """
    def __init__(self, num_classes, feat_dim, lambda_centr):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(CECentroidsLossNoGrad, self).__init__()
        self.centroids_loss = DiscCentroidsLossNoGrad(num_classes, feat_dim)
        self.lambda_centr = lambda_centr

    def cuda(self):
        super().cuda()
        self.centroids_loss.cuda()

    def forward(self, x, feat, target):
        # cross entropy loss
        loss_ce = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()

        # centroids loss
        _, _target = target.max(dim=1)
        loss_centr = self.centroids_loss(feat, _target)

        return loss_ce + self.lambda_centr * loss_centr, loss_ce, self.lambda_centr * loss_centr
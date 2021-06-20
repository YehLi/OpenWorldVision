import torch
import torch.nn as nn
import torch.nn.functional as F


class RplLoss(nn.Module):
    """
    RPL loss = L_c + lambda_o * L_o.
    """
    def __init__(self, gamma=0.5, lambda_o=0.1, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(RplLoss, self).__init__()
        self.gamma = gamma
        self.lambda_o = lambda_o
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, model, feat, target):
        # compute dist to reciprocal points
        dist_to_rp = torch.cdist(feat, model.P, p=2).square().reshape(
            feat.size(0), model.num_classes, model.num_rp_per_cls)  # (N, C, 8)
        if not (target == 0).all():
            # L_c
            logits = self.gamma * torch.mean(dist_to_rp, dim=2)
            logprobs = F.log_softmax(logits, dim=-1)[target != 0]
            known_target = target[target != 0].unsqueeze(1) - 1
            nll_loss = -logprobs.gather(dim=-1, index=known_target)
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            L_c = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
            # L_o
            known_dist_to_rp = dist_to_rp[target != 0, ...]
            open_dist = (known_dist_to_rp[torch.arange(known_dist_to_rp.size(0)), known_target, :] \
                         - model.R[known_target].unsqueeze(dim=1)).square()
            L_o = open_dist.mean()
        else:
            L_c = feat.sum() * 0.
            L_o = feat.sum() * 0.
        # total loss
        loss = L_c + self.lambda_o * L_o

        return loss, L_c, L_o

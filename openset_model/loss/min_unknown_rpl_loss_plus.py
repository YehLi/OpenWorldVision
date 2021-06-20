import torch
import torch.nn as nn
import torch.nn.functional as F


class MinUnknownRplLossPlus(nn.Module):
    """
    RPL loss = L_c + lambda_o * L_o.
    """
    def __init__(self, gamma=0.5, lambda_o=0.1, lambda_unknown=0.1, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(MinUnknownRplLossPlus, self).__init__()
        self.gamma = gamma
        self.lambda_o = lambda_o
        self.lambda_unknown = lambda_unknown
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, logits, dist_to_rp, open_dist, target):
        # L_c
        logprobs = F.log_softmax(logits, dim=-1)
        known_target = target.unsqueeze(1)
        nll_loss = -logprobs.gather(dim=-1, index=known_target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        L_c = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
        # L_o
        _open_dist = open_dist[:, known_target.reshape(-1), :]
        L_o = _open_dist.mean()
        # L_unknown
        if (target == 0).any():
            unknown_dist_to_rp = dist_to_rp[target == 0, ...]  # (U, C, 8)
            unknown_dist_to_rp = unknown_dist_to_rp.reshape(unknown_dist_to_rp.size(0), -1)  # (U, Cx8)
            _, min_inds = unknown_dist_to_rp.min(dim=1)
            unknown_dist_to_rp_min = unknown_dist_to_rp[torch.arange(unknown_dist_to_rp.size(0)), min_inds]
            L_unknown = unknown_dist_to_rp_min.mean()
        else:
            L_unknown = dist_to_rp.sum() * 0.

        # total loss
        loss = L_c + self.lambda_o * L_o + self.lambda_unknown * L_unknown

        return loss, L_c, self.lambda_o * L_o, self.lambda_unknown * L_unknown

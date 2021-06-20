import torch
import torch.nn as nn
import torch.nn.functional as F


class CESupMoCoLoss(nn.Module):
    """
    CE Loss + Centroids Loss
    """
    def __init__(self, lambda_supmoco):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(CESupMoCoLoss, self).__init__()
        self.lambda_supmoco = lambda_supmoco
        self.contrastive_loss = nn.CrossEntropyLoss()

    def forward(self, x, target, selfsup_logits, selfsup_labels):
        # cross entropy loss
        loss_ce = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()

        # contrastive loss
        loss_selfsup = torch.sum(-selfsup_labels * F.log_softmax(selfsup_logits, dim=-1), dim=-1) / selfsup_labels.sum(dim=1)
        loss_selfsup = loss_selfsup.mean()

        return loss_ce + self.lambda_supmoco * loss_selfsup, loss_ce, self.lambda_supmoco * loss_selfsup
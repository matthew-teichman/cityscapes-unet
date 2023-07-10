import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-7):
        # flatten label and prediction tensors
        # calculate dice coefficient
        intersection = (inputs * targets).sum(dim=(2, 3))
        dice = (2.*intersection + eps) / (inputs.pow(2).sum(dim=(2, 3)) + targets.pow(3).sum(dim=(2, 3)) + eps)
        # pytorch optimizers minimize a loss. In this case, we would luke to
        # maximize the dice loss, so we return the negative dice loss.
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        # reduction = 'mean': the sum of the output will be divided by the number
        # of elements in the output.
        # BCELoss does not include Sigmoid
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, inputs, targets, eps=1e-7):
        bce_loss = self.bce(inputs, targets)
        # flatten label and prediction tensors
        # BCE Loss applies signmoid
        intersection = (inputs * targets).sum(dim=(2, 3))
        dice = (2.*intersection + eps) / (inputs.pow(2).sum(dim=(2, 3)) + targets.pow(3).sum(dim=(2, 3)) + eps)
        dice_loss = 1.0 - dice.mean()
        loss_total = bce_loss + dice_loss
        return loss_total

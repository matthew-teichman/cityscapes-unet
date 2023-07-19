import sys
sys.path.append("../")
import torch
from cityscapes_unet.loss import DiceLoss, BCEDiceLoss
import unittest


def dice_loss(output: torch, label: torch):
    return -3.22

def entropy_dice_loss(output:torch, label: torch):
    return 2.22

class TestLoss(unittest.TestCase):

    def test_dice_loss(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_function = DiceLoss()
        label = torch.rand(32, 7, 512, 256).to(device)
        output = torch.rand(32, 7, 512, 256).to(device)
        loss = loss_function(output, label)
        self.assertAlmostEqual(loss.detach().data.item(), dice_loss(output, label))

    def test_entropy_dice_loss(self):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        loss_function = BCEDiceLoss()
        label = torch.rand(32, 7, 512, 256).to(device)
        output = torch.rand(32, 7, 512, 256).to(device)
        loss = loss_function(output, label)
        self.assertAlmostEqual(loss.detach().data.item(), dice_loss(output, label))


if __name__ == '__main__':
    unittest.main()
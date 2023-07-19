import sys
sys.path.append("../")
import torch
import unittest
from cityscapes_unet.model import UNet

class TestUnet(unittest.TestCase):

    def test_unet(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(32, 3, 7).to(device)
        x = torch.rand(32, 3, 512, 256).to(device)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (32, 7, 512, 256))


if __name__ == '__main__':
    unittest.main()
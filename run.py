import os
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader

from cityscape import CitySegmentation
from loss import DiceLoss, BCEDiceLoss
from train import train
from model import UNet


if __name__ == '__main__':
    # resize images
    dim = (256, 512)

    # data augmentation
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(dim, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])

    mask_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(dim, interpolation=transforms.InterpolationMode.NEAREST)])

    # define dataset
    train_set = CitySegmentation(dataset="train", image_transforms=image_transforms,
                                 mask_transforms=mask_transforms, dimensions=dim)

    valid_set = CitySegmentation(dataset="val", image_transforms=image_transforms,
                                 mask_transforms=mask_transforms, dimensions=dim)

    # load data loader object from dataset
    batch_size = 8
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, shuffle=True)

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # random seed generator
    seed = 10

    # training hyperparameters
    learning_rate = 0.01
    epochs = 30

    # define loss function
    loss_function = DiceLoss()
    #loss_function = BCEDiceLoss()

    # define model
    n_channels = 3
    n_classes = 7
    filter_size = 32

    model = UNet(filter_size, n_channels, n_classes)
    model.load_state_dict(torch.load(os.getcwd() + "\\model\\saved_model_epoch49.pth"))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # path to save model
    path = "model"

    # training loop
    model = train(path, model, train_dataloader, valid_dataloader, loss_function, optimizer, device, seed, epochs)

import os
import sys
import torch.optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from cityscapes_unet.cityscape import CitySegmentation
from cityscapes_unet.loss import DiceLoss, BCEDiceLoss
from cityscapes_unet.training import train
from cityscapes_unet.model import UNet


class Cityscapes_UNet:
    def __init__(self,
                pretrained_model: str="",
                model_saved_folder: str="",
                input_channels: int=3,
                labels: list=['all'],
                hidden_layer_size: int=16) -> None:
        
        self.pretrained_model = pretrained_model
        self.model_saved_folder = model_saved_folder
        self.labels = ['void', 'road', 'non_road', 'construction', 'traffic_sign', 'human', 'vehicle'] if labels[0] == "all" else labels
        self.input_channels = input_channels
        self.output_classes = len(self.labels)
        self.hidden_layer_size = hidden_layer_size

        self.model = UNet(self.hidden_layer_size, self.input_channels, self.output_classes)
        if (self.pretrained_model != ""):
            self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), model_saved_folder, pretrained_model)))

    def train(self,
            path_cityscape_dataset: str,
            image_resize: tuple,
            loss_function: str,
            optimizer:str,
            learning_rate: float,
            epochs: int,
            batch_size: str) -> UNet:
        
        # data augmentation
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_resize, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

        mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_resize, interpolation=transforms.InterpolationMode.NEAREST)])

        # define dataset
        train_set = CitySegmentation(
            image_root=path_cityscape_dataset,
            dataset="train",
            image_transforms=image_transforms,
            mask_transforms=mask_transforms,
            dimensions=image_resize,
            labels=self.labels)

        valid_set = CitySegmentation(
            image_root=path_cityscape_dataset,
            dataset="val",
            image_transforms=image_transforms,
            mask_transforms=mask_transforms,
            dimensions=image_resize,
            labels=self.labels)
        
        # load data loader object from dataset
        if batch_size == "auto":
            pass
            # to-do: calulate maximum batch-size based on gpu memory avaliable
        else:
            batch_size = int(batch_size)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, shuffle=True)

        # device selection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # random seed generator
        seed = 10

        # define loss function
        if loss_function == "dice":
            loss_function = DiceLoss()
        elif loss_function == "entropy_dice":
            loss_function = BCEDiceLoss()
        elif loss_function == "entropy":
            loss_function = nn.CrossEntropyLoss()
        else:
            print("Bad input optimizer argument")
            exit()  

        # define optimizer
        if optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            print("Bad input optimizer argument")
            exit()    

        # training loop
        self.model = train(self.model_saved_folder, self.model, train_dataloader, valid_dataloader, loss_function, optimizer, device, seed, epochs)

        # return UNet model
        return self.model


    def results(self):
        pass

    def export(self):
        pass
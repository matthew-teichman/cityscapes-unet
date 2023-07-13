import os
import sys
import torch.optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from cityscape import CitySegmentation
from loss import DiceLoss, BCEDiceLoss
from train import train
from model import UNet


if __name__ == '__main__':

    # python run.py -dataset "../Cityscape/dataset/"
    #               -model "saved_model_epoch49.pth"
    #               -hidden_layer 16
    #               -image_resize (256, 512)
    #               -loss_function dice
    #               -optimizer adam
    #               -learning_rate 1e-3
    #               -epochs 30
    #               -batch_size 15
    #               -model_save_folder "model"
    
    # training arguments
    arg_path_cityscape_dataset: str = ""
    arg_path_pretrained_model: str = ""
    arg_hidden_layer_size:int = 32
    arg_image_resize: tuple = (256, 512)
    arg_loss_function: str = "dice"
    arg_optimizer:str = "sgd"
    arg_learning_rate: float = 1e-3
    arg_epochs: int = 30
    arg_batch_size: str = "auto"
    arg_model_saved_folder: str = ""

    for i, arg in enumerate(sys.argv):
        match arg:
            case "-dataset":
                arg_path_cityscape_dataset = str(sys.argv[i+1])
            case "-model":
                arg_path_pretrained_model = str(sys.argv[i+1])
            case "-hidden_layer":
                arg_hidden_layer_size = int(sys.argv[i+1])
            case "-image_resize":
                arg_image_resize = tuple(sys.argv[i+1])
                i = 0
                num1 = []
                num2 = []
                for x in arg_image_resize:
                    if x == ",":
                        i += 1
                    else:
                        if i < 1:
                            num1.append(x)
                        else:
                            num2.append(x)
                print(''.join(num1))
                print(''.join(num2))
                arg_image_resize = (int(''.join(num1)), int(''.join(num2)))

            case "-loss_function":
                arg_loss_function = sys.argv[i+1].lower()
            case "-optimizer":
                arg_optimizer = sys.argv[i+1].lower()
            case "-learning_rate":
                arg_learning_rate = float(sys.argv[i+1])
            case "-epochs":
                arg_epochs = int(sys.argv[i+1])
            case "-batch_size":
                arg_batch_size = str(sys.argv[i+1])
            case "-model_save_folder":
                arg_model_saved_folder = str(sys.argv[i+1])

    # resize images
    dim = arg_image_resize

    # data augmentation
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(dim, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])

    mask_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(dim, interpolation=transforms.InterpolationMode.NEAREST)])

    # define dataset
    train_set = CitySegmentation(
        image_root=arg_path_cityscape_dataset,
        dataset="train",
        image_transforms=image_transforms,
        mask_transforms=mask_transforms,
        dimensions=dim)

    valid_set = CitySegmentation(
        image_root=arg_path_cityscape_dataset,
        dataset="val",
        image_transforms=image_transforms,
        mask_transforms=mask_transforms,
        dimensions=dim)

    # load data loader object from dataset
    if arg_batch_size == "auto":
        pass
        # to-do: calulate maximum batch-size based on gpu memory avaliable
    else:
        batch_size = int(arg_batch_size)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, shuffle=True)

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # random seed generator
    seed = 10

    # training hyperparameters
    learning_rate = arg_learning_rate
    epochs = arg_epochs

    # define loss function
    if arg_loss_function == "dice":
        loss_function = DiceLoss()
    elif arg_loss_function == "entropy_dice":
        loss_function = BCEDiceLoss()
    elif arg_loss_function == "entropy":
        loss_function = nn.CrossEntropyLoss()
    else:
        print("Bad input optimizer argument")
        exit()  


    # define model
    n_channels = 3
    n_classes = 7
    hidden_size = arg_hidden_layer_size

    model = UNet(hidden_size, n_channels, n_classes)
    if (len(arg_path_pretrained_model) > 0):
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), arg_model_saved_folder, arg_path_pretrained_model)))

    # define optimizer
    if arg_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif arg_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif arg_optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        print("Bad input optimizer argument")
        exit()    

    # path to save model
    path = arg_model_saved_folder

    # training loop
    model = train(path, model, train_dataloader, valid_dataloader, loss_function, optimizer, device, seed, epochs)

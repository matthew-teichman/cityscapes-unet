import sys
from cityscapes_unet.cityscapes_unet import Cityscapes_UNet

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
#               -labels "vehicle" "human"


if __name__ == '__main__':

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
    arg_labels = []

    if "-dataset" in sys.argv:
        index:int = sys.argv.index("-dataset")
        arg_path_cityscape_dataset = str(sys.argv[index+1])
    
    if "-model" in sys.argv:
        index:int = sys.argv.index("-model")
        arg_path_pretrained_model = str(sys.argv[index+1])
            
    if "-hidden_layer" in sys.argv:
        index:int = sys.argv.index("-hidden_layer")
        arg_hidden_layer_size = int(sys.argv[index+1])
    
    if "-image_resize" in sys.argv:
        index:int = sys.argv.index("-image_resize")
        resize_dim:list = sys.argv[index+1].split(',')
        arg_image_resize = tuple(int(value) for value in resize_dim)

    if "-loss_function" in sys.argv:
        index:int = sys.argv.index("-loss_function")
        arg_loss_function = sys.argv[index+1].lower()
    
    if "-optimizer" in sys.argv:
        index:int = sys.argv.index("-optimizer")
        arg_optimizer = sys.argv[index+1].lower()
    
    if "-learning_rate" in sys.argv:
        index:int = sys.argv.index("-learning_rate")
        arg_learning_rate = float(sys.argv[index+1])

    if "-epochs" in sys.argv:
        index:int = sys.argv.index("-epochs")
        arg_epochs = int(sys.argv[index+1])

    if "-batch_size" in sys.argv:
        index:int = sys.argv.index("-batch_size")
        arg_batch_size = str(sys.argv[index+1])

    if "-model_save_folder" in sys.argv:
        index:int = sys.argv.index("-model_save_folder")
        arg_model_saved_folder = str(sys.argv[index+1])

    if "-labels" in sys.argv:
        index:int = sys.argv.index("-labels")
        while index + 1 < len(sys.argv):
            arg_labels.append(sys.argv[index + 1])
            index += 1 

    cityscapes_unet = Cityscapes_UNet(pretrained_model=arg_path_pretrained_model,
                                      model_saved_folder=arg_model_saved_folder,
                                      labels=arg_labels,
                                      hidden_layer_size=arg_hidden_layer_size)
    
    unet = cityscapes_unet.train(arg_path_cityscape_dataset,
                                arg_image_resize,
                                arg_loss_function,
                                arg_optimizer,
                                arg_learning_rate,
                                arg_epochs,
                                arg_batch_size)
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from cityscapes_unet.model import UNet
from cityscapes_unet.cityscape import CityscapeDemo, CityscapeLabelsDecoder


def image_mask_overlay(img, mask, label_colors, decoder):
    img = img.cpu().numpy()
    mask = mask.cpu().numpy()

    img = np.transpose(img, [1, 2, 0])

    # decode image into rgb
    rgb_mask = decoder.decode_masks(mask, label_colors)

    img = img * 255
    img = np.asarray(img, np.uint8)
    rgb_mask = np.asarray(rgb_mask, np.uint8)
    image = cv2.addWeighted(img, 0.8, rgb_mask, 0.2, 0.0)

    return image


def results(num_results=10):
    # define image dimension should match model input
    dimensions = (256, 512)

    # define testing data, define a set of transformations to apply on image
    # should be same as training set.
    testdata = CityscapeDemo(transforms=transforms.Compose([
        transforms.Resize(dimensions, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()]))

    batch_size = 1
    train_dataloader = DataLoader(testdata, batch_size=batch_size, num_workers=1, shuffle=True)

    # define model
    n_channels = 3
    n_classes = 7
    filter_size = 32

    # load best model from testing
    model = UNet(filter_size, n_channels, n_classes)
    model.load_state_dict(torch.load(os.getcwd() + "\\model\\saved_model_epoch49.pth"))

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # covert model to cpu or gpu
    model.to(device)
    model.eval()

    # cityscape decode
    decoder = CityscapeLabelsDecoder()

    # mask prediction threshold
    threshold = 0.7

    # label of colors
    label_colors = [
        [0, 0, 0],  # 1) void - black
        [250, 0, 0],  # 2) road - blue
        [250, 215, 0],  # 3) non-road - orange
        [100, 100, 100],  # 4) construction - grey
        [0, 250, 0],  # 5) traffic - green
        [250, 225, 0],  # 6) human - yellow
        [0, 0, 250]  # 7) vehicle - red
    ]

    count = 0
    for i, img in enumerate(train_dataloader):
        # send tensor to cpu or gpu
        img = img.to(device)
        with torch.no_grad():
            output = model(img)

        # Loop through batch
        for i in range(output.shape[0]):
            image = img[i]
            mask = output[i]

            mask = (mask > threshold)

            if device != "cpu":
                image, mask = image.to("cpu"), mask.to("cpu")

            overlay = image_mask_overlay(image, mask, label_colors, decoder)
            cv2.imshow("result", overlay)
            cv2.waitKey(0)

        # break based on number of results
        if count >= num_results:
            break
        count += 1


def generate_video():
    fps = 30
    export_video_path = os.getcwd() + "\\videos\\demo_video.avi"

    # define image dimension should match model input
    dimensions = (256, 512)

    # define testing data, define a set of transformations to apply on image
    # should be same as training set.
    testdata = CityscapeDemo(transforms=transforms.Compose([
        transforms.Resize(dimensions, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()]))

    batch_size = 8
    test_dataloader = DataLoader(testdata, batch_size=batch_size, num_workers=1, shuffle=False)

    # define model
    n_channels = 3
    n_classes = 7
    filter_size = 32

    # load best model from testing
    model = UNet(filter_size, n_channels, n_classes)
    model.load_state_dict(torch.load(os.getcwd() + "\\model\\saved_model_epoch49.pth"))

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # covert model to cpu or gpu
    model.to(device)
    model.eval()

    # cityscape decode
    decoder = CityscapeLabelsDecoder()

    # mask prediction threshold
    threshold = 0.7

    # label of colors
    label_colors = [
        [0, 0, 0],  # 1) void - black
        [250, 0, 0],  # 2) road - red
        [250, 215, 0],  # 3) non-road - orange
        [100, 100, 100],  # 4) construction - grey
        [0, 250, 0],  # 5) traffic - green
        [250, 225, 0],  # 6) human - yellow
        [0, 0, 250]  # 7) vehicle - blue
    ]

    print("Processing Demo Frames...")
    video_overlay = []
    for i, img in tqdm(enumerate(test_dataloader), unit="batch", total=len(test_dataloader)):
        # send tensor to cpu or gpu
        img = img.to(device)
        with torch.no_grad():
            output = model(img)

        # Loop through batch
        for i in range(output.shape[0]):
            image = img[i]
            mask = output[i]

            if device != "cpu":
                image, mask = image.to("cpu"), mask.to("cpu")

            overlay = image_mask_overlay(image, mask, label_colors, decoder)
            video_overlay.append(overlay)

    print("Model Inferencing Complete!")
    print("Writing Video...")

    video_writer = cv2.VideoWriter(export_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, dimensions)

    for overlay in tqdm(video_overlay):
        video_writer.write(overlay)
    video_writer.release()
    print("Done!")





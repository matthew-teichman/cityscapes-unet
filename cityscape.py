import os, copy, json, time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class CityscapeLabelsDecoder:
    def __init__(self):
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.labels = [
            ("name", "id", "trainId", "category", "catId", "hasInstances", "ignoreInEval", "color"),
            ('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
            ('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
            ('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
            ('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
            ('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
            ('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
            ('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
            ('road', 7, 0, 'road', 1, False, False, (128, 64, 128)),
            ('sidewalk', 8, 1, 'non_road', 2, False, False, (244, 35, 232)),
            ('parking', 9, 255, 'non_road', 2, False, True, (250, 170, 160)),
            ('rail track', 10, 255, 'non_road', 2, False, True, (230, 150, 140)),
            ('building', 11, 2, 'construction', 3, False, False, (70, 70, 70)),
            ('wall', 12, 3, 'construction', 3, False, False, (102, 102, 156)),
            ('fence', 13, 4, 'construction', 3, False, False, (190, 153, 153)),
            ('guard rail', 14, 255, 'construction', 3, False, True, (180, 165, 180)),
            ('bridge', 15, 255, 'construction', 3, False, True, (150, 100, 100)),
            ('tunnel', 16, 255, 'construction', 3, False, True, (150, 120, 90)),
            ('traffic light', 19, 6, 'traffic_sign', 4, False, False, (250, 170, 30)),
            ('traffic sign', 20, 7, 'traffic_sign', 4, False, False, (220, 220, 0)),
            ('person', 24, 11, 'human', 5, True, False, (220, 20, 60)),
            ('rider', 25, 12, 'human', 5, True, False, (255, 0, 0)),
            ('car', 26, 13, 'vehicle', 6, True, False, (0, 0, 142)),
            ('truck', 27, 14, 'vehicle', 6, True, False, (0, 0, 70)),
            ('bus', 28, 15, 'vehicle', 6, True, False, (0, 60, 100)),
            ('caravan', 29, 255, 'vehicle', 6, True, True, (0, 0, 90)),
            ('trailer', 30, 255, 'vehicle', 6, True, True, (0, 0, 110)),
            ('train', 31, 16, 'vehicle', 6, True, False, (0, 80, 100)),
            ('motorcycle', 32, 17, 'vehicle', 6, True, False, (0, 0, 230)),
            ('bicycle', 33, 18, 'vehicle', 6, True, False, (119, 11, 32)),
            ('license plate', -1, -1, 'vehicle', 6, False, True, (0, 0, 142)),
        ]
        self.labels_df = pd.DataFrame(self.labels[1:], columns=self.labels[0])

    # Take single mask image RxGxB and exports NxHxW where N is number classes
    def hot_decode_masks(self, mask, mode="catId"):
        color_labels = self.labels_df["color"].to_numpy()
        class_labels = self.labels_df[mode].to_numpy()

        # makes mask of 35 class using the color pixel values
        masks = []
        for i, color in enumerate(color_labels):
            equality = np.equal(mask, color)
            class_map = np.all(equality, axis=-1)
            masks.append(class_map)
        masks = np.stack(masks, axis=-1)
        masks = masks.astype(int)

        # we don't need all the classes of masks so encode the masks to a simplified version.
        custom_mask = []
        head = 1
        unqiue_classes = np.unique(class_labels, return_index=True)[1]
        unqiue_classes = np.append(unqiue_classes, len(class_labels) - 1)

        for i in range(1, len(unqiue_classes)):
            mask = masks[:, :, unqiue_classes[i - 1]]
            while (head <= unqiue_classes[i] - 1):
                mask = (mask == 1) | (masks[:, :, head] == 1)
                head += 1

            custom_mask.append(mask)
        custom_mask = np.stack(custom_mask, axis=-1)
        custom_mask = custom_mask.astype(int)

        custom_mask = np.transpose(custom_mask, (2, 0, 1))

        return custom_mask

    def decode_masks(self, mask, label_colors):
        mask = np.squeeze(mask)
        red = torch.zeros((mask.shape[1], mask.shape[2]))
        green = torch.zeros((mask.shape[1], mask.shape[2]))
        blue = torch.zeros((mask.shape[1], mask.shape[2]))

        for i in range(mask.shape[0]):
            red = red + mask[i, :, :] * label_colors[i][0]          # Concatenation of Red Channels
            green = green + mask[i, :, :] * label_colors[i][1]      # Concatentation of Green Channels
            blue = blue + mask[i, :, :] * label_colors[i][2]        # Concatentation of Blue Channels

        rgb_mask = np.stack([red, green, blue], axis=-1)
        rgb_mask = rgb_mask.astype(int)
        return rgb_mask


class CitySegmentation(Dataset):
    def __init__(self, image_root='dataset\\leftImg8bit\\', dimensions=(256, 512),
                 mask_root='dataset\\gtFine\\', mode="catId", dataset="train", image_transforms=None,
                 mask_transforms=None):
        # add check if path exists
        self.mode = mode
        self.dim = dimensions
        self.img_root = os.path.join(os.getcwd(), image_root)
        self.mask_root = os.path.join(os.getcwd(), mask_root)
        self.imgs, self.masks, self.poly_files = self._get_pairs(dataset)
        assert (len(self.imgs) == len(self.masks))
        assert (len(self.masks) == len(self.poly_files))
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.poly = []
        self._json_polygon_objects()
        self._label_encoder_decoder = CityscapeLabelsDecoder()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        img = np.array(img)

        # tranform to PIL Image
        # resize image --> 256x512
        # convert to tensor
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        masks = Image.open(self.masks[idx]).convert('RGB')
        masks = np.array(masks)

        # tranform to PIL Image
        # resize image --> 256x512
        # convert to numpy
        if self.mask_transforms is not None:
            masks = self.mask_transforms(masks)
        masks = np.array(masks)
        masks = self._label_encoder_decoder.hot_decode_masks(masks, mode=self.mode)

        return img, masks

    def _get_pairs(self, dataset):
        img_path = []
        mask_path = []
        json_path = []
        for root, _, files in os.walk((os.path.join(self.img_root, dataset))):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_color')
                    jsonname = filename.replace('leftImg8bit', 'gtFine_polygons')
                    jsonname = jsonname.replace('png', 'json')
                    maskpath = os.path.join(self.mask_root, dataset, foldername, maskname)
                    jsonpath = os.path.join(self.mask_root, dataset, foldername, jsonname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath) and os.path.isfile(jsonpath):
                        img_path.append(imgpath)
                        mask_path.append(maskpath)
                        json_path.append(jsonpath)
                    else:
                        print("Cannot find the mask or image or json file: ", imgpath, maskpath, jsonpath)
        print("Found {} images in the folder {}".format(len(img_path), os.path.join(self.img_root, dataset)))
        return img_path, mask_path, json_path

    def __build_masks(self):
        print("Building Masks")
        collection_masks = []
        for mask in tqdm(self.masks):
            mask = Image.open(mask).convert('RGB')
            mask = self._resize_fcn(mask)
            mask = np.array(mask)
            mask = self._label_encoder_decoder.hot_decode_masks(mask, mode=self.mode)
            collection_masks.append(mask)
        return collection_masks

    def _json_polygon_objects(self):
        for file in self.poly_files:
            with open(file) as f:
                self.poly.append(json.load(f)['objects'])


class CityscapeDemo:
    def __init__(self, demo_root="evaluate\\demoVideo", transforms=None):
        self.root = os.path.join(os.getcwd(), demo_root)
        self.transforms = transforms
        self.images = []
        self.__get_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __get_images(self):
        for root, dirs, files in os.walk(self.root):
            for filename in files:
                if filename.endswith('.png'):
                    image_path = os.path.join(root, filename)
                    self.images.append(image_path)
        self.images.sort()
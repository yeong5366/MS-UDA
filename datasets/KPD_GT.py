import os
import os.path as path
import torch
from torch.utils import data
from torchvision import transforms, utils

import numpy as np
from PIL import Image


class KP_dataset_wGT(data.Dataset):
    """
    rgb_img : BxCxHxW
    th_img : Bx1xHxW
    label : BxHxW
    """

    def __init__(self, data_dir, split='day', input_folder='pseudo_KP', label_folder='labels', transform=None):

        assert (split in ['day', 'night', 'val_day', 'val_night']), 'split must be day | night | val_day | val_night |'

        with open(os.path.join(data_dir, 'filenames_KP', split + '_rgb.txt'), 'r') as file:
            self.rgb_names = [name.strip() for idx, name in enumerate(file)]
        with open(os.path.join(data_dir, 'filenames_KP', split + '_th.txt'), 'r') as file:
            self.th_names = [name.strip() for idx, name in enumerate(file)]

        assert len(self.rgb_names) == len(self.th_names)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.data_dir = data_dir
        self.domain = split
        self.inputs = input_folder

        self.label_folder = label_folder
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        name = (self.rgb_names[index]).replace('_visible', '', 1)
        name = name.split('.png')[0]

        rgb_name = self.rgb_names[index]
        rgb_path = os.path.join(self.data_dir, self.inputs, self.domain, rgb_name)
        rgb_image = Image.open(rgb_path)
        rgb_image = np.asarray(rgb_image, dtype=np.float32)  # HxWxC

        th_name = self.th_names[index]
        th_path = os.path.join(self.data_dir, self.inputs, self.domain, th_name)
        th_image = Image.open(th_path).convert('L')
        th_image = np.asarray(th_image, dtype=np.float32)  # HxWxC

        label_path = os.path.join(self.data_dir, self.label_folder, self.domain, name + '.png')
        label = Image.open(label_path)
        label = np.asarray(label, dtype=np.int64)

        pseudo_path = os.path.join(self.data_dir, self.inputs, self.domain, name + '_pseudo.png')
        pseudo = Image.open(pseudo_path)
        pseudo = np.asarray(pseudo, dtype=np.int64)

        if self.transform is not None:
            for func in self.transform:
                rgb_image, th_image, label, pseudo = func(rgb_image, th_image, label, pseudo)
        rgb_image = rgb_image.transpose((2, 0, 1)) / 255  # [0,255]->[0,1] CxHxW
        rgb_image = torch.tensor(rgb_image)
        rgb_image = self.normalize(rgb_image)
        th_image = th_image / 255
        th_image = torch.tensor(th_image)
        th_image = th_image.unsqueeze(0)

        label = torch.tensor(label)
        pseudo = torch.tensor(pseudo)

        return rgb_image, th_image, label, pseudo, name

    def __len__(self):
        return len(self.rgb_names)

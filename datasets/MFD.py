import os
import torch
from torch.utils import data
from torchvision import transforms, utils

import numpy as np
from PIL import Image

class MF_dataset(data.Dataset):
    """
    rgb_img : BxCxHxW
    th_img : Bx1xHxW
    label : BxHxW
    """

    def __init__(self, data_dir, split='day', transform=None, pseudo_folder = 'pseudo_all'):

        assert (split in ['day', 'night']), 'split must be day | night'

        with open(os.path.join(data_dir, split + '.txt'), 'r') as file:
            self.names = [name.strip() for idx, name in enumerate(file)]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.data_dir = data_dir
        self.image_folder = 'images'
        self.label_folder = 'labels'
        self.pseudo_folder = pseudo_folder
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        img_name = self.names[index]

        image_path = os.path.join(self.data_dir, self.image_folder, img_name + '.png')
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.float32)  # HxWxC

        label_path = os.path.join(self.data_dir, self.label_folder, img_name + '.png')
        label = Image.open(label_path)
        label = np.asarray(label, dtype=np.int64)

        pseudo_path = os.path.join(self.data_dir, self.pseudo_folder, img_name + '_pseudo.png')

        pseudo = Image.open(pseudo_path)
        pseudo = np.asarray(pseudo, dtype=np.int64)

        if self.transform is not None:
            for func in self.transform:
                image, label, pseudo = func(image, label, pseudo)  # horizontal flip

        image = image.transpose((2, 0, 1)) / 255  # [0,255]->[0,1] CxHxW
        image = torch.tensor(image)

        rgb_image = image[:3]
        th_image = image[3]
        rgb_image = self.normalize(rgb_image)
        th_image = th_image.unsqueeze(0)

        label = torch.tensor(label)
        pseudo = torch.tensor(pseudo)

        return rgb_image, th_image, label, pseudo, img_name

    def __len__(self):
        return len(self.names)
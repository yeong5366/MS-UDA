
import numpy as np
from PIL import Image


class RandomFlip_KP():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, rgb_image,th_image, label):
        if np.random.rand() < self.prob:
            # To prevent negative stride error in pytorch, subtract 0 from arrays
            rgb_image = rgb_image[:,::-1] - np.zeros_like(rgb_image)
            th_image = th_image[:, ::-1] - np.zeros_like(th_image)
            label = label[:,::-1]  - np.zeros_like(label)
            #The label can be pseudo label/real label

        return rgb_image, th_image, label

class RandomFlip_KP_fake():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, rgb_image,th_image, label, fake):
        if np.random.rand() < self.prob:
            rgb_image = rgb_image[:,::-1] - np.zeros_like(rgb_image)
            th_image = th_image[:, ::-1] - np.zeros_like(th_image)
            label = label[:,::-1] -np.zeros_like(label)
            fake = fake[:,::-1] - np.zeros_like(fake)

        return rgb_image, th_image, label, fake

class RandomFlip_KP_real_label():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, rgb_image,th_image, real_label, pseudo_label):
        if np.random.rand() < self.prob:
            rgb_image = rgb_image[:,::-1] - np.zeros_like(rgb_image)
            th_image = th_image[:, ::-1] - np.zeros_like(th_image)
            real_label = real_label[:,::-1] -np.zeros_like(real_label)
            pseudo_label = pseudo_label[:,::-1] - np.zeros_like(pseudo_label)

        return rgb_image, th_image, real_label, pseudo_label

class RandomFlip_MF():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label, pseudo):
        if np.random.rand() < self.prob:
            image = image[:,::-1] - np.zeros_like(image)
            label = label[:,::-1]  - np.zeros_like(label)
            pseudo = pseudo[:, ::-1] - np.zeros_like(pseudo)
        return image, label, pseudo

class RandomFlip_MF_fake():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label, pseudo, fake):
        if np.random.rand() < self.prob:
            image = image[:,::-1] - np.zeros_like(image)
            label = label[:,::-1]  - np.zeros_like(label)
            pseudo = pseudo[:, ::-1] - np.zeros_like(pseudo)
            fake = fake[:, ::-1] - np.zeros_like(fake)
        return image, label, pseudo, fake

#####################################################
"""
Below augmentation code is not used in the paper, but useful to test robustness of the network.
"""

class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label

class RandomCrop_pseudo():
    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label, pseudo):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]
            pseudo = pseudo[w1:w2, h1:h2]

        return image, label, pseudo

class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            image[w1:w2, h1:h2] = 0
            label[w1:w2, h1:h2] = 0

        return image, label


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            image = (image * bright_factor).astype(image.dtype)

        return image, label


class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w,h,c)
            )

            image = (image + noise).clip(0,255).astype(image.dtype)

        return image, label
        



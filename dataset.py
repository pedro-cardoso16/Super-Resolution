import PIL.Image
import torch
import torch.nn as nn
import PIL
from torch.utils.data import DataLoader, Dataset
import os


def load_img(filepath: str) -> PIL.Image.Image:
    """
    Loads an image from filepath

    :param filepath: path/to/image
    """
    img = PIL.Image.open(filepath).convert("YCbCr")
    y, _, _ = img.split()

    return y


class ImageSet(Dataset):
    """Creates Dataset from an image folder for super resolution"""

    def __init__(self, image_dir: str, input_transform=None, target_transform=None):
        super(ImageSet, self).__init__()

        # listing of files paths
        self.files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]

        # transformations
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index) -> any:
        input = load_img(self.files[index])
        target = input.copy()

        # apply transforms
        if self.input_transform:
            input = self.input_transform(input)

        if self.target_transform:
            target = self.target_transform(target)

        return input, target

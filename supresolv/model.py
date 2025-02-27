"""
Contains the possible NN models 
"""
# import PIL.Image
# import torch, torchvision
import torch.nn as nn
import torch.nn.init as init
# import PIL
# from torch.utils.data import DataLoader, Dataset
# import os


class ESPCN(nn.Module):
    """
    Neural Network for super resolution of images

    Real-Time Single Image and Video Super-Resolution Using an Efficient  
    Sub-Pixel Convolutional Neural Network [`arxiv<https://arxiv.org/abs/1609.05158>`_]
    """

    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        # self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        # self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        # self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        # x = self.pixel_shuffle(self.conv4(x))

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        # init.orthogonal_(self.conv4.weight)

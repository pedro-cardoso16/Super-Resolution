import PIL.Image
import torch, torchvision
import torch.nn as nn
import PIL
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.init as init


class ESPCN(nn.Module):
    """
    Neural Network for super resolution of images

    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    """

    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv4.weight)


# class SRNet2(nn.Module):
#     """
#     Neural Network for super resolution of images

#     `Image Super-Resolution Using Deep Convolutional Networks <https://arxiv.org/pdf/1501.00092>`_
#     """

#     def __init__(
#         self,
#         channels: int = 1,
#         n1: int = 64,
#         n2: int = 32,
#         f1: int = 9,
#         f2: int = 1,
#         f3: int = 5,
#     ):
#         """
#         Initializes the SRNet2 model.

#         :param n1: Number of filters in the first convolution.
#         :type n1: int
#         :param n2: Number of filters in the second convolution.
#         :type n2: int
#         :param f1: Kernel size for the first convolution.
#         :type f1: int
#         :param f2: Stride (kernel size) for the second convolution.
#         :type f2: int
#         :param f3: Kernel size for the third convolution.
#         :type f3: int

#         """
#         super(SRNet2, self).__init__()

#         self.relu = nn.ReLU()

#         self.conv1 = nn.Conv2d(channels, n1, f1, padding="same")
#         self.conv2 = nn.Conv2d(n1, n2, f2, padding="same")
#         self.conv3 = nn.Conv2d(n2, channels, f3, padding="same")

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))

#         return x

#     def _initialize_weights(self):
#         init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
#         init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
#         init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))

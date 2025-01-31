import torchvision
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def noise_image(img):
    # noise = torch.randn_like(img)
    noise = torch.normal(torch.zeros_like(img), std=1.0)
    return img + noise


class SuperResolutionSet(Dataset):
    '''

    '''
    def __init__(self, list_ids, labels):
        self.list_ids = list_ids
        self.labels = labels
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        id = self.list_ids[index]

        # Load data and get label
        image = torch.load('data/' + id + '.jpg')
        label = self.labels[id] # target

        return image, label

if __name__ == '__main__':
    img = torchvision.io.decode_image('Images/Denoising/test/3096.jpg')
    
    plt.figure()
    plt.imshow(img.numpy()[0])
    plt.show()

    pass
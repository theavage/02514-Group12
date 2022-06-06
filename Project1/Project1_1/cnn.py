import torch
import torch.nn as nn

class cnn():#insert input
    def __init__(self):
        super(self).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(3,8 , kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(8,16 , kernel_size=3, padding='same'),
                nn.ReLU())


    def forward(self, x):
        out = self.convolutional(x)
        return out
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def EfficientNetB7():

    #model = EfficientNet.from_name('efficientnet-b7')
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
    num_classes =1
    model.fc = nn.Linear(512, num_classes)

    return model

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

class resnet50():#insert input
    def __init__(self):
        super(self).__init__()
        self.convolutional = (
                ...,
                ...,
                ...,
                ...)


    def forward(self, x):
        out = x
        return out
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.models as models

def EfficientNetB7():

    model = EfficientNet.from_name('efficientnet-b7')
    #model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 1)
    #num_classes =1
    #model.fc = nn.Linear(512, num_classes)

    return model

class cnn(nn.Module):#insert input
    def __init__(self):
        super(self,cnn).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(128*128,128 , kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128,1, kernel_size=3, padding='same'),
                nn.ReLU())

    def forward(self, x):
        out = self.convolutional(x)
        return out

def resnet50():#insert input
    model = models.resnet50(pretrained=True)

    #If requires_grad is set to false: freezing the part of the model as no changes happen to its parameters. 
    for param in model.parameters():
        param.requires_grad = False   
        
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2))
    
    return model
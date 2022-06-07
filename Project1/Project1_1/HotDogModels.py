import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.models as models

def createEfficientNetB7():

    model = EfficientNet.from_pretrained('efficientnet-b7')
    model._fc= torch.nn.Linear(in_features=model._fc.in_features, out_features=2, bias=True)
    return model

class cnn(nn.Module):#insert input
    def _init_(self):
        super(cnn,self)._init_()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=3),
            nn.Dropout2d()
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(74420, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0),-1)
        x = self.fully_connected(x)
        return x

def createResNet50():#insert input
    model = models.resnet50(pretrained=True)

    #If requires_grad is set to false: freezing the part of the model as no changes happen to its parameters. 
    for param in model.parameters():
        param.requires_grad = False   
        
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2))
    
    return model
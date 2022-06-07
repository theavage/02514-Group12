import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def EfficientNetB7():

    # OPTION 1
    #model = EfficientNet.from_name('efficientnet-b7')
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
    num_classes =1
    model.fc = nn.Linear(512, num_classes)

    return model

    #OPTION 2
    #efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
    #                                'nvidia_efficientnet_b0', pretrained=True)





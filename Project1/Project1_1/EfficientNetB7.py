import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNetB7():

    # OPTION 1
    model = EfficientNet.from_name('efficientnet-b7')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 1)

    #OPTION 2
    #efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
    #                                'nvidia_efficientnet_b0', pretrained=True)





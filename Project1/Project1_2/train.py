import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
from model import *
from objectproposal import *

model = createModel()

train_ids = 0
val_ids = 0

indices = 0
images = 0
groundtruth = 0
classes = 0

trainset = createDataSet(images, indices, classes, groundtruth, train_ids)
valset = createDataSet(images, indices, classes, groundtruth, val_ids)

optimizer = 0
criterion = 0
batch_size = 0
num_epochs = 0

batch_size = 32
trainloader, valloader = DataLoader(trainset, batch_size=batch_size), DataLoader(trainset, batch_size=batch_size)

device = checkDevice()
model = trainModel(model, trainloader, valloader, optimizer, criterion, num_epochs, device)

path = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/model.pt'
torch.save(model.state_dict(), path)
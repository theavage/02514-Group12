import torch
from torch.utils.data import DataLoader

from utils import *
from model import *
from objectproposal import *

train_ids, val_ids, _, indices, groundtruth, classes, images = loadData()

trainset = createDataSet(images, indices, classes, groundtruth, train_ids)
valset = createDataSet(images, indices, classes, groundtruth, val_ids)

model = createModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
criterion = nn.MSELoss()
batch_size = 32
num_epochs = 10

batch_size = 32
trainloader, valloader = DataLoader(trainset, batch_size=batch_size), DataLoader(valset, batch_size=batch_size)

device = checkDevice()
model = trainModel(model, trainloader, valloader, optimizer, criterion, num_epochs, device)

path = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/model.pt'
torch.save(model.state_dict(), path)
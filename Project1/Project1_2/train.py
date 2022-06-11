import torch
from torch.utils.data import DataLoader

from utils import *
from model import *
from objectproposal import *

device = checkDevice()

train_ids, val_ids, _, indices, groundtruth, classes, images = loadData()

trainset = dataset('train')
valset = dataset('val')

model = createModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
criterion = nn.CrossEntropyLoss()
batch_size = 32
num_epochs = 5

batch_size = 32
trainloader, valloader = DataLoader(trainset, batch_size=batch_size), DataLoader(valset, batch_size=batch_size)



model, hist = trainModel(model, trainloader, valloader, optimizer, criterion, num_epochs, device)

path = 'model.pt'
torch.save(model.state_dict(), path)

hist_data = hist.items()
train_data = np.asarray(list(hist_data))
np.save('traindata', train_data)
plot_graphs(hist, 'trainplots.png')
import torch

from HotDogUtils import checkDevice, loadHotDogData, loadHotDogData, showHotDogData, trainNet
# from HotDogModels import ...

# Import models
from resnet50 import resnet50
from cnn import cnn

# Check if we run on GPU
device = checkDevice()

# Load data
train_loader, test_loader = loadHotDogData(128, 64)

# Show data
showHotDogData(train_loader)

# Load model
model = cnn
model.to(device)

#Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train model
model, train_acc, test_acc = trainNet(model, 5, train_loader, test_loader, device)
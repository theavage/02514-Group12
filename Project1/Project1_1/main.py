import torch

from HotDogUtils import checkDevice, loadHotDogData, loadHotDogData, showHotDogData, trainNet, results
from HotDogModels import resnet50, cnn, EfficientNetB7

SHOW_RESULTS = False

# Check if we run on GPU
device = checkDevice()

# Load data
# Check if it is augmented
isAugmented = True
train_loader, test_loader, trainset, testset = loadHotDogData(128, 64, isAugmented)

# Show data
showHotDogData(train_loader)

# Load model
model = EfficientNetB7
model = model()
model.to(device)

# Optimizers
sgd = torch.optim.SGD(model.parameters(), lr=0.1)
momentum = torch.optim.SGD(model.parameters(), lr=0.1, momentum=1)
rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.1)
adam = torch.optim.Adam(model.parameters(), lr=0.1)

# Train model
model, train_acc, test_acc = trainNet(model, 5, sgd, train_loader, test_loader,trainset,testset, device)

if(SHOW_RESULTS):
    results(device, test_loader)
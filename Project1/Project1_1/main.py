import torch
import torch.nn as nn

from HotDogUtils import checkDevice, loadHotDogData, loadHotDogData, showHotDogData, trainNet, saliency_map, plot_graphs
from HotDogModels import createEfficientNetB7

SHOW_RESULTS = False
isAugmented = False

name = "modelBLABLABLA.pt"
# ARCHITECTURE_OPTIMIZER_LOSS_EPOCHS_DA_BN.pt
path = "models/"
model_path = path + name


# Check if we run on GPU
device = checkDevice()

# Load data
# Check if it is augmented
train_loader, test_loader, trainset, testset = loadHotDogData(128, 64, isAugmented)

# Show data
showHotDogData(train_loader)

# Load model
model = createEfficientNetB7()
model.to(device)

# Optimizers
sgd = torch.optim.SGD(model.parameters(), lr=0.1)
momentum = torch.optim.SGD(model.parameters(), lr=0.1, momentum=1)
rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.1)
adam = torch.optim.Adam(model.parameters(), lr=0.1)

# Loss Functions
criterion = nn.CrossEntropyLoss()

# Train model
model, out_dict = trainNet(model, 10, sgd, criterion, train_loader, test_loader,trainset,testset, device)
torch.save(model, model_path)

plot_graphs(out_dict, name)

if SHOW_RESULTS: saliency_map(device, test_loader, model_path)
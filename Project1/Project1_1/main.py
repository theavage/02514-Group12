import torch

from HotDogUtils import checkDevice, loadHotDogData, loadHotDogData, showHotDogData, trainNet, saliency_map, plot_graphs
from HotDogModels import resnet50, cnn, EfficientNetB7

SHOW_RESULTS = False
isAugmented = False

model_path = "models/modelBLABLABLA.pt"

# Check if we run on GPU
device = checkDevice()

# Load data
# Check if it is augmented
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
model, out_dict = trainNet(model, 5, sgd, train_loader, test_loader,trainset,testset, device)
torch.save(model, model_path)

plot_graphs(out_dict)

if(SHOW_RESULTS):
    saliency_map(device, test_loader, model_path)
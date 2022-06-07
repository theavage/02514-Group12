import torch
import torch.nn as nn
from torchmetrics import HingeLoss
from PIL import Image
from HotDogUtils import checkDevice, loadHotDogData, loadHotDogData, showHotDogData, trainNet, saliency_map, plot_graphs
from HotDogModels import createEfficientNetB7, cnn, createResNet50

SHOW_RESULTS = False
isAugmented = False

name = "modelBLABLABLA.pt"
# ARCHITECTURE_OPTIMIZER_LOSS_EPOCHS_DA.pt
name_np = "models/modelBLABLABLA"
path = "models/"
model_path = path + name
print(model_path)


# Check if we run on GPU
device = checkDevice()

if (SHOW_RESULTS == False):    
    # Load data
    # Check if it is augmented
    train_loader, test_loader, trainset, testset = loadHotDogData(128, 64, isAugmented)

    # Show data
    #showHotDogData(train_loader)

    # Load model
    model = cnn
    model = model()
    model.to(device)

    # Optimizers
    sgd = torch.optim.SGD(model.parameters(), lr=0.1)  
    momentum = torch.optim.SGD(model.parameters(), lr=0.1, momentum=1) 
    adam = torch.optim.Adam(model.parameters(), lr=0.1)

    # Loss Functions
    criterion = nn.CrossEntropyLoss()
    hinge = HingeLoss()


    # Train model
    model, out_dict = trainNet(model, 3, sgd, criterion, train_loader, test_loader,trainset,testset, device)
    torch.save(model, model_path)

    plot_graphs(out_dict, name)

else:
    #LOAD IMAGE FROM COMPUTER
    image = Image.open('gpu_meme.jpg')
    saliency_map(device, image, model_path, name)


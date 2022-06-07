from cgi import test
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.notebook import tqdm
from random import randint


def checkDevice():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def showHotDogData(train_loader):
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(20,10))

    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')


def augmentedTransform(size):
    train_transform = transforms.Compose([
                                        transforms.Resize((size, size)),
                                        transforms.Pad(8),
                                        transforms.RandomRotation(180),  # .05 rad
                                        transforms.ColorJitter(hue=.05, saturation=.05),
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([
                                        transforms.Resize((size, size)),
                                        transforms.Pad(8),
                                        transforms.RandomRotation(180),  # .05 rad
                                        transforms.ColorJitter(hue=.05, saturation=.05),
                                        transforms.ToTensor()])

    return train_transform, test_transform

def loadHotDogData(size, batch_size, isAugmented):
    if isAugmented:
        train_transform, test_transform = augmentedTransform(size)
    else:
        train_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)
    return train_loader, test_loader, trainset, testset

class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def trainNet(model, num_epochs, optimizer, train_loader, test_loader,trainset,testset, device):

    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}

    # #Get the first minibatch
    # data = next(iter(train_loader))[0].cuda()
    # #Try running the model on a minibatch
    # print('Shape of the output from the convolutional part', model.convolutional(data).shape)
    # model(data); #if this runs the model dimensions fit

    for _ in tqdm(range(num_epochs), unit='epoch'):
        #For each epoch
        train_correct = 0
        train_loss = []
        for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = F.nll_loss(torch.log(output), target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            
            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1).cpu()
            test_correct += (target==predicted).sum().item()

        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")

    
    return model, out_dict


def plot_graphs(out_dict):

    plt.plot(out_dict['train_loss'],'-o')
    plt.plot(out_dict['train_acc'],'-o')
    plt.legend(('Train error','Train accuracy'))
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('images/train.png')

    plt.plot(out_dict['test_loss'],'-o')
    plt.plot(out_dict['test_acc'],'-o')
    plt.legend(('Test error','Test accuracy'))
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('images/test.png')


def saliency_map(device, test_loader, model_path):

    model = torch.load(model_path)
    model = model.to(device)

    model.eval()

    i = randint(0, 10)
    print("Chosen image ", i, "from the test set")
    images, labels = next(iter(test_loader))
    image = images[i]

    image = image.to(device)
    image.requires_grad_()


    output = model(image)
    # Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    #Retireve the saliency map and also pick the maximum value from channels on each pixel.
    saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
    #saliency = saliency.reshape(128, 128)


    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.show()


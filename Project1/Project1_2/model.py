import numpy as np
import torch
import torch.nn as nn
import torchvision

def createModel():
    model = torchvision.models.resnet50(pretrained=True)
    num_classes = 29
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def trainModel(model, trainloader, testloader, optimizer, criterion, num_epochs, device):

    out_dict = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for e in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = []
        for data, target in trainloader:
            data, target = data.to(device), data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            train_total += target.size(0)

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = []
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad(): output = model(data)
            loss = criterion(output, target)
            test_loss.append(loss.item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            test_total += target.size(0)

        out_dict['train_acc'].append(train_correct/train_total)
        out_dict['test_acc'].append(test_correct/test_total)
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))

        print('Epoch ' + str(e) + ', training accuracy: ' + str(out_dict['train_acc'][-1]) + ', test accuracy: ' + str(out_dict['test_acc'][-1]))

    return model, out_dict
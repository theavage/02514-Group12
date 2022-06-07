import torchvision.transforms as transforms

def augmentedTransform(isAug):
    if isAug:
        train_transform = transforms.Compose([
                                            transforms.Resize((128, 128)),
                                            transforms.Pad(8),
                                            transforms.RandomRotation(180),  # .05 rad
                                            transforms.ColorJitter(hue=.05, saturation=.05),
                                            transforms.ToTensor()])
        test_transform = transforms.Compose([
                                            transforms.Resize((128, 128)),
                                            transforms.Pad(8),
                                            transforms.RandomRotation(180),  # .05 rad
                                            transforms.ColorJitter(hue=.05, saturation=.05),
                                            transforms.ToTensor()])
    else:
        train_transform = transforms.Compose([transforms.Resize((128, 128)), 
                                        transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((128, 128)), 
                                        transforms.ToTensor()])
                                            

    return train_transform, test_transform
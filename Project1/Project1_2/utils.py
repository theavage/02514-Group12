import numpy as np
import torch
import torchvision
import cv2
import torchvision.transforms as transforms

from objectproposal import *
from utils import *

def checkDevice():

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cropAndResize(image, rectangle, size):

    x, y, w, h = rectangle
    x1, y1 = x, y
    x2, y2, = x + w, y + h
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, size)
    
    return resized

class dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=transforms.ToTensor()):
        self.transform = transform
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(self.data[idx, :, :, :]), self.transform(self.targets[idx])

def createDataSet(images, indices, classes, groundtruth, ids):
    for id in ids:
        mask = indices == id
        im = images[mask, :, :, :]
        gts = groundtruth[mask, :]
        cls = classes[mask]

        rects = createObjectProposals(im)
        rects, gts = torch.as_tensor(rects), torch.as_tensor(gts)
        iou = torchvision.ops.box_iou(rects, gts)
        iou = np.asarray(iou)

        gts_id = iou.argmax(axis=1)
        y_temp = cls[gts_id]

        size = (224, 224)
        X_temp = np.empty((0, 224, 224, 3))
        for rect in rects:
            im_temp = cropAndResize(im, rect, size)
            X_temp = np.concatenate(X, im_temp, axis=0)

        n_total = len(y_temp)
        n_object = np.count_nonzero(y_temp)
        object_ids = np.nonzero(y_temp)
        n_background = np.minimum(3 * n_object, n_total - n_object)
        background_ids = np.nonzero(y_temp == 0)
        choice = np.random.choice(background_ids, n_background)

        take = np.zeros_like(y_temp)
        take[object_ids or choice] = True

        X = np.concatenate(X, X_temp[take, :, :, :], axis=0)
        y = np.concatenate(y, y_temp[take])

        for gt, cl in zip(gts, cls):
            im_temp = cropAndResize(im, gt, size)
            X = np.concatenate(X, im_temp, axis=0)
            y = np.concatenate(y, cl)

        return dataset(X, y)
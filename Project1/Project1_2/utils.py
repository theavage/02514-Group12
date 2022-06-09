from tkinter import image_names
import torch
import cv2
import json
import csv
from PIL import Image, ExifTags
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from objectproposal import *


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
    def __init__(self, X, y, transform = transforms.ToTensor()):
         self.transform = transform
         self.data = X
         self.targets = y

    def __len__(self):
         return len(self.image_paths)

    def __getitem__(self, idx):
         return self.transform(self.data[idx, :, :, :]), self.transform(self.targets[idx])

def createDataSet(images, indices, classes, groundtruth, ids):
    X = np.empty((0, 224, 224, 3))
    y = np.empty((0, 0))
    for id in ids:
        mask = indices == id
        im_data = images[id]
        im = np.asarray(Image.open('/dtu/datasets1/02514/data_wastedetection' + '/' + im_data['file_name']))
        gts = groundtruth[mask, :]
        cls = classes[mask]
        cls = np.append(0, cls)

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
            X_temp = np.concatenate(X_temp, im_temp, axis=0)

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

def loadDataset():
    dataset_path = '/dtu/datasets1/02514/data_wastedetection/'
    anns_file_path = dataset_path + '/' + 'annotations.json'
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

        categories = dataset['categories']
        imgs = dataset['images']
        anns = dataset['annotations']
        img_ids = []
        bb = []
        y = []
        for annotation in anns: 
            img_ids.append(annotation['image_id'])
            bb.append(annotation['bbox'])
            y.append(annotation['category_id'])

    #Converting categories into supercategories
    super_cats = {'bg':0,'Aluminium foil':1, 'Battery':2 ,'Blister pack':3, 'Bottle':4,'Bottle cap':5,'Broken glass':6,'Can':7,'Carton':8,'Cup':9,'Food waste':10,'Glass jar':11,'Lid':12,'Other plastic':13,'Paper':14,'Paper bag':15,'Plastic bag & wrapper':16,'Plastic container':17,'Plastic glooves':18,'Plastic utensils':19,'Pop tab':20,'Rope & strings':21,'Scrap metal':22,'Shoe':23,'Squeezable tube':24,'Straw':25,'Styrofoam piece':26,'Unlabeled litter':27,'Cigarette':28}
    merged_y = []
    for i in y:
        for x in categories:
            if i == x['id']:
                merged_y.append(super_cats[x['supercategory']])
    y = merged_y

    return img_ids, bb, y, imgs

def loadImages():
    _,_,_,imgs = loadDataset()
    img_ids = list(range(0,1500))
    actual_images = []
    for id in img_ids:
        for img in imgs:
            if img['id'] == id:
                I = np.asarray(Image.open('/dtu/datasets1/02514/data_wastedetection' + '/' + img['file_name']))
                actual_images.append(I)

    return actual_images

def get_split_vector():
    img_ids = list(range(0,1500))
    np.random.shuffle(img_ids)
    train,val,test = img_ids[0:1050],img_ids[1050:1275],img_ids[1275:] #70,15,15 split:-)!!!
    return train,val,test

def dataset_split():
    img_ids, bb, y, _ = loadDataset()
    actual_images = loadImages()
    train,val,test = get_split_vector()
    X_train = []
    y_train = []
    bb_train = []
    for i in train:
        
        indices = [index for index, element in enumerate(img_ids) if element == i]
        print(indices)
        X_train.append(actual_images[i])
        for index in indices:
            y_train.append(y[index])
            bb_train.append(bb[index])

    X_val = []
    y_val = []
    bb_val = []
    for i in val:
        indices = [index for index, element in enumerate(img_ids) if element == i]
        X_val.append(actual_images[i])
        for index in indices:
            y_val.append(y[index])
            bb_val.append(bb[index])

    X_test = []
    y_test = []
    bb_test = []
    for i in test:
        indices = [index for index, element in enumerate(img_ids) if element == i]
        X_test.append(actual_images[i])
        for index in indices:
            y_test.append(y[index])
            bb_test.append(bb[index])        

    return X_train, y_train, bb_train, X_val, y_val, bb_val, X_test, y_test, bb_test
        
def save_data():
    _, y_train, bb_train, _, y_val, bb_val, _, y_test, bb_test = dataset_split()
    train_id,test_id,val_id = get_split_vector()
    
    bb_train = np.asarray(bb_train)
    y_train = np.asarray(y_train)
    bb_test = np.asarray(bb_test)
    y_test = np.asarray(y_test)
    bb_val = np.asarray(bb_val)
    y_val = np.asarray(y_val)
    train_id = np.asarray(train_id)
    test_id = np.asarray(test_id)
    val_id = np.asarray(val_id)

    with open('data.npy','wb') as f:
        np.save(f, bb_train)
        np.save(f, bb_test)
        np.save(f, bb_val)
        np.save(f, y_train)
        np.save(f, y_test)
        np.save(f, y_val)
        np.save(f, train_id)
        np.save(f, test_id)
        np.save(f, val_id)

def loadData():
    data = np.load('/zhome/df/9/164401/02514-Group12/Project1/Project1_2/data.npz', allow_pickle = True)
    return data['train_ids'], data['val_ids'], data['test_ids'], data['ids'], data['gt'], data['y'], data['ims']
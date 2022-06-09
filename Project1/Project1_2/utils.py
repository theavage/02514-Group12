from tkinter import image_names
import torch
import cv2
import json
import csv
from PIL import Image, ExifTags
import numpy as np
from sklearn.model_selection import train_test_split


def checkDevice():

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkDevice()

def cropAndResize(image, rectangle, size):

    x, y, w, h = rectangle
    x1, y1 = x, y
    x2, y2, = x + w, y + h
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, size)
    
    return resized

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

    return img_ids, bb, y, imgs, categories

img_ids, bb, y, imgs, categories = loadDataset()

def cat_to_supercat(y, categories):
    super_cats = {'bg':0,'Aluminium foil':1, 'Battery':2 ,'Blister pack':3, 'Bottle':4,'Bottle cap':5,'Broken glass':6,'Can':7,'Carton':8,'Cup':9,'Food waste':10,'Glass jar':11,'Lid':12,'Other plastic':13,'Paper':14,'Paper bag':15,'Plastic bag & wrapper':16,'Plastic container':17,'Plastic glooves':18,'Plastic utensils':19,'Pop tab':20,'Rope & strings':21,'Scrap metal':22,'Shoe':23,'Squeezable tube':24,'Straw':25,'Styrofoam piece':26,'Unlabeled litter':27,'Cigarette':28}
    merged_y = []
    for i in y:
        for x in categories:
            if i == x['id']:
                merged_y.append(super_cats[x['supercategory']])
    y = merged_y
    return y

y = cat_to_supercat(y,categories)

def loadImages(imgs):
    img_ids = list(range(0,1500))
    actual_images = []
    for id in img_ids:
        for img in imgs:
            if img['id'] == id:
                I = np.asarray(Image.open('/dtu/datasets1/02514/data_wastedetection' + '/' + img['file_name']))
                actual_images.append(I)

    return actual_images

actual_images = loadImages(imgs)

def get_split_vector():
    img_ids = list(range(0,1500))
    np.random.shuffle(img_ids)
    train,val,test = img_ids[0:1050],img_ids[1050:1275],img_ids[1275:] #70,15,15 split:-)!!!
    return train,val,test

train,val,test = get_split_vector()

def dataset_split(train, val, test,y,bb,actual_images,img_ids):
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
        

X_train, y_train, bb_train, X_val, y_val, bb_val, X_test, y_test, bb_test = dataset_split(train, val, test,y,bb,actual_images,img_ids)

bb_train = np.asarray(bb_train)
y_train = np.asarray(y_train)
bb_test = np.asarray(bb_test)
y_test = np.asarray(y_test)
bb_val = np.asarray(bb_val)
y_val = np.asarray(y_val)


np.save('train_data.npy', bb_train,y_train)
np.save('test_data.npy', bb_test,y_test)
np.save('val_data.npy', bb_val,y_val)

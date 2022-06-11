import torch
import json
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from objectproposal import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def checkDevice():

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop(image, rectangle):

    x, y, w, h = rectangle
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    cropped = image[y1:y2, x1:x2]
    
    return cropped

def transformBoundingBox(rects):
    x, y, w, h = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    x1, x2, y1, y2 = x, x + w, y, y + h
    x1, x2, y1, y2 = np.atleast_2d(x1).T, np.atleast_2d(x2).T, np.atleast_2d(y1).T, np.atleast_2d(y2).T
    return np.hstack([x1, y1, x2, y2])

class dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        size = (224, 224)
        self.transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.targets[idx]

class testDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        size = (224, 224)
        self.transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        self.data = X

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx])

def createDataSet(images, indices, classes, groundtruth, ids):
    X = []
    y = np.array([])
    for id in ids:
        mask = indices == id
        im_data = images[id]
        im = np.asarray(Image.open('/dtu/datasets1/02514/data_wastedetection' + '/' + im_data['file_name']))
        gts = groundtruth[mask, :]
        cls = classes[mask]

        im, gts = resize(im, gts)

        rects = edgeBoxDetection(im)
        rects, gts = transformBoundingBox(rects), transformBoundingBox(gts)
        rects, gts = torch.as_tensor(rects), torch.as_tensor(gts)
        iou = torchvision.ops.box_iou(rects, gts)
        iou = np.asarray(iou)

        gts_id = iou.argmax(axis=1)
        gts_scores = iou.max(axis=1)
        threshold = 0.5
        gts_mask = gts_scores > threshold
        y_temp = gts_mask * cls[gts_id]

        n_total = len(y_temp)
        n_object = np.count_nonzero(y_temp)
        object_ids = np.nonzero(y_temp)
        n_background = np.minimum(3 * n_object, n_total - n_object)
        background_ids = np.nonzero(y_temp == 0)
        choice = np.random.choice(background_ids[0], n_background)

        temp_mask = np.hstack([object_ids[0], choice])
        y = np.concatenate([y, y_temp[temp_mask]])
        rects_taken = rects[temp_mask, :]

        for rect in rects_taken:
            try:
                X.append(Image.fromarray(crop(im, rect)))
            except:
                continue

        for gt, cl in zip(np.asarray(gts).astype(int), cls):
            try:
                X.append(Image.fromarray(crop(im, gt)))
            except:
                continue
            y = np.append(y, cl)

        print(id)

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

def plot_graphs(out_dict, name):
    plt.figure()
    plt.plot(out_dict['train_loss'],'-o')
    plt.plot(out_dict['train_acc'],'-o')
    plt.legend(('Train error','Train accuracy'))
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.savefig(name)
    
    plt.figure()
    plt.plot(out_dict['test_loss'],'-o')
    plt.plot(out_dict['test_acc'],'-o')
    plt.legend(('Test error','Test accuracy'))
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.savefig(name)

def resize(image, bb, size = 1000):
    if image.shape[1] > image.shape[0]:
        size = (size, int(size * image.shape[0] / image.shape[1]))
    else:
        size = (int(size * image.shape[1] / image.shape[0]), size)

    img_resized = cv2.resize(image.copy(), size, interpolation = cv2.INTER_AREA)

    lx = size[0] / image.shape[1]
    ly = size[1] / image.shape[0]

    bb_new = bb * np.array([[lx, ly, lx, ly]])

    return img_resized, bb_new.astype(int)

from utils import *

train_ids, val_ids, test_ids, indices, groundtruth, classes, images = loadData()

createDataSet(images, indices, classes, groundtruth, train_ids, 'train')
createDataSet(images, indices, classes, groundtruth, test_ids, 'val')
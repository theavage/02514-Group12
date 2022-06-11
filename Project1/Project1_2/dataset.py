from utils import *

train_ids, val_ids, _, indices, groundtruth, classes, images = loadData()

#trainset = createDataSet(images, indices, classes, groundtruth, train_ids)
valset = createDataSet(images, indices, classes, groundtruth, val_ids)

#trainsetpath = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/trainset.pt'
#torch.save(trainset, trainsetpath)
valsetpath = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/valset.pt'
torch.save(valset, valsetpath)

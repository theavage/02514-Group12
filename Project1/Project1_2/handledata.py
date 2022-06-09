from utils import *

train_ids, val_ids, test_ids = get_split_vector()
ids, gt, y, ims = loadDataset()
np.savez('data', train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, ids=ids, gt=gt, y=y, ims=ims)
import cv2
import numpy as np
from objectproposal import *
from PIL import Image

#im = cv2.imread('/dtu/datasets1/02514/data_wastedetection/batch_1/000000.jpg')
im = Image.open('/dtu/datasets1/02514/data_wastedetection/batch_1/000000.jpg')
im = im.crop((0, 0, 100, 100))
im = im.resize((224, 224), Image.NEAREST)
print(np.asarray(im))
#rects = createObjectProposals(im)
#print(rects.shape)
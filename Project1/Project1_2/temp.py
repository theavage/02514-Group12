import cv2
from objectproposal import *

im = cv2.imread('/dtu/datasets1/02514/data_wastedetection/batch_1/000000.jpg')
print(im.shape)
rects = createObjectProposals(im)
print(rects.shape)
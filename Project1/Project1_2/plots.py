import matplotlib.pyplot as plt
import cv2

from objectproposal import *
from utils import *

im = cv2.imread('/dtu/datasets1/02514/data_wastedetection/batch_1/000005.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
rects = edgeBoxDetection(im, 500)
for rect in transformBoundingBox(rects):
    im = cv2.rectangle(im, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

cv2.imwrite('box_plot.jpeg', im)
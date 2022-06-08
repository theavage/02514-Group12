import cv2
import numpy as np
import cv2
import torch
import torchvision

from utils import *
from model import *
from trainmodel import *
from objectproposal import *

test_images = 0
ground_truth_boxes = 0
path = ''
model = torch.load(path)

for im in test_images:
    rectangles = createObjectProposals(im)
    classes = []
    
    for i, rectangle in enumerate(rectangles):
        input = cropAndResize(im, rectangle, (224, 224))
        input = torch.as_tensor(input)
        output = model(input)
        _, predicted = torch.max(output, 1)
        classes.append(predicted)

    for gt in ground_truth_boxes:
        rectangles, gt = torch.as_tensor(rectangles), torch.as_tensor(gt)
        iou = torchvision.ops.box_iou(gt, rectangles)
        index = torchvision.ops.nms(rectangles, iou, 0.5)
        rectangles, classes, index = np.asarray(rectangles), np.asarray(classes), np.asarray(index)
        detected_box = rectangles[index, :]
        detetcted_class = classes[index]
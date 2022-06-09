import numpy as np
import torch
import torchvision

from utils import *
from model import *
from objectproposal import *

test_images = 0
test_image_ids = 0
test_classes, test_rectangles = np.empty(0), np.empty(0)

path = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/model.pt'
model = createModel()
model.load_state_dict(torch.load(path))

for id, im in zip(test_image_ids, test_images):
    rectangles = createObjectProposals(im)
    classes, scores = np.empty(0), np.empty(0)
    
    for rect in rectangles:
        size = (224, 224)
        input = cropAndResize(im, rect, size)
        input = torch.as_tensor(input)
        output = model(input)
        s, c = torch.max(output, 1)
        classes = np.append(classes, c)
        scores = np.append(scores, s)

    object_indices, _ = np.nonzero(classes)
    object_classes = classes[object_indices]
    object_rectangles = rectangles[object_indices, :]
    object_scores = scores[object_indices]

    final_classes, final_rectangles = np.empty(0), np.empty(0)

    for i in set(object_classes):
        temp_classes = object_classes[i == object_classes]
        temp_scores = object_scores[i == object_classes]
        temp_rectangles = object_rectangles[i == object_classes, :]

        temp_rectangles, temp_scores = torch.as_tensor(temp_rectangles), torch.as_tensor(temp_scores)
        final_indices = torchvision.ops.nms(temp_rectangles, temp_scores, 0)
        final_indices = np.asarray(final_indices)
        final_classes = np.append(final_classes, temp_classes[final_indices])
        final_rectangles = np.append(final_rectangles, temp_rectangles[final_indices])
    
    for c, r in zip(final_classes, final_rectangles):
        test_classes = np.append(test_classes, c)
        test_rectangles = np.append(test_rectangles, r, axis=0)
        test_ids = np.append(test_ids, id)
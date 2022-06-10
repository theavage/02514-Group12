import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from utils import *
from model import *
from objectproposal import *

_, _, test_ids, indices, groundtruth, classes, images = loadData()

test_classes, test_rectangles = np.empty(0), np.empty(0)

path = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/model.pt'
model = createModel()
model.load_state_dict(torch.load(path))

device = checkDevice()
model.to(device)
model.eval()

for id, im_data in zip(test_ids, images):
    im = Image.open('/dtu/datasets1/02514/data_wastedetection' + '/' + im_data['file_name'])
    rects = edgeBoxDetection(im)
    classes, scores = np.array([]), np.array([])
    
    input = torch.empty((0, 3, 224, 224))
    proposal_images = []
    for rect in transformBoundingBox(rects):
        l, t, r, b = rect
        cropped = torchvision.transforms.functional.crop(im, l, t, r, b)
        proposal_images.append(cropped)
    
    dataset = testDataset(proposal_images)
    dataloader = DataLoader(dataset, batch_size=16)
    for input in dataloader:
        input.to(device)
        output = model(input)
        s, c = torch.max(output, 1)
        classes = np.append(classes, c.detach().numpy())
        scores = np.append(scores, s.detach().numpy())

    object_indices = np.nonzero(classes)
    object_indices = object_indices[0]
    object_classes = classes[object_indices]
    object_rectangles = rects[object_indices, :]
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
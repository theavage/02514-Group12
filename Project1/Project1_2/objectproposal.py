import cv2
import numpy as np

def createObjectProposals(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    # ss.switchToSingleStrategy()
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    rectangles = ss.process()

    return rectangles

def edgeBoxDetection(image):
    model = '/zhome/df/9/164401/02514-Group12/Project1/Project1_2/model.yml'
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    edges = edge_detection.detectEdges(np.float32(image) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(5000)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    
    return boxes[0]

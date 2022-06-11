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

def edgeBoxDetection(image, n):
    model = 'model.yml'
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    edges = edge_detection.detectEdges(np.float32(image) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(n)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    
    return boxes[0]

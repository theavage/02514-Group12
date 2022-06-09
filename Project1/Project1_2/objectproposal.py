import cv2

def createObjectProposals(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSingleStrategy()
    # ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    rectangles = ss.process()

    return rectangles

def edgeBoxDetection(image):
    db = cv2.ximgproc.createStructuredEdgeDetection()
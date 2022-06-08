import cv2
from cv2.ximgproc.segmentation import createSelectiveSearchSegmentation


def createObjectProposals(image):

    # Propose object rectangles in an image by Selective Search

    shape = (100, 100)
    resized = cv2.resize(image, shape)

    ss = createSelectiveSearchSegmentation()
    ss.setBaseImage(resized)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    rectangles = ss.process()

    return rectangles


def plotProposals(image, rectangles, n):

    # Plot n proposed rectangles in an image
    # All rectangles are plotted by n = -1

    if n == -1: n = rectangles.shape[1]

    image_rectangles = image.copy()
    for i, rectangle in enumerate(rectangles):
        if i < n:
            x, y, w, h = rectangle
            image_rectangles = cv2.rectangle(image_rectangles, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Object proposals", image_rectangles)













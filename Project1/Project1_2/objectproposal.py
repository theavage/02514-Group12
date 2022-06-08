import cv2

def createObjectProposals(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    rectangles = ss.process()

    return rectangles

def plotProposals(image, rectangles, n):

    image_rectangles = image.copy()
    for i, rectangle in enumerate(rectangles):
        if i < n:
            x, y, w, h = rectangle
            image_rectangles = cv2.rectangle(image_rectangles, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Object proposals", image_rectangles)
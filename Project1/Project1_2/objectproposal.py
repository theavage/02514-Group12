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

def cropAndResize(image, rectangle, size):

    x, y, w, h = rectangle
    x1, y1 = x, y
    x2, y2, = x + w, y + h
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, size)
    
    return resized


image = cv2.imread('testimage.jpeg')
rects = createObjectProposals(image)
print(rects)
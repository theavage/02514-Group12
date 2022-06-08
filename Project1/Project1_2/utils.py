import torch
import cv2

def checkDevice():

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cropAndResize(image, rectangle, size):

    x, y, w, h = rectangle
    x1, y1 = x, y
    x2, y2, = x + w, y + h
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, size)
    
    return resized
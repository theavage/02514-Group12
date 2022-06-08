import torch
import cv2

def checkDevice():

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
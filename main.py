from os import listdir
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 

def load_images(load_dir:str):

    image_list = listdir(load_dir)
    image_list.sort()
    print(image_list)

    images = []
    return images

load_images(load_dir="dogs-vs-cats/train_subset/")
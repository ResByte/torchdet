from typing import Callable, List
import pandas as pd
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WheatDataset(Dataset):
    """This is a sample dataset class, modify according to the new dataset at hand"""
    def __init__(self,
                 img_root: str,
                 data_list: List,
                 img_transforms: transforms = None,
                 is_train: bool = True
                 ):
        # here data list is a list of dicts, each containing img info and labels
        self.data_list = data_list
        self.img_root = img_root
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        filename = self.data_list[idx][0]
        
        # extract key info from data list 
        width = self.data_list[idx][1]['width']
        height = self.data_list[idx][1]['height']
        _bboxes = np.array(self.data_list[idx][1]['bbox'])
        source = self.data_list[idx][1]['source'] # source value is specific to this dataset 
        
        image = Image.open(os.path.join(
            self.img_root, filename+'.jpg')).convert('RGB')
        image = np.asarray(image)
        
        category_id_to_name = {1:'wheat'}
        annotations = {
            'image' : image, 
            'bboxes' : _bboxes,
            'category_id' : np.ones(len(_bboxes), dtype=np.int)
        }
        
        # augment both image and bboxes 
        if self.img_transforms is not None:
            augmented = self.img_transforms(**annotations)
            image = augmented['image']
            _bboxes = augmented['bboxes']
            
        bboxes = []
        for i in range(len(_bboxes)):
            x_min, y_min, w, h = _bboxes[i]
            x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
            bboxes.append([x_min, y_min, x_max, y_max])
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # filter out small boxes
        area = (bboxes[:,3]-bboxes[:,1]) * (bboxes[:,2]-bboxes[:,0])
        bboxes = bboxes[area.gt(100.)]
        
        labels = torch.ones((len(bboxes)), dtype=torch.int64)
        image_id = torch.tensor([idx])        
        iscrowd = torch.zeros((len(bboxes)), dtype=torch.int64)
        
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        #target["masks"] = None
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return image, target

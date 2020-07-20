"""
Training script that loads dataset stored in local 
and writes model at every epoch. Config file can be 
obtained at config.py. 
"""
# general imports
from typing import Callable, List, Tuple
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import seaborn as sns
sns.set()

import cv2
import os  
from PIL import Image 
from pprint import pprint
from collections import OrderedDict
import random 
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# torch and torchvision
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from config import config
from augment import get_transforms
from dataset import WheatDataset
from model import create_model, create_model_with_backbone
from catalyst import dl
from trainer import Trainer


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # set the following true for inference otherwise training speed will be slow
    torch.backends.cudnn.deterministic = False
    
def get_merged_data(df):
    merged_dict = {}
    for idx,row in tqdm(df.iterrows(), total=len(df)):
        filename = row['image_id']
        width = row['width']
        height = row['height']
        bbox = np.asarray(row['bbox'], dtype=np.float)
        source = row['source']

        if filename in merged_dict:
            merged_dict[filename]['bbox'].append(bbox)
        else:
            merged_dict[filename] = {
                'width' : width,
                'height' : height,
                'bbox' : [bbox],
                'source':source
            }
    return merged_dict

def collate_fn(batch):
    return tuple(zip(*batch))


class CustomRunner(dl.Runner):
    """Using catalyst framework for training, there are some bugs, does not train properly"""
    def _handle_batch(self, batch):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
        bs = len(images)
        loss_dict = self.model(images, targets)
        loss_cls = loss_dict['loss_classifier'].cpu().item()
        loss_bbox = loss_dict['loss_box_reg'].cpu().item()
        losses_reduced = sum(loss for loss in loss_dict.values())

        loss_value = losses_reduced.item()

        self.batch_metrics.update({
            "loss_cls":loss_cls,
            "loss_bbox":loss_bbox,
            "loss":losses_reduced
        })

        if self.is_train_loader:
            losses_reduced.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

def main():
    # setup config
    cfg = config()
    cfg['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cfg['logdir'] += timestr
    set_global_seed(cfg['random_state'])
    pprint(cfg)
    
    # load csv
    train_df = pd.read_csv(cfg['train_csv_path'])
    train_df['bbox'] = train_df['bbox'].apply(lambda x: 
                           np.fromstring(
                               x.replace('[','')
                                .replace(']','')
                                .replace(',',' '), sep=' '))
    merged_dict = get_merged_data(train_df)
    
    train_list = list(merged_dict.items())
    
    # transforms
    train_transforms, test_transforms = get_transforms(cfg['image_size'])
    # dataset and data loader 
    train_dataset = WheatDataset(cfg['train_img_root'], train_list, train_transforms)
    train_loader = DataLoader(train_dataset, cfg['batch_size'], shuffle=True,
                              num_workers=4,drop_last=True, collate_fn=collate_fn)
    
    # model 
    fasterrcnn = create_model()
#     fasterrcnn = create_model_with_backbone(arch='resnet34', pretrained=True, num_classes=cfg['num_classes'])
    
    # optimizer
    params = [p for p in fasterrcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'],  weight_decay=1e-5)

    # trainer 
    runner = Trainer(
                model=fasterrcnn, 
                criterion=None,
                optimizer=optimizer, 
                train_loader=train_loader, 
                test_loader=None, 
                logdir=cfg['logdir'],
                epochs=cfg['num_epochs'],
                device=cfg['device']
    )
    epoch_loss = 0.
    count = 0
    for epoch in range(cfg['num_epochs']):
        try:
            epoch_loss, count = runner.train(epoch,count,30)
            print(f"Epoch: {epoch} Loss: {epoch_loss}")
            runner.save_checkpoint(epoch, epoch_loss, cfg['logdir'], filename='model.pth')
        except KeyboardInterrupt:
            print(f"Exiting: last epoch:{epoch-1} with loss {epoch_loss}" )
            break
        
    

    
if __name__ == "__main__":
    main()
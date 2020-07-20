import torch 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from collections import deque 
import time
import os

class AverageMeter(object):
    """Logs average statistics and is used for tracking metrics"""
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = "{avg:.4f} ({global_avg:.4f})"
        
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n 
        self.total += value * n
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property 
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            avg=self.avg,
            global_acg = self.global_avg, 
        )


    

class Trainer():
    """Generic trainer class to create model training and testing api
    """
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 train_loader, 
                 test_loader, 
                 logdir,
                 epochs,
                 device
                ):
        self.model = model.to(device) 
        if criterion is not None:
            self.criterion = criterion.to(device) 
        self.optimizer = optimizer 
        self.epochs = epochs
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.device = device
        self.logdir = logdir 
        self.writer = SummaryWriter(self.logdir)
            
    def adjust_lr(self, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch // 20))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train(self,epoch,count=0,freq=50):
        """Training loop for a given epoch"""
        
        steps = len(self.train_loader)
        loss_cls_log = AverageMeter()
        loss_bbox_log = AverageMeter()
        loss_log = AverageMeter()
        batch_time = AverageMeter()
        
        # switch to train mode
        self.model.train()
        end = time.time()
        for idx, batch in enumerate(self.train_loader):
            
            images, targets = batch
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
            bs = len(images)
            loss_dict = self.model(images, targets)
            loss_cls = loss_dict['loss_classifier'].cpu().item()
            loss_bbox = loss_dict['loss_box_reg'].cpu().item()
            losses_reduced = sum(loss for loss in loss_dict.values())

            loss_value = losses_reduced.item()

            loss_cls_log.update(loss_cls, bs)
            loss_bbox_log.update(loss_bbox, bs)
            loss_log.update(loss_value, bs)


            self.optimizer.zero_grad()
            losses_reduced.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end, bs)
            end = time.time()
            
            self.writer.add_scalar('Train/batch/loss', loss_log.value, count)
            self.writer.add_scalar('Train/batch/loss_cls', loss_cls_log.value, count)
            self.writer.add_scalar('Train/batch/loss_bbox', loss_bbox_log.value, count)
            count+=1
            if idx % freq == 0:
                print(f"Train Epoch:{epoch} Batch:{idx}/{steps} loss:{loss_log.avg:.4f}",
                  f"loss_cls:{loss_cls_log.avg:.4f} loss_bbox:{loss_bbox_log.avg:.4f}")
#             if count > 100:
#                 break
        self.writer.add_scalar('Train/epoch/loss', loss_log.avg, epoch)        
        return loss_log.global_avg, count
    
    def save_checkpoint(self,epoch, loss, file_path, filename='model.pth'):
        """saves model at given epoch
        Args:
            epoch (int): iteration number over dataset 
            loss (float): loss value for the current epoch 
            file_path (string): path to store the model file 
            filename (string,optional): defualt filename for storage
        """
        state = {'epoch': epoch + 1,
                 'state_dict': self.model.state_dict(),
                 'best_performance': loss,
                 'optimizer' : self.optimizer.state_dict(),
                } 
        torch.save(state, os.path.join(file_path, 'epoch_'+str(state['epoch'])+ '_'+ filename))
        print(f"Model at epoch:{epoch} is saved at {os.path.join(file_path, 'epoch_'+str(state['epoch'])+ '_'+ filename)}")
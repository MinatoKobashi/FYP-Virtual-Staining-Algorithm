import matplotlib.pyplot as plt

import numpy as np
import random
import os
import glob
import torch
import torchvision.transforms as transforms
import time
import datetime
import sys

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom(port=8097)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.losses_save = {}
        self.loss_windows = {}
        self.image_windows = {}
        
    def get_loss(self,loss_name):
        return self.losses_save[loss_name]
        
    def log(self, losses = None, images = None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()
        
        if (self.batch % 100) == 0:
            sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- '  % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
            
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name].item()
                else:
                    self.losses[loss_name] += losses[loss_name].item()
                if (i+1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))
                
            batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
            batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
            sys.stdout.write('ETA: %s\n' % (datetime.timedelta(seconds = batches_left*self.mean_period/batches_done)))
        
            # Draw images
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts = {'title':image_name})
                else:
                    self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})
        
        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            #plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), opts={'xlabel': 'epochs','ylabel':loss_name,'title':loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name],update='append')
                # Reset losses for ext epoch
                self.losses_save[loss_name] = loss/self.batch
                self.losses[loss_name] = 0.0
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class ImageDataset(Dataset):
    def __init__(self,root,sample,transforms_=None, unaligned = False, mode = 'Train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/%s/Unstained' % (mode, sample)) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/%s/Stained' % (mode, sample)) + '/*.*'))
        
    def __getitem__(self,index):
    
        A=Image.open(self.files_A[index%len(self.files_A)])
        item_A = self.transform(A.convert("RGB"))
        if self.unaligned:
            B = Image.open(self.files_B[random.randint(0,len(self.files_B)-1)])
            item_B = self.transform(B.convert("RGB"))
        else:
            B = Image.open(self.files_B[index % len(self.files_B)])
            item_B = self.transform(B.convert("RGB"))
        
        return {'A': item_A, 'B': item_B}
        
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class LambdaLR():
    def __init__(self,n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch)>0), "Decay must start before the training ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0,epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def add_loss_graph_tensorboard(loss, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        

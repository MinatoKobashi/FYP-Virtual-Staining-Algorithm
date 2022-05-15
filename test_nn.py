import argparse
import os
import sys

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from utils import ImageDataset
from cycleGan_models import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help = 'use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netF_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: you have a CUDA device, so you should probably run with --cuda")
if opt.cuda:
    device = 'cuda'
else:
    device = 'cpu'

netF_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

# Load state dicts
netF_A2B.load_state_dict(torch.load(opt.generator_A2B,map_location=torch.device(device)))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A,map_location=torch.device(device)))

# Set model's test mode
netF_A2B.eval()
netG_B2A.eval()

# Inputs memory allocations
input_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Data Loader
transforms_ = [   transforms.RandomCrop(opt.size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  ]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

# Testing
# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')

for i, batch in enumerate(dataloader):
    real_A = input_A.copy_(batch['A'])
    real_B = input_B.copy_(batch['B'])
    
    fake_B = (netF_A2B(real_A).data
    fake_A = (netG_B2A(real_B).data
    
    save_image(fake_A, 'output/A/%04d.png' % (i+1))
    save_image(fake_B, 'output/B/%04d.png' % (i+1))
    
    sys.stdout.write('\r Generated images %04d of %04d' % (i+1, len(dataloader)))
    
sys.stdout.write('\n')

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import datetime

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from PIL import Image
from random import randrange

from cycleGan_models import ResBlock
from cycleGan_models import Generator
from cycleGan_models import Discriminator
from utils import *

if __name__ == '__main__':
    # Train parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=5, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./Dataset', help='root directory of the dataset')
    parser.add_argument('--output_file', type=str, default='./run', help='directory for saving model results')
    parser.add_argument('--sample', type=str, default='Brain', help='type of dataset sample')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='lambda multiplier for cycle consistency loss of generator A-B-A')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='lambda multiplier for cycle consistency loss of generator B-A-B')
    parser.add_argument('--lambda_ID', type=float, default=10.0, help='lambda multiplier for identity losses')
    parser.add_argument('--lambda_GAN', type=float, default=1.0, help='lambda multiplier for identity losses')
    parser.add_argument('--lambda_content', type=float, default=0.0, help='lambda multiplier for content losses')
    parser.add_argument('--threshold_A', type=float, default=70.0, help='threshold for image A for content losses')
    parser.add_argument('--threshold_B', type=float, default=220.0, help='threshold for image B for content losses')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action = 'store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=6, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)
    # print(torch.cuda.is_available())
    # if torch.cuda.is_available() and not opt.cuda:
    #     print("WARNING: You  have a CUDA device, so you should probably run with --cuda")

    # Networks
    netF_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_B = Discriminator(opt.input_nc)
    netD_A = Discriminator(opt.output_nc)

    netF_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    if opt.cuda:
        netF_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cyc = torch.nn.L1Loss()
    criterion_id = torch.nn.L1Loss()

    # Optimizers and L# schedulers
    optimizer_F = torch.optim.Adam(netF_A2B.parameters(), lr=opt.lr)
    optimizer_G = torch.optim.Adam(netG_B2A.parameters(), lr=opt.lr)
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr/10)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr/10)

    lr_scheduler_F = torch.optim.lr_scheduler.LambdaLR(optimizer_F,lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)

    # Inputs and Targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False).view(-1,1)
    target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False).view(-1,1)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset Loader
    transforms_ = [ transforms.Resize(int(opt.size), transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  ]
                    
    transforms_val = [  transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  ]

    dataloader = DataLoader(ImageDataset(opt.dataroot, opt.sample, transforms_=transforms_, unaligned=True), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    val_dataloader = DataLoader(ImageDataset(opt.dataroot, opt.sample, transforms_=transforms_val, mode='Validation'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    # Make Validation dir and single image data
    if not os.path.exists(opt.output_file):
        os.makedirs(opt.output_file)

    if not os.path.exists(opt.output_file+'/val_img'):
        os.makedirs(opt.output_file+'/val_img')

    if not os.path.exists(opt.output_file+'/losses'):
        os.makedirs(opt.output_file+'/losses')

    
    if not os.path.exists(opt.output_file+'/model'):
        os.makedirs(opt.output_file+'/model')

    opt_file = open(opt.output_file+'/settings.csv','w')
    opt_file.write(str(opt))
    opt_file.close()

    vloss_file = open(opt.output_file+'/losses/val_loss.csv','w')
    vloss_file.write("Epoch,val_loss\n")
    vloss_file.close()
    idloss_file = open(opt.output_file+'/losses/ID_loss.csv','w')
    idloss_file.write("Epoch,id_loss\n")
    idloss_file.close()
    ganloss_file = open(opt.output_file+'/losses/GAN_loss.csv','w')
    ganloss_file.write("Epoch,GAN_loss\n")
    ganloss_file.close()
    totloss_file = open(opt.output_file+'/losses/tot_loss.csv','w')
    totloss_file.write("Epoch,tot_loss\n")
    totloss_file.close()
    dloss_file = open(opt.output_file+'/losses/D_loss.csv','w')
    dloss_file.write("Epoch,D_loss\n")
    dloss_file.close()
    cycloss_file = open(opt.output_file+'/losses/cyc_loss.csv','w')
    cycloss_file.write("Epoch,cyc_loss\n")
    cycloss_file.close()

    # cycle loss Weight Decay
    # cyc_decay = LambdaLR(epochs, epoch, epochs//2)
    # Loss Plot
    # logger = Logger(opt.epochs, len(dataloader))

    val_tot = 0

    #####################
    ##### Training ######
    start = datetime.datetime.now()
    print('Start Training at Time: %s'  % start.strftime("%H:%M:%S"))
    for epoch in range(opt.start_epoch, opt.epochs+1):
        tloss_G_identity = 0
        tloss_G_GAN = 0
        tloss_FG_Cycle = 0
        tloss_D = 0
        ttot_loss = 0
        for i, batch in enumerate(dataloader):
            # Set model input
            #input_A, input_B = batch
            real_A = input_A.copy_(batch['A'])
            real_B = input_B.copy_(batch['B'])
            
            # Initialise  Generators F and B
            optimizer_F.zero_grad()
            optimizer_G.zero_grad()
            
            # Identity Loss
            if opt.lambda_ID != 0:
                # FA2B ID Loss
                same_B = netF_A2B(real_B)
                loss_ID_B = criterion_id(same_B,real_B)*opt.lambda_ID
                # GB2A ID Loss
                same_A = netG_B2A(real_A)
                loss_ID_A = criterion_id(same_A,real_A)*opt.lambda_ID
            else:
                loss_ID_B = torch.tensor(0)
                loss_ID_A = torch.tensor(0)
                    
            # GAN Loss
            fake_B = netF_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*opt.lambda_GAN

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*opt.lambda_GAN
            
            # Cycle Loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cyc(recovered_A, real_A)*opt.lambda_A
            
            recovered_B = netF_A2B(fake_A)
            loss_cycle_BAB = criterion_cyc(recovered_B, real_B)*opt.lambda_B
                    
            # # Content Loss
            if opt.lambda_content != 0:
                L1_function = torch.nn.L1Loss()
                real_A_mean = torch.mean(real_A,dim=1,keepdim=True)
                real_B_mean = torch.mean(real_B,dim=1,keepdim=True)
                fake_A_mean = torch.mean(fake_A,dim=1,keepdim=True)
                fake_B_mean = torch.mean(fake_B,dim=1,keepdim=True)

                real_A_normal = (real_A_mean - (opt.threshold_A/127.5-1))*100
                real_B_normal = (real_B_mean - (opt.threshold_B/3/127.5-1))*100

                fake_A_normal = (fake_A_mean - (opt.threshold_A/127.5-1))*100
                fake_B_normal = (fake_B_mean - (opt.threshold_B/3/127.5-1))*100

                real_A_sigmoid = torch.sigmoid(real_A_normal)
                real_B_sigmoid = 1 - torch.sigmoid(real_B_normal)

                fake_A_sigmoid = torch.sigmoid(fake_A_normal)
                fake_B_sigmoid = 1 - torch.sigmoid(fake_B_normal)

                content_loss_A = L1_function( real_A_sigmoid , fake_B_sigmoid )
                content_loss_B = L1_function( fake_A_sigmoid , real_B_sigmoid )

                content_loss_rate = 50*np.exp(-(i/len(dataloader)))
                content_loss = (content_loss_A + content_loss_B)*content_loss_rate*opt.lambda_content
            else:
                content_loss = torch.tensor(0)

            # Total Loss
            loss = loss_ID_A + loss_ID_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + content_loss
            loss.backward()
            
            optimizer_F.step()
            optimizer_G.step()
            
            #################################
            
            # Discriminator A
            optimizer_D_A.zero_grad()
            
            # Real Loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake Loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            
            # Total Loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            
            optimizer_D_A.step()
            
            #################################

            # Discriminator B
            optimizer_D_B.zero_grad()
            
            # Real Loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake Loss
            fake_B = fake_A_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            
            # Total Loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            
            optimizer_D_B.step()
            
            with torch.no_grad():
                if i == (len(dataloader)-1):
                    # Validation
                    val_tot = 0
                    for i, batch in enumerate(val_dataloader):
                        val_real_A = input_A.copy_(batch['A'])
                        val_real_B = input_B.copy_(batch['B'])
                        
                        # GAN Loss
                        fake_B = netF_A2B(val_real_A)
                        pred_fake = netD_B(fake_B)
                        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                        fake_A = netG_B2A(val_real_B)
                        pred_fake = netD_A(fake_A)
                        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
                        
                        # Total validation loss
                        val_loss = loss_GAN_A2B + loss_GAN_B2A
                        val_tot += val_loss
                                
                    # store data loss
                    vloss_file = open(opt.output_file+'/losses/val_loss.csv','a')
                    vloss_file.write(str(epoch) + "," + str(val_tot.item()/len(val_dataloader)) + "\n")
                    vloss_file.close()
                    idloss_file = open(opt.output_file+'/losses/ID_loss.csv','a')
                    idloss_file.write(str(epoch) + "," + str(tloss_G_identity.item()/len(dataloader)) + "\n")
                    idloss_file.close()
                    ganloss_file = open(opt.output_file+'/losses/GAN_loss.csv','a')
                    ganloss_file.write(str(epoch) + "," + str(tloss_G_GAN.item()/len(dataloader)) + "\n")
                    ganloss_file.close()
                    totloss_file = open(opt.output_file+'/losses/tot_loss.csv','a')
                    totloss_file.write(str(epoch) + "," + str(ttot_loss.item()/len(dataloader)) + "\n")
                    totloss_file.close()
                    dloss_file = open(opt.output_file+'/losses/D_loss.csv','a')
                    dloss_file.write(str(epoch) + "," + str(tloss_D.item()/len(dataloader)) + "\n")
                    dloss_file.close()
                    cycloss_file = open(opt.output_file+'/losses/cyc_loss.csv','a')
                    cycloss_file.write(str(epoch) + "," + str(tloss_FG_Cycle.item()/len(dataloader)) + "\n")
                    cycloss_file.close()
                    """
                    # Progress Report (http:locolhost:8097)
                    logger.log( {'loss_G': loss, 'loss_G_identity': (loss_ID_A + loss_ID_B),
                                    'loss_G_GAN': (loss_GAN_A2B+loss_GAN_B2A),
                                    'loss_FG_Cycle': (loss_cycle_ABA+loss_cycle_BAB),
                                    'loss_D': (loss_D_A+loss_D_B),
                                    'val_loss': (val_tot*len(dataloader)/len(val_dataloader))},
                                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B} )
                else:
                    # Progress Report (http:locolhost:8097)
                    logger.log( {'loss_G': loss, 'loss_G_identity': (loss_ID_A + loss_ID_B),
                                    'loss_G_GAN': (loss_GAN_A2B+loss_GAN_B2A),
                                    'loss_FG_Cycle': (loss_cycle_ABA+loss_cycle_BAB),
                                    'loss_D': (loss_D_A+loss_D_B)},
                                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B} )
                """ 
                else:
                    ttot_loss += loss
                    tloss_G_identity += loss_ID_A + loss_ID_B
                    tloss_G_GAN += loss_GAN_A2B+loss_GAN_B2A
                    tloss_FG_Cycle += loss_cycle_ABA+loss_cycle_BAB
                    tloss_D += loss_D_A + loss_D_B

        # Update learning rates
        lr_scheduler_F.step()
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        if ((epoch % 10) == 0) or (epoch == opt.epochs):
            sys.stdout.write('\rEpoch %03d/%03d -- Total loss: %.4f -- Validation loss: %.4f -- Time Elapsed: %s\n'  % (epoch, opt.epochs, ttot_loss, val_loss, str(datetime.datetime.now() - start)))
            torch.save(netF_A2B.state_dict(), opt.output_file+'/model/netF_A2B.pth')
            torch.save(netG_B2A.state_dict(), opt.output_file+'/model/netG_B2A.pth')
            torch.save(netD_A.state_dict(), opt.output_file+'/model/netD_A.pth')
            torch.save(netD_B.state_dict(), opt.output_file+'/model/netD_B.pth')
            # with torch.no_grad():
            #     batch = next(iter(val_dataloader))
            #     val_real_A = input_A.copy_(batch['A'])
            #     val_real_B = input_B.copy_(batch['B'])
            #     fake_B = netF_A2B(val_real_A)
            #     fake_A = netG_B2A(val_real_B)
            #     imgcon = torch.cat((val_real_A,fake_B,val_real_B,fake_A), 2)
            #     # print(imgcon.size())
            #     save_image(imgcon.data, opt.output_file+'/val_img/epoch_%04d.png' % (epoch))  
    
    plots = ['val_loss','ID_loss','GAN_loss','tot_loss','D_loss','cyc_loss']

    fig, axs = plt.subplots(2,3)

    for i in range(2):
        for k in range(3):
            df = pd.read_csv('run11/losses/%s.csv' % plots[i*3+k]) 
            # print(df.head())
            df.plot(x=0, y=1, ax=axs[i,k], figsize = (10,3), title = plots[i*3+k], xlim = (1), legend = False, sharex = True)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.5)

    plt.savefig('test_graphs.jpeg', bbox_inches='tight')


        
    print("~~~~~~~~~~~~TRAINING COMPLETE~~~~~~~~~~~~")

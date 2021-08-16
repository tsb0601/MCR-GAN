import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from augmentloader import AugmentLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torchvision.utils as vutils
import train_func as tf
from loss import MaximalCodingRateReduction
import utils



parser = argparse.ArgumentParser(description='Supervised Learning')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--fd', type=int, default=128,
                    help='dimension of feature dimension (default: 128)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--epo', type=int, default=800,
                    help='number of epochs for training (default: 800)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gam1', type=float, default=1.,
                    help='gamma1 for tuning empirical loss (default: 1.)')
parser.add_argument('--gam2', type=float, default=1.,
                    help='gamma2 for tuning empirical loss (default: 1.)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared (default: 0.5)')
parser.add_argument('--corrupt', type=str, default="default",
                    help='corruption mode. See corrupt.py for details. (default: default)')
parser.add_argument('--lcr', type=float, default=0.,
                    help='label corruption ratio (default: 0)')
parser.add_argument('--lcs', type=int, default=10,
                    help='label corruption seed for index randomization (default: 10)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
args = parser.parse_args()



### Parameters
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 100
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 128
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 30
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


import torchvision.transforms.functional as F
def show(imgs, epoch):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #plt.show()
    plt.savefig(str(epoch)+"mcr"+".png")
    #plt.close()



def main():


    ## Pipelines Setup
    model_dir = os.path.join(args.save_dir,
                   'sup_{}+{}_{}_epo{}_bs{}_lr{}_mom{}_wd{}_gam1{}_gam2{}_eps{}_lcr{}{}'.format(
                        args.arch, args.fd, args.data, args.epo, args.bs, args.lr, args.mom,
                        args.wd, args.gam1, args.gam2, args.eps, args.lcr, args.tail))

    if args.pretrain_dir is not None:
        net, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
        #utils.update_params(model_dir, args.pretrain_dir)
    else:
        net = tf.load_architectures(args.arch, args.fd)

    transforms = tf.load_transforms(args.transform)
    trainset = tf.load_trainset(args.data, transforms, path=args.data_dir)
    trainset = tf.corrupt_labels(args.corrupt)(trainset, args.lcr, args.lcs)
    trainloader = DataLoader(trainset, batch_size=args.bs, drop_last=True, num_workers=4)


    criterionF = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)
    #utils.save_params(model_dir, vars(args))

    #print("I have begun to train!")
    ## Training
    device = torch.device("cuda")

    criterionG = nn.MSELoss()
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    train_G = True
    for epoch in range(500):
        netG.train()
        print("epoch", epoch)

        train = train_G

        for step, (batch_imgs, batch_labels) in enumerate(trainloader):

            Z_origin = net(batch_imgs.cuda().detach())

            Z = torch.reshape(Z_origin, (batch_size, 128, 1, 1))

            X_bar = netG(Z)
            Z_bar = net(X_bar.detach())


            new_Z = torch.cat((Z_origin, Z_bar), 0)
            new_idx = torch.cat((batch_labels, batch_labels + 10), 0)
            X_bar_show = vutils.make_grid(X_bar, normalize=True)
            X_show = vutils.make_grid(batch_imgs, normalize=True)



            loss, loss_empi, loss_theo = criterionF(new_Z, new_idx)
            loss = loss
            net.zero_grad()
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f"F error is {loss}")

            Z_origin = net(batch_imgs.cuda())
            Z = torch.reshape(Z_origin, (batch_size, 128, 1, 1))
            X_bar = netG(Z)
            Z_bar = net(X_bar)
            new_Z = torch.cat((Z_origin, Z_bar), 0)

            netG.zero_grad()
            #optimizerG.zero_grad()
            lossG, loss_empi, loss_theo = criterionF(new_Z, new_idx)
            lossG = -lossG
            lossG.backward()
            optimizerG.step()
            #print(f"G error is {loss}")


        #show(X_show, epoch)
        show(X_bar_show, epoch)

        scheduler.step()
        print(f"G error is {lossG}")
        print(f"F error is {loss}")
        show(X_show, epoch)
        show(X_bar_show, epoch)



    print("training complete.")


if __name__ == '__main__':
    main()

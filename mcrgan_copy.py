### Import
# from __future__ import print_function
import argparse
import random # to set the python random seed
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
from torchvision import datasets, transforms
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

from loss import MaximalCodingRateReduction



######################
## 给Xili打Call！！！ ##
######################


# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)


### Parameters
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
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


### Model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
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

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf*2, 4, 1, 0, bias=False),
            nn.Flatten()#new
            #nn.LeakyReLU(0.2, inplace=True), #New
            #nn.Linear(ndf, ndf, bias=False)
            #nn.Sigmoid()
        )

    def forward(self, input):
        return F.normalize(self.main(input))


### Train
def train(gen, disc, device, dataloader, optimizerG, optimizerD, criterion, epoch, iters):
  gen.train()
  disc.train()
  img_list = []
  fixed_noise = torch.randn(64, nz, 1, 1, device=device)

  # Establish convention for real and fake labels during training (with label smoothing)
  real_label = 0.9
  fake_label = 0.1
  #for i, data in enumerate(dataloader, 0):
  for i, (data, label) in enumerate(dataloader):

      #*****
      # Update Discriminator
      #*****
      ## Train with all-real batch
      disc.zero_grad()
      # Format batch
      real_cpu = data.to(device)
      #print(real_cpu.shape)
      b_size = real_cpu.size(0)

      #label = torch.full((b_size,), real_label, device=device)
      # Forward pass real batch through D
      Z = disc(real_cpu)
      ## Train with all-fake batch
      # Generate batch of latent vectors
      noise = torch.randn(b_size, nz, 1, 1, device=device)
      fake = gen(noise)
      label.fill_(fake_label)
      new_label = torch.cat((label, label + 10))
      Z_bar = disc(fake.detach())
      new_Z = torch.cat((Z, Z_bar), 0)

      errD, loss_empi, loss_theo = criterion(new_Z, new_label)
      errD = errD
      errD.backward()

      # Update D
      optimizerD.step()

      #*****
      # Update Generator
      #*****
      gen.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost

      # Calculate gradients for G
      Z = disc(real_cpu)
      new_label = torch.cat((label, label + 10))
      Z_bar = disc(fake)
      new_Z = torch.cat((Z, Z_bar), 0)
      errG, loss_empi, loss_theo = criterion(new_Z, new_label)
      errG = (-1) * errG
      errG.backward()
      # Update G
      optimizerG.step()


      # Output training stats
      """
      if i % 50 == 0:
          print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
      """
      # Check how the generator is doing by saving G's output on fixed_noise
      if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):

          print(f"errD is {errD}")
          print(f"ErrG is {errG}")

          with torch.no_grad():
              fake = gen(fixed_noise).detach().cpu()
              show(vutils.make_grid(fake, padding=2, normalize=True), epoch, iters)




      iters += 1
import torchvision.transforms.functional as FF
def show(imgs, epoch, iters):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #plt.show()
    plt.savefig(str(epoch)+str(iters)+".png")
    #plt.close()


def main():
    print("I am executing")
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(manualSeed)  # python random seed
    torch.manual_seed(manualSeed)  # pytorch random seed
    np.random.seed(manualSeed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize cri
    criterionMCR = MaximalCodingRateReduction(gam1= 1, gam2= 1, eps= 0.5)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients


    iters = 0
    for epoch in range(1, num_epochs + 1):
        train(netG, netD, device, trainloader, optimizerG, optimizerD, criterionMCR, epoch, iters)

    # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.



if __name__ == '__main__':
    main()
















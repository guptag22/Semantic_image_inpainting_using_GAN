import os
import torch
import torch.nn as nn
import torch.optim as optim
from dcgan_model import Generator,Discriminator,weights_init

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

params = {
    "bsize" : 128,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 10,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2 }

netG = Generator(params).to(device)
netD = Discriminator(params).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))


filename = "pretrained_model.pth"
if os.path.isfile(filename):
    saved_model = torch.load(filename, map_location=torch.device('cpu'))
    netG.load_state_dict(saved_model['generator'])
    netD.load_state_dict(saved_model['discriminator'])
    optimizerD.load_state_dict(saved_model['optimizerD'])
    optimizerG.load_state_dict(saved_model['optimizerG'])
    params = saved_model['params']
    print("Parameters for this Model - \n", params)

# Start inpainting


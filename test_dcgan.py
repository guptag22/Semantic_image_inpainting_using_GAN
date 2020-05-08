import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dcgan_model import Generator,Discriminator,weights_init
import dataset


checkpoint_filename = "checkpoint/saved_model.pth"
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

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

dataloader = dataset.get_celeba_data(params['bsize'])

def load_checkpoint(G_model, D_model, G_optimizer, D_optimizer, params, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        G_model.load_state_dict(checkpoint['G_state_dict'])
        G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        D_model.load_state_dict(checkpoint['D_state_dict'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        params = checkpoint['params']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return G_model, D_model, G_optimizer, D_optimizer, start_epoch, params

# Create the generator
netG = Generator(params).to(device)
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)
# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(params).to(device)
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Load model if available
netG, netD, optimizerG, optimizerD, start_epoch, params = load_checkpoint(netG,netD,optimizerG,optimizerD,params,checkpoint_filename)
netG = netG.to(device)
netD = netD.to(device)
for state in optimizerG.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)
for state in optimizerD.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    plt.figure(figsize=(8,8))
    plt.subplot(1,1,1)
    plt.axis("off")
    plt.title("Gen Images")
    plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
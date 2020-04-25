import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dcgan_model import Generator,Discriminator,weights_init

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# DCGAN parameters

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

real_label = 1
fake_label = 0
# Initialize BCELoss function for dcgan
criterion = nn.BCELoss()

# Start inpainting

class Inpaint:
    def __init__(self):
        # Initialize the DCGAN model and optimizers
        self.netG = Generator(params).to(device)
        self.netD = Discriminator(params).to(device)

        filename = "pretrained_model.pth"
        if os.path.isfile(filename):
            saved_model = torch.load(filename, map_location=torch.device('cpu'))
            self.netG.load_state_dict(saved_model['generator'])
            self.netD.load_state_dict(saved_model['discriminator'])
            params = saved_model['params']
        
        self.batch_size = 64 # Batch size for inpainting
        self.image_size = params['imsize'] # 64
        self.num_channels = params['nc'] # 3
        self.z_dim = params['nz'] # 100
        self.nIters = 3000 # Iterations 
        self.lamda = 0.2
        self.z = torch.randn(self.batch_size, self.z_dim, 1, 1, device=device, requires_grad=True)
        self.opt = torch.optim.Adam(self.z, lr = 0.0003)

    def get_mask(self,images):
        # Create mask according to the input image

        return 0

    def preprocess(self,images,mask):
        # preprocess the images and masks

        return 0

    def get_imp_weighting(self, mask):
        # Implement eq 3

        return 0

    def run_dcgan(self,z_i):
        G_z_i = self.netG(z_i)
        label = torch.full((self.batch_size,), real_label, device=device)
        D_G_z_i = self.netD(G_z_i)
        errG = criterion(D_G_z_i, label)

        return G_z_i, errG

    def get_context_loss(self, G_z_i, images, mask):
        # Calculate context loss
        # Implement eq 4
        W = self.get_imp_weighting(mask)
        # TODO: verify norm output. Its probably a vector. we need a single value
        context_loss = torch.norm(torch.mul(W, G_z_i - images), p=1) 

        return context_loss

    def generate_z_hat(self, images, mask):
        # Backpropagation for z
        for i in range(self.nIters):
            self.opt.zero_grad()
            self.G_z_i, self.errG = self.run_dcgan(self.z)
            self.perceptual_loss = self.errG
            self.context_loss = self.get_context_loss(self.G_z_i, images, mask)
            self.loss = self.context_loss + (self.lamda * self.perceptual_loss)
            grad = torch.autograd.grad(self.loss, self.z)

            # Update z
            # https://github.com/moodoki/semantic_image_inpainting/blob/extensions/src/model.py#L182
            
            # TODO: Not sure if this next would work to update z. Check
            self.opt.step() 


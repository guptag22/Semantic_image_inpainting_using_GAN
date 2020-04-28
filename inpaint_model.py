import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import convolve2d
from dcgan_model import Generator,Discriminator,weights_init
import dataset

import torchvision.utils as vutils
import matplotlib.pyplot as plt 

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# DCGAN parameters

real_label = 1
fake_label = 0
# Initialize BCELoss function for dcgan
criterion = nn.BCELoss()

# Start inpainting

class Inpaint:
    def __init__(self):
        # Initialize the DCGAN model and optimizers
        params = {
            "bsize" : 128,# Batch size during DCGAN training.
            'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
            'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
            'nz' : 100,# Size of the Z latent vector (the input to the generator).
            'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
            'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
            'nepochs' : 10,# Number of training epochs.
            'lr' : 0.0002,# Learning rate for optimizers
            'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
            'save_epoch' : 2 }
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
        self.nIters = 1000 # Iterations 
        self.lamda = 0.2

    def preprocess(self,images,masks):
        # preprocess the images and masks

        return 0

    def get_imp_weighting(self, masks, nsize):
        # TODO: Implement eq 3
        kernel = np.ones((nsize,nsize), dtype=np.float32)
        kernel = kernel/np.sum(kernel)
        weighted_masks = torch.empty(masks.shape[0],3,masks.shape[1],masks.shape[2])
        for i in range(len(masks)):
            weighted_mask = masks[i] * convolve2d(masks[i], kernel, mode='same', boundary='symm')
            # create 3 channels to match image channels
            weighted_mask = torch.unsqueeze(weighted_mask,0)
            weighted_masks[i] = torch.repeat_interleave(weighted_mask, 3, dim = 0)

        return weighted_masks

    def run_dcgan(self,z_i):
        G_z_i = self.netG(z_i)
        label = torch.full((z_i.shape[0],), real_label, dtype=torch.float, device=device)
        D_G_z_i = torch.squeeze(self.netD(G_z_i))
        errG = criterion(D_G_z_i, label)

        return G_z_i, errG

    def get_context_loss(self, G_z_i, images, masks):
        # Calculate context loss
        # Implement eq 4
        nsize = 7
        W = self.get_imp_weighting(masks, nsize)
        # TODO: verify norm output. Its probably a vector. We need a single value
        context_loss = torch.sum(torch.abs(torch.mul(W, G_z_i - images))) 

        return context_loss

    def generate_z_hat(self,real_images, images, masks):
        # Backpropagation for z
        z = torch.randn(images.shape[0], self.z_dim, 1, 1, device=device, requires_grad=True)
        opt = torch.optim.Adam([z], lr = 0.0003)
        for i in range(self.nIters):
            opt.zero_grad()
            G_z_i, errG = self.run_dcgan(z)
            # with torch.no_grad():
            #     plt.figure(figsize=(8,8))
            #     # plt.subplot(1,2,1)
            #     plt.axis("off")
            #     plt.title("Generated Images")
            #     plt.imshow(np.transpose(vutils.make_grid(G_z_i.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
            #     plt.show()

            perceptual_loss = errG
            context_loss = self.get_context_loss(G_z_i, images, masks)
            loss = context_loss + (self.lamda * perceptual_loss)
            loss.backward()
            # grad = torch.autograd.grad(loss, self.z)

            # Update z
            # https://github.com/moodoki/semantic_image_inpainting/blob/extensions/src/model.py#L182
            
            # TODO: Not sure if this next would work to update z. Check
            opt.step() 

            # TODO: Clip Z to be between -1 and 1

            if i%50 == 0:
                print(i)
            if i%250 == 0:
                with torch.no_grad():
                    plt.figure(figsize=(8,8))
                    plt.subplot(1,2,1)
                    plt.axis("off")
                    plt.title("Real Images")
                    plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

                    plt.subplot(1,2,2)
                    plt.axis("off")
                    plt.title("Generated Images")
                    plt.imshow(np.transpose(vutils.make_grid(G_z_i.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
                    plt.show()

        return z

    def main(self, dataloader):
        for i, data in enumerate(dataloader, 0):
            print(i)
            if i>0:
                break
            real_images = data[0]
            corrupt_images = data[1]
            masks = data[2]/255
            # Get optimal latent space vectors (Z^) for corrupt images
            z_hat = self.generate_z_hat(real_images,corrupt_images,masks)
            with torch.no_grad():
                G_z_hat, errG = self.run_dcgan(z_hat)
                plt.figure(figsize=(8,8))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Real Images")
                plt.imshow(np.transpose(vutils.make_grid(real_images.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Generated Images")
                plt.imshow(np.transpose(vutils.make_grid(G_z_hat.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
                plt.show()


if __name__ == "__main__":
    dataloader = dataset.get_celeba_data()
    Inpaint_net = Inpaint()
    Inpaint_net.main(dataloader)

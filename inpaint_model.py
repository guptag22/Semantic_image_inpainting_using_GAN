import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from scipy.signal import convolve2d
from dcgan_model import Generator,Discriminator,weights_init
import dataset
from poissonblending import blend

import torchvision.utils as vutils
import matplotlib.pyplot as plt 

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# DCGAN parameters

real_label = 1
fake_label = 0
# Initialize BCELoss function for dcgan
criterion = nn.BCELoss()
bsize = 4
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
        filename = "./checkpoint/saved_model.pth"
        # filename = "pretrained_model.pth"
        if os.path.isfile(filename):
            saved_model = torch.load(filename, map_location=torch.device(device))
            self.netG.load_state_dict(saved_model['G_state_dict'])
            self.netD.load_state_dict(saved_model['D_state_dict'])
        else:
            print("Trained DCGAN not found!")
            sys.exit()
            # params = saved_model['params']
        
        self.batch_size = 64 # Batch size for inpainting
        self.image_size = params['imsize'] # 64
        self.num_channels = params['nc'] # 3
        self.z_dim = params['nz'] # 100
        self.nIters = 3000 # Inpainting Iterations
        self.blending_steps = 100
        self.lamda = 0.2
        self.momentum = 0.9
        self.lr = 0.0003


    def image_gradient(self,image):
        a = torch.Tensor([[[[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]]]).to(device)
        a = torch.repeat_interleave(a, 3, dim = 1)
        G_x = F.conv2d(image, a, padding=1)
        b = torch.Tensor([[[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]]]).to(device)
        b = torch.repeat_interleave(b, 3, dim = 1)
        G_y = F.conv2d(image, b, padding=1)
        return G_x, G_y


    def posisson_blending(self,corrupted_images,generated_images,masks):
        print("Starting Poisson blending ...")
        initial_guess = masks*corrupted_images + (1-masks)*generated_images
        image_optimum = nn.Parameter(torch.FloatTensor(initial_guess.detach().cpu().numpy()).to(device))
        optimizer_blending = optim.Adam([image_optimum])
        generated_grad_x, generated_grad_y = self.image_gradient(generated_images)

        for epoch in range(self.blending_steps):
            optimizer_blending.zero_grad()
            image_optimum_grad_x, image_optimum_grad_y = self.image_gradient(image_optimum)
            blending_loss = torch.sum(((generated_grad_x-image_optimum_grad_x)**2 + (generated_grad_y-image_optimum_grad_y)**2)*(1-masks))
            blending_loss.backward()
            image_optimum.grad = image_optimum.grad*(1-masks)
            optimizer_blending.step()

            print("[Epoch: {}/{}] \t[Blending loss: {:.3f}]   \r".format(1+epoch, self.blending_steps, blending_loss), end="") 
        print("")

        del optimizer_blending
        return image_optimum.detach()


    def get_imp_weighting(self, masks, nsize):
        # TODO: Implement eq 3
        kernel = torch.ones((1,1,nsize,nsize)).to(device)
        kernel = kernel/torch.sum(kernel)
        weighted_masks = torch.empty(masks.shape[0], 3, masks.shape[2], masks.shape[3]).to(device)
        padded_masks = F.pad(masks, (2, 2, 2, 2), "constant", 1)
        # print(kernel.shape, masks.shape)
        conv = F.conv2d(input=padded_masks, weight=kernel, padding=1)
        # print(conv.shape)
        # print(masks.shape)
        weighted_masks = masks * conv
        # print(weighted_masks.shape)
        # for i in range(len(masks)):
        #     weighted_mask = masks[i] * convolve2d(masks[i], kernel, mode='same', boundary='symm')
        #     # create 3 channels to match image channels
        #     weighted_mask = torch.unsqueeze(weighted_mask,0)
        #     weighted_masks[i] = torch.repeat_interleave(weighted_mask, 3, dim = 0)

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
        # z = 2*torch.rand(images.shape[0], self.z_dim, 1, 1, device=device) -1
        z = torch.randn(images.shape[0], self.z_dim, 1, 1, device=device)
        opt = torch.optim.Adam([z], lr = 0.0003)
        v = 0
        for i in range(self.nIters):
            opt.zero_grad()
            z.requires_grad = True
            G_z_i, errG = self.run_dcgan(z)
            perceptual_loss = errG
            context_loss = self.get_context_loss(G_z_i, images, masks)
            loss = context_loss + (self.lamda * perceptual_loss)
            # loss.backward()
            grad = torch.autograd.grad(loss, z)

            # Update z
            # https://github.com/moodoki/semantic_image_inpainting/blob/extensions/src/model.py#L182
            v_prev = v
            v = self.momentum*v - self.lr*grad[0]
            with torch.no_grad():
                z += (-self.momentum * v_prev +
                        (1 + self.momentum) * v)
                z = torch.clamp(z, -1, 1)
            # TODO: Not sure if this next would work to update z. Check
            # opt.step() 

            # TODO: Clip Z to be between -1 and 1

            if i%100 == 0:
                print(i)
            if i%250 == 0:
                with torch.no_grad():
                    # print("masks shape:", masks.shape)
                    channeled_masks = torch.empty(masks.shape[0],3,masks.shape[2],masks.shape[3]).to(device)
                    # print("channeled_masks shape:", channeled_masks.shape)
                    # unsq_masks = torch.unsqueeze(masks,1)
                    # print("unsq masks shape: ", unsq_masks.shape)
                    for j in range(len(masks)):
                        channeled_masks[j] = torch.repeat_interleave(masks[j], 3, dim = 0)
                    merged_images = channeled_masks*images + (1-channeled_masks)*G_z_i
                    plt.figure(figsize=(8,8))
                    plt.subplot(2,1,1)
                    plt.axis("off")
                    plt.title("Real Images")
                    plt.imshow(np.transpose(vutils.make_grid(real_images.to(device)[:bsize], padding=5, normalize=True).cpu(),(1,2,0)))

                    plt.subplot(2,1,2)
                    plt.axis("off")
                    plt.title("Generated Images")
                    plt.imshow(np.transpose(vutils.make_grid(merged_images.to(device)[:bsize], padding=5, normalize=True).cpu(),(1,2,0)))
                    plt.savefig("iter_{}.png".format(i))
                    # plt.show()
                    

        return z

    def main(self, dataloader):
        for i, data in enumerate(dataloader, 0):
            print(i)
            if i>0:
                break
            real_images = data[0].to(device)
            corrupt_images = data[1].to(device)
            masks = (data[2]/255).to(device)
            masks.unsqueeze_(1)
            # Get optimal latent space vectors (Z^) for corrupt images
            z_hat = self.generate_z_hat(real_images, corrupt_images, masks)
            with torch.no_grad():
                G_z_hat, _ = self.run_dcgan(z_hat)
                channeled_masks = torch.empty(masks.shape[0],3,masks.shape[2],masks.shape[3]).to(device)
                for j in range(len(masks)):
                    channeled_masks[j] = torch.repeat_interleave(masks[j], 3, dim = 0)
                merged_images = channeled_masks*corrupt_images + (1-channeled_masks)*G_z_hat
            # blended_images = np.empty_like(corrupt_images.cpu().numpy())
            # for k in range(len(merged_images)):
                # blended_images[k] = blend( corrupt_images[k].cpu().numpy(), G_z_hat[k].detach().cpu().numpy(), (masks[k]).cpu().numpy() )
            # blended_images = self.posisson_blending( corrupt_images, G_z_hat.detach(), channeled_masks )
            plt.figure(figsize=(8,8))
            plt.subplot(3,1,1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_images.to(device)[:bsize], padding=5, normalize=True).cpu(),(1,2,0)))

            plt.subplot(3,1,2)
            plt.axis("off")
            plt.title("Corrupt Images")
            plt.imshow(np.transpose(vutils.make_grid(corrupt_images.to(device)[:bsize], padding=5, normalize=True).cpu(),(1,2,0)))
            plt.savefig("final.png")

            plt.subplot(3,1,3)
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(merged_images.to(device)[:bsize], padding=5, normalize=True).cpu(),(1,2,0)))
            plt.savefig("final.png")
            
            # plt.subplot(3,1,3)
            # plt.axis("off")
            # plt.title("Before Merging Images")
            # plt.imshow(np.transpose(vutils.make_grid(torch.tensor(G_z_hat.to(device)[:bsize]), padding=5, normalize=True).cpu(),(1,2,0)))
            # plt.savefig("final.png")
            # plt.show()
                
            

if __name__ == "__main__":
    dataloader = dataset.get_celeba_data(bsize)
    Inpaint_net = Inpaint()
    Inpaint_net.main(dataloader)

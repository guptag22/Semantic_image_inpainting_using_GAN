import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
import numpy as np
import poissonblending as blending
from scipy.signal import convolve2d

class ModelInpaint():
	def __init__(self, modelfilename, params = , gene_input = , gen_output = , 
		disc_input = , disc_out = , z_dim = , batch_size = 		):

	"""
	Model for semantic image inpainting
	Load weights of DCGAN
	Create a graph based on loss function in the paper

			modelfilename - Not sure if we need it 
            params - stored training parameters: lambda, num_iter
            gen_input - generator input
            gen_output - generator output
            disc_input - discriminator input
            disc_output - discriminator output
            z_dim - dimentsion of z, where z is the latent space dimension of GAN in paper
            batch_size - training batch size

	"""
	# In their code online, they also used model_name as another argument. 
	# I suppose we don't need it? 

	self.params = params
	self.batch_size = batch_size
	self.z_dim = z_dim


	self.gen_input = 
	self.gen_output =
	self.gen_loss = 
	self.disc_input = 
	self.disc_output = 
 
	self.lambda = params.lambda  # lambda is the balance between Lc and Lp
	self.init_z()

	def init_z(self):
		self.z = np.random.randn(self.batch_size, self.z_dim)

	def preprocess(self, images, image_masks, nsize = 7):
		"""
		To prepare data to for the network. 
		Not exactly sure what we should do yet. 
		They 1) rescaled the pixel value, 
		     2) considered situations where importance weight W is used or not for masks
		     3) considered different situations of image shapes


		Deleted the useWeightMask = True argument. Can add it back. 

			images - input images
			image_masks - input mask
			nsize - N(i) in equation (3) in the paper. 
					It refers to the set of neighbors of pixel i in a local window.
	
		"""

	def postprocess(self, gen_output, blend = True):
		"""
		Apply poisson blending using binary mask. 
		Refer to Figure 5 in the paper. 
		We can also give evaluation of blending = false vs. blending = true.
		"""

		image_in = 
		image_out = 
		if blend:

		else:

		return image_out

	# LOSSSSSS and GRADDDDDDD 
	def calculate_loss(self): 
		"""
		called 'build_inpaint_graph' in the author's code.
		To calculate the loss and return gradient. 
		Loss = Lc + Lp see equation (2) in the paper. 
		"""
		
 		self.loss_c =                   #context loss. see equation (4).

		self.loss_p = self.gen_loss     #perceptual loss. See equation (5). 
		self.inpaint_loss = self.loss_c + self.lambda*self.loss_p	#equation (2)
		self.inpaint_grad =    

	# Baaaaaaaackpropogation to input
	def backprop(self):
		"""
		To obtain latent space representation of target image
		Their code used (accelerated) gradient descent
        
        Return: generator output images. 
		"""
		v = 0 
		for i in range(self.params.num_iter):

			loss, grad, imout = 

			# check their code and try to understand. 
			# still not sure. 

			self.z = np.clip(self.z, -1, 1)

		return imout 

	@staticmethod
	def loadpb(filename):
		"""
		TODO:
		They used loadpb to load pretrained graph from protobuf file. 
		Not sure how to do it in pytorch and how to make it compatible with our code.
		"""


	@staticmethod
    def createWeightedMask(mask, nsize=7):
        """
        Calculate the importance weighting term, W, defined in equation (3) in the paper.
    
            mask - binary mask input. numpy float32 array
            nsize - N(i) in equation (3) in the paper. 
					It refers to the set of neighbors of pixel i in a local window.
        """
        kernel = np.ones((nsize,nsize), dtype=np.float32)
        kernel = kernel/np.sum(kernel)
        weighted_mask = mask * convolve2d(mask, kernel, mode='same', boundary='symm')
        return weighted_mask

	@staticmethod
    def poissonblending(image1, image2, mask):
        """
        We are using code from poissonblending. 
		Need to give the original code credits. 
        """
        return blending.blend(image1, image2, 1 - mask)

















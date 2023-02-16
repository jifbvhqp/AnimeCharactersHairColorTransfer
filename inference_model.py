# Imports PIL module 
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.utils import save_image
import torch
import os
import random
import matplotlib.pyplot as plt
from model import Generator
#from model import Discriminator
from upscale_img import upscale_tensor
import numpy as np

#Denormalize
def denorm(x):
	out = (x + 1) / 2
	out.clamp_(0, 1)
	return out

image_size = 96
g_conv_dim = 64
d_conv_dim = 64
c_dim = 5
g_repeat_num = 6
d_repeat_num = 6
batch_size = 1

input_path = 'inputs'
mode_path = 'model'
result_dir = 'outputs'

#Load model
G = Generator(g_conv_dim,c_dim,g_repeat_num)
#D = Discriminator(image_size, d_conv_dim, c_dim,d_repeat_num) 
G_path = os.path.join(mode_path, 'G.ckpt')
#D_path = os.path.join(mode_path, 'D.ckpt')
G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
#D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

images_paths_list = [os.path.join(input_path, path) for path in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, path))]

for img_path in images_paths_list:
	#Load image
	im = Image.open(img_path)	

	#Image transform
	transform = []
	transform.append(T.Resize((image_size,image_size)))
	transform.append(T.ToTensor())
	transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = T.Compose(transform)
	im = transform(im)
	im = torch.unsqueeze(im, 0)
	print('Input image transfered shape.',im.shape)

	#Inference
	with torch.no_grad():
		c_trg_list = []
		for i in range(c_dim):		  
			out = torch.zeros(batch_size, c_dim)
			labels = torch.ones(batch_size)*i
			out[np.arange(batch_size), labels.long()] = 1.0
			c_trg_list.append(out)

		img_result_list = [im]
		for c_trg in c_trg_list:
			result = G(im, c_trg)
			img_result_list.append(result)
		x_concat = torch.cat(img_result_list, dim=3)
		x_concat = denorm(x_concat)
	print('Model output image shape.',x_concat.shape)
	res = upscale_tensor(x_concat)
	print('Model output image upscaled shape.',res.shape)

	#Show Image
	plt.imshow(res.squeeze(0).permute(1,2,0))
	plt.show()


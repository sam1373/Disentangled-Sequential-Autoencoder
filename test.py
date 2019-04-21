import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import *
from tqdm import *
#from dataset import *
from moving_mnist.moving_mnist_loader import MovingMnistLoader
#from dataset.sprites_test import Sprites
import matplotlib.pyplot as plt


np.random.seed(2019)
torch.manual_seed(2019)

batch_size = 64
frames = 10

#frames = 10
#16000 / bs
#trainloader = Sprites('dataset/lpc-dataset/train', 6000, batch_size=batch_size)
trainloader = MovingMnistLoader(16000, frames=frames, batch_size=batch_size)
testLoader = trainloader
#for now

#loader = torch.utils.data.DataLoader(sprite, batch_size=batch_size, shuffle=True, num_workers=4)
device = torch.device('cuda:0')

#model = DisentangledVAE(frames=30, f_dim=256, z_dim=32, step=256, factorised=True, device=device)
#in_channels=3
model = DisentangledVAE_Adversarial(frames=frames, in_channels=1)

checkpoint_dir = "model_attn.cp"

checkpoint = torch.load(checkpoint_dir)
model.load_state_dict(checkpoint['state_dict'])

def get_norm(model0):
    total_norm = 0
    for p in model0.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm

with torch.no_grad():

        while(1):
            print("Enter mult:")
            mult = float(input())

            x_gen = model.gen_seq(10, mult=mult)
            
            x_gen = x_gen.view(10*model.frames, model.in_channels, 64,64)

            #torchvision.utils.save_image(x_gen,'sample/test.png', nrow=frames)

            #batch_tensor = torch.randn(*(10, 3, 256, 256))

            grid_img = torchvision.utils.make_grid(x_gen, nrow=frames)

            plt.imshow(grid_img.permute(1, 2, 0).cpu())
            plt.show()



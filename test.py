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
from dataset import *
from moving_mnist.moving_mnist_loader import MovingMnistLoader
from dataset.sprites_test import Sprites
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


with torch.no_grad():

        while(1):
            x_gen, sc = model.gen_seq(4, get_score=True)
            sc_mean = sc.mean().item()
            print(sc_mean)
            if(sc_mean < 0.1):
                continue
            x_gen = x_gen.view(4*model.frames, model.in_channels, 64,64)

            #torchvision.utils.save_image(x_gen,'%s/epoch%d.png' % (self.sample_path,epoch), nrow=self.model.frames)

            #batch_tensor = torch.randn(*(10, 3, 256, 256))

            grid_img = torchvision.utils.make_grid(x_gen, nrow=frames)

            print(grid_img.shape)

            plt.imshow(grid_img.permute(1, 2, 0).cpu())
            plt.show()

            print(sc)


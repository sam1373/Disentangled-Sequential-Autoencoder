import os
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import *
from tqdm import *

#from dataset import *
from moving_mnist.moving_mnist_loader import MovingMnistLoader
#from dataset.sprites_test import Sprites

__all__ = ['loss_fn', 'Trainer']


np.random.seed(2019)
torch.manual_seed(2019)


def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_post_mean,z_post_logvar, z_prior_mean, z_prior_logvar):
    """
    Loss function consists of 3 parts, the reconstruction term that is the MSE loss between the generated and the original images
    the KL divergence of f, and the sum over the KL divergence of each z_t, with the sum divided by batch_size

    Loss = {mse + KL of f + sum(KL of z_t)} / batch_size
    Prior of f is a spherical zero mean unit variance Gaussian and the prior of each z_t is a Gaussian whose mean and variance
    are given by the LSTM
    """
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum');
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return (mse + kld_f + kld_z)/batch_size, kld_f/batch_size, kld_z/batch_size

def loss_fn2(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum');
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean,2) - torch.exp(z_logvar))

    #multiply kdl?
    return (mse + kld_f + kld_z)/batch_size, kld_f/batch_size, kld_z/batch_size

def loss_autoenc(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum');

    return mse, torch.Tensor([0]), torch.Tensor([0])


class Trainer(object):
    def __init__(self,model,train,test,trainloader,testloader, test_f_expand,
                 epochs=100,batch_size=64,learning_rate=0.001,nsamples=1,sample_path='sample',
                 recon_path='recon', transfer_path = 'transfer', 
                 checkpoints='model.pth', style1='image1.sprite', style2='image2.sprite', device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.train = train
        self.test = test
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(),self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.transfer_path = transfer_path
        self.test_f_expand = test_f_expand

        self.test_z = torch.randn(4, self.model.frames, self.model.z_dim_seq, device=device) * 0.5
        #self.test_z = torch.randn(4, self.model.z_dim, device=device) * 0.5
        self.test_f = torch.randn(4, self.model.f_dim, device=device) * 0.5

        self.epoch_losses = []


        self.test.shuffle()

        self.image1 = self.test[0][0]
        self.image2 = self.test[5][3]
        #self.image1 = torch.unsqueeze(self.image1,0)
        #self.image2 = torch.unsqueeze(self.image2,0)
        
    
    def save_checkpoint(self,epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'losses' : self.epoch_losses},
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0


    def sample_frames(self,epoch):
        with torch.no_grad():
            """
            #_,_,test_z = self.model.sample_z(1, random_sampling=False)
            test_z = self.test_z
            test_f = self.test_f
            #print(test_z.shape)
            #print(self.test_f_expand.shape)
            #test_zf = torch.cat((test_z, self.test_f_expand), dim=2)
            #recon_x = self.model.decode_frames(test_zf) 
            #recon_x = recon_x.view(self.samples*30,3,64,64)
            recon_x = self.model.decode(test_f, test_z)
            recon_x = recon_x.view(4*self.model.frames, self.model.in_channels, 64,64)
            """
            x_gen = self.model.gen_seq(4).view(4*self.model.frames, self.model.in_channels, 64,64)
            torchvision.utils.save_image(x_gen,'%s/epoch%d.png' % (self.sample_path,epoch), nrow=self.model.frames)
    
    def recon_frame(self,epoch,original):
        with torch.no_grad():
            recon = self.model(original)[0]
            #af, az = self.model.get_attn_maps(original)

            original = original.view(self.batch_size, self.model.frames, self.model.in_channels, 64, 64)
            original = original[0]
            recon = recon.view(self.batch_size, self.model.frames, self.model.in_channels, 64, 64)
            recon = recon[0]
            #af = af.view(-1, self.model.frames, 1, 64, 64).expand(-1, self.model.frames, self.model.in_channels, 64, 64)
            #az = az.view(-1, self.model.frames, 1, 64, 64).expand(-1, self.model.frames, self.model.in_channels, 64, 64)
            #af = af[0]
            #az = az[0]

            image = torch.cat((original,recon),dim=0)
            image = image.view(2*self.model.frames, self.model.in_channels, 64,64)
            #os.makedirs(os.path.dirname('%s/epoch%d.png' % (self.recon_path,epoch)),exist_ok=True)
            torchvision.utils.save_image(image,'%s/epoch%d.png' % (self.recon_path,epoch), nrow=self.model.frames)

    def style_transfer(self,epoch):
        with torch.no_grad():
            image1, image2 = self.image1, self.image2
            f1, z1 = model.encode(image1)
            f2, z2 = model.encode(image2)
            image21 = model.decode(f2, z1)[0]
            image12 = model.decode(f1, z2)[0]
            #os.makedirs(os.path.dirname('%s/epoch%d/image12.png' % (self.transfer_path,epoch)),exist_ok=True)

            image = torch.cat((image1, image2, image12, image21), dim=0)
            image = image.view(4*self.model.frames, self.model.in_channels, 64, 64)

            #torchvision.utils.save_image(image1,'%s/image1.png' % (self.transfer_path))
            #torchvision.utils.save_image(image2,'%s/image2.png' % (self.transfer_path))
            #torchvision.utils.save_image(image12,'%s/epoch%d_image12.png' % (self.transfer_path,epoch))
            torchvision.utils.save_image(image,'%s/epoch%d.png' % (self.transfer_path,epoch), nrow=self.model.frames)

    def train_model(self):
       self.model.train()

       #avgDiff = 0
       for epoch in range(self.start_epoch,self.epochs):
           self.trainloader.shuffle()
           losses = []
           kld_fs = []
           kld_zs = []
           print("Running Epoch : {}".format(epoch+1))
           print(len(self.trainloader))


           lastDiff = 0
           #lastDiff = avgDiff
           avgDiff = 0

           for i,dataitem in tqdm(enumerate(self.trainloader,1)):
               if i >= len(self.trainloader):
                break
               data = dataitem
               data = data.to(self.device)

               TINY = 1e-15

               loss4 = torch.Tensor([0])


               if lastDiff > 0.1:
                 self.model.zero_grad_all()

                 x_recon, enc_score = self.model(data)

                 loss1 = 1000 * F.mse_loss(x_recon, data, reduction='mean') - torch.mean(enc_score)

                 #we want enc to confuse discr and have discr give 1 to real data(even though it should give 0)

                 loss1.backward()


                 self.model.enc_dec_step()

               else:

                 self.model.zero_grad_all()

                 x_recon, _ = self.model(data)

                 loss1 = 1000 * F.mse_loss(x_recon, data, reduction='mean')

                 #only recon loss when discr is already confused

                 loss1.backward()


                 self.model.enc_dec_step()


               ##

               self.model.zero_grad_all()

               discr_real_score = self.model.forward_score(data)

               loss2 = torch.mean(discr_real_score)
               #we want discr to give 0 for real data

               if loss2.item() > -0.95:
                   loss2.backward()

                   self.model.discr.optimizer.step()

               ##

               self.model.zero_grad_all()

               f, z, discr_gen_score = self.model.gen_codes()



               loss3 = -torch.mean(discr_gen_score)
               #we want discr to give 1 for generated data

               if loss3.item() > -0.95:
                   loss3.backward()

                   self.model.discr.optimizer.step()

               ##

               if lastDiff > 0.1:

                 self.model.zero_grad_all()

                 f, z, discr_gen_score = self.model.gen_codes()


                 loss4 = torch.mean(discr_gen_score)
                 #we want distr to get 0 (to be more similar to discr)

                 loss4.backward()

                 self.model.distr.optimizer.step()


               print(loss1, loss2, loss3, loss4)

               total_loss = loss1 + loss2 + loss3 + loss4

               avgDiff += loss3.item() * -1 - loss2.item()
               lastDiff = loss3.item() * -1 - loss2.item()



               """
               self.optimizer.zero_grad()

               f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, recon_x = self.model(data)
               loss, kld_f, kld_z = loss_autoenc(data, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar)
               #loss shoul be fn2
               """

               #f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean, z_prior_logvar, recon_x = self.model(data)
               #loss, kld_f, kld_z = loss_fn(data, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
               #print(loss, kld_f, kld_z)

               #loss.backward()
               #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
               #self.optimizer.step()
               losses.append(total_loss.item())
               #discr_real.append((loss2 + )
               #kld_fs.append(kld_f.item())
               #kld_zs.append(kld_z.item())
           #discr_meanloss = np.mean(discr_loss)
           meanloss = np.mean(losses)

           avgDiff /= len(self.trainloader)
           #meanf = np.mean(kld_fs)
           #meanz = np.mean(kld_zs)
           self.epoch_losses.append(meanloss)
           print("Epoch {} : Average Loss: {}".format(epoch+1, meanloss))

           print("Disc. quality: {}".format(avgDiff))
           self.save_checkpoint(epoch)
           self.model.eval()
           #self.sample_frames(epoch+1)
           sample = self.test[int(torch.randint(0,len(self.test),(1,)).item())]
           sample = torch.unsqueeze(sample,0)
           sample = sample.to(self.device)
           if (epoch + 1) % 4 == 0:
            self.sample_frames(epoch+1)
            self.recon_frame(epoch+1,sample)
            self.style_transfer(epoch+1)
           self.model.train()
       print("Training is complete")

#sprite = Sprites('./dataset/lpc-dataset/train', 6767)
#sprite_test = Sprites('./dataset/lpc-dataset/test', 791)
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

test_f = torch.rand(1,256, device=device)
test_f = test_f.unsqueeze(1).expand(1, frames, 256)
#0.0002
trainer = Trainer(model, trainloader, testLoader, trainloader, None, test_f, batch_size=batch_size, epochs=1500, learning_rate=0.0002, device=device, checkpoints="model_attn.cp")
trainer.load_checkpoint()
trainer.sample_frames(0)

sample = trainer.test[int(torch.randint(0,len(trainer.test),(1,)).item())]
sample = torch.unsqueeze(sample,0)
sample = sample.to(trainer.device)
trainer.recon_frame(0, sample)

trainer.style_transfer(0)

trainer.train_model()

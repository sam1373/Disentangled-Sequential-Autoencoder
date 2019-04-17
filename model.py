import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

from modules import *

LEARNING_RATE = 0.0002

class DisentangledVAE(nn.Module):
    """
    Network Architecture:
        PRIOR OF Z:
            The prior of z is a Gaussian with mean and variance computed by the LSTM as follows
                h_t, c_t = prior_lstm(z_t-1, (h_t, c_t)) where h_t is the hidden state and c_t is the cell state
            Now the hidden state h_t is used to compute the mean and variance of z_t using an affine transform
                z_mean, z_log_variance = affine_mean(h_t), affine_logvar(h_t)
                z = reparameterize(z_mean, z_log_variance)
            The hidden state has dimension 512 and z has dimension 32

        CONVOLUTIONAL ENCODER:
            The convolutional encoder consists of 4 convolutional layers with 256 layers and a kernel size of 5 
            Each convolution is followed by a batch normalization layer and a LeakyReLU(0.2) nonlinearity. 
            For the 3,64,64 frames (all image dimensions are in channel, width, height) in the sprites dataset the following dimension changes take place
            
            3,64,64 -> 256,64,64 -> 256,32,32 -> 256,16,16 -> 256,8,8 (where each -> consists of a convolution, batch normalization followed by LeakyReLU(0.2))

            The 8,8,256 tensor is unrolled into a vector of size 8*8*256 which is then made to undergo the following tansformations
            
            8*8*256 -> 4096 -> 2048 (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

        APPROXIMATE POSTERIOR FOR f:
            The approximate posterior is parameterized by a bidirectional LSTM that takes the entire sequence of transformed x_ts (after being fed into the convolutional encoder)
            as input in each timestep. The hidden layer dimension is 512

            Then the features from the unit corresponding to the last timestep of the forward LSTM and the unit corresponding to the first timestep of the 
            backward LSTM (as shown in the diagram in the paper) are concatenated and fed to two affine layers (without any added nonlinearity) to compute
            the mean and variance of the Gaussian posterior for f

        APPROXIMATE POSTERIOR FOR z (FACTORIZED q)
            Each x_t is first fed into an affine layer followed by a LeakyReLU(0.2) nonlinearity to generate an intermediate feature vector of dimension 512,
            which is then followed by two affine layers (without any added nonlinearity) to compute the mean and variance of the Gaussian Posterior of each z_t

            inter_t = intermediate_affine(x_t)
            z_mean_t, z_log_variance_t = affine_mean(inter_t), affine_logvar(inter_t)
            z = reparameterize(z_mean_t, z_log_variance_t)

        APPROXIMATE POSTERIOR FOR z (FULL q)
            The vector f is concatenated to each v_t where v_t is the encodings generated for each frame x_t by the convolutional encoder. This entire sequence  is fed into a bi-LSTM
            of hidden layer dimension 512. Then the features of the forward and backward LSTMs are fed into an RNN having a hidden layer dimension 512. The output h_t of each timestep
            of this RNN transformed by two affine transformations (without any added nonlinearity) to compute the mean and variance of the Gaussian Posterior of each z_t

            g_t = [v_t, f] for each timestep
            forward_features, backward_features = lstm(g_t for all timesteps)
            h_t = rnn([forward_features, backward_features])
            z_mean_t, z_log_variance_t = affine_mean(h_t), affine_logvar(h_t)
            z = reparameterize(z_mean_t, z_log_variance_t)

        CONVOLUTIONAL DECODER FOR CONDITIONAL DISTRIBUTION p(x_t | f, z_t)
            The architecture is symmetric to that of the convolutional encoder. The vector f is concatenated to each z_t, which then undergoes two subsequent
            affine transforms, causing the following change in dimensions
            
            256 + 32 -> 4096 -> 8*8*256 (where each -> consists of an affine transformation, batch normalization followed by LeakyReLU(0.2))

            The 8*8*256 tensor is reshaped into a tensor of shape 256,8,8 and then undergoes the following dimension changes 

            256,8,8 -> 256,16,16 -> 256,32,32 -> 256,64,64 -> 3,64,64 (where each -> consists of a transposed convolution, batch normalization followed by LeakyReLU(0.2)
            with the exception of the last layer that does not have batchnorm and uses tanh nonlinearity)

    Hyperparameters:
        f_dim: Dimension of the content encoding f. f has the shape (batch_size, f_dim)
        z_dim: Dimension of the dynamics encoding of a frame z_t. z has the shape (batch_size, frames, z_dim) 
        frames: Number of frames in the video. 
        hidden_dim: Dimension of the hidden states of the RNNs 
        nonlinearity: Nonlinearity used in convolutional and deconvolutional layers, defaults to LeakyReLU(0.2)
        in_size: Height and width of each frame in the video (assumed square)
        step: Number of channels in the convolutional and deconvolutional layers
        conv_dim: The convolutional encoder converts each frame into an intermediate encoding vector of size conv_dim, i.e,
                  The initial video tensor (batch_size, frames, num_channels, in_size, in_size) is converted to (batch_size, frames, conv_dim)
        factorised: Toggles between full and factorised posterior for z as discussed in the paper

    Optimization:
        The model is trained with the Adam optimizer with a learning rate of 0.0002, betas of 0.9 and 0.999, with a batch size of 25 for 200 epochs

    """
    def __init__(self, f_dim=256, z_dim=32, conv_dim=128, step=256, in_size=64, hidden_dim=128,
                 frames=8, nonlinearity=None, factorised=False, in_channels=3, device=torch.device('cuda')):
        super(DisentangledVAE, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.factorised = factorised
        self.step = step
        self.in_size = in_size
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        # TODO: Check if only one affine transform is sufficient. Paper says distribution is parameterised by LSTM
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        if self.factorised is True:
            # Paper says : 1 Hidden Layer MLP. Last layers shouldn't have any nonlinearities
            self.z_inter = LinearUnit(self.conv_dim, self.hidden_dim, batchnorm=False)
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        else:
            # TODO: Check if one affine transform is sufficient. Paper says distribution is parameterised by RNN over LSTM. Last layer shouldn't have any nonlinearities
            self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
            self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
            # Each timestep is for each z so no reshaping and feature mixing
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.final_conv_size = 34
        
        self.conv = nn.Sequential(
                ConvUnit(self.in_channels, step // 4, 3, 1, 2),
                ConvUnit(step // 4, step, 3, 2, 2)
                )
        self.conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.deconv_fc = nn.Sequential(LinearUnit(self.f_dim + self.z_dim, self.conv_dim * 2, False),
                LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2), False))
        self.deconv = nn.Sequential(
                #ConvUnitTranspose(step, step, 3, 2, 2, 1),
                #ConvUnitTranspose(step, step, 3, 2, 2, 1),
                ConvUnitTranspose(step, step // 4, 3, 2, 2, 1),
                ConvUnitTranspose(step // 4, self.in_channels, 3, 1, 2, 0, nonlinearity=nn.Sigmoid()))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


        self.cuda()

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_mean_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_logvar_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for _ in range(self.frames):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out


    def encode_frames(self, x):
        # The frames are unrolled into the batch dimension for batch processing such that x goes from
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x = x.view(-1, 3, self.in_size, self.in_size)
        x = self.conv(x)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_fc(x)
        # The frame dimension is reintroduced and x shape becomes [batch_size, frames, conv_dim]
        # This technique is repeated at several points in the code
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = self.deconv_fc(zf)
        x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        x = self.deconv(x)
        return x.view(-1, self.frames, 3, self.in_size, self.in_size)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z(self, x, f):
        if self.factorised is True:
            features = self.z_inter(x)
        else:
            # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
            f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
            features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode(self, x):
        conv_x = self.encode_frames(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        return f_mean, f_logvar, f, z_mean, z_logvar, z

    def decode(self, f, z):

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decode_frames(zf)

        return recon_x

    def forward(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)

        f_mean, f_logvar, f, z_mean, z_logvar, z = self.encode(x)

        recon_x = self.decode(f, z)
        
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x




class DisentangledVAE_Attention(nn.Module):
    #f_dim = 128
    def __init__(self, f_dim=32, z_dim=128, z_dim_seq=32, conv_dim=128, step=64, in_size=64, in_channels=3,
                 frames=8, nonlinearity=None, device=torch.device('cuda:0')):

        super(DisentangledVAE_Attention, self).__init__()
        self.device = device
        self.f_dim = f_dim
        #self.f_mha_dim = f_mha_dim
        self.z_dim = z_dim
        self.z_dim_seq = z_dim_seq
        self.frames = frames
        self.conv_dim = conv_dim
        self.step = step
        self.in_size = in_size
        self.in_channels = in_channels
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        self.pos_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.frames, self.conv_dim, padding_idx=0),
            freeze=True)

        self.pos_enc_decoder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.frames, self.z_dim_seq, padding_idx=0),
            freeze=True)

        #self.mha_z_gen = mha_stack(z_dim_seq, residual=True, causality=True)
        #positional_encoding(num_units=self.conv_dim, zeros_pad=False, scale=False)
        self.pix_attn_z = pixel_attention(in_channels, 64, 64, 32)
        self.pix_attn_f = pixel_attention(in_channels, 64, 64, 32)

        self.fc_z_mean = LinearUnit(self.z_dim_seq * self.frames, self.z_dim)
        self.fc_z_logvar = LinearUnit(self.z_dim_seq * self.frames, self.z_dim)

        self.fc_z_decoder = LinearUnit(self.z_dim, self.z_dim_seq * self.frames)

        self.mha_z_convert = multihead_attention(conv_dim, z_dim_seq, residual=False)
        self.mha_z_stack = mha_stack(z_dim_seq, residual=True)

        self.mha_f_convert = multihead_attention(conv_dim, z_dim_seq, residual=False)
        self.mha_f_stack = mha_stack(z_dim_seq, residual=True)


        #self.mha_z_mean = multihead_attention(z_dim_seq, z_dim_seq, residual=True)
        #self.mha_z_logvar = multihead_attention(z_dim_seq, z_dim_seq, residual=True)

        #self.mha_f = multihead_attention(conv_dim, f_mha_dim, residual=False)

        self.mha_z_decoder = mha_stack(z_dim_seq, stack_size=5, residual=True, self_attention=False, causality=True)

        self.fc_f_mean = LinearUnit(self.z_dim_seq * self.frames, self.f_dim, False)
        self.fc_f_logvar = LinearUnit(self.z_dim_seq * self.frames, self.f_dim, False)

        self.fc_zf = LinearUnit(self.z_dim_seq + self.f_dim, self.z_dim_seq + self.f_dim)


        self.final_conv_size = 32
        
        self.conv_f = nn.Sequential(
                ConvUnit(self.in_channels, step // 4, 3, 1, 1),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                ConvUnit(step // 4, step, 3, 2, 1)
                )
        self.conv_f_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.conv_z = nn.Sequential(
                ConvUnit(self.in_channels, step // 4, 3, 1, 1),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                ConvUnit(step // 4, step, 3, 2, 1)
                )
        self.conv_z_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.deconv_fc = nn.Sequential(LinearUnit(self.f_dim + self.z_dim_seq, self.conv_dim * 2, False),
                LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2), False))
        self.deconv = nn.Sequential(
                ConvUnitTranspose(step, step // 4, 3, 2, 1, 1),
                #ConvUnitTranspose(step // 4, step // 4, 3, 2, 2, 1),
                #ConvUnitTranspose(step // 4, step // 4, 3, 2, 2, 1),
                ConvUnitTranspose(step // 4, self.in_channels, 3, 1, 1, 0, nonlinearity=nn.Sigmoid()))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        self.to(self.device)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_conv_f(self, x):
        x = x.view(-1, self.in_channels, self.in_size, self.in_size)
        x, _ = self.pix_attn_f(x)
        x = self.conv_f(x)
        #print(x.shape)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_f_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def encode_conv_z(self, x):
        x = x.view(-1, self.in_channels, self.in_size, self.in_size)
        x, _ = self.pix_attn_z(x)
        x = self.conv_z(x)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_z_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_conv(self, zf):
        x = self.deconv_fc(zf)
        x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        x = self.deconv(x)
        #print(x.shape)
        return x.view(-1, self.frames, self.in_channels, self.in_size, self.in_size)

    def encode(self, x):

        x_conv = self.encode_conv_f(x)

        
        bs = x_conv.shape[0]
        x_pos = Variable(torch.Tensor(np.array(range(self.frames)).repeat(bs)).long()).cuda()
        x_pos = x_pos.view(-1, bs)
        x_pos = x_pos.transpose(1, 0)
        x_pos = self.pos_enc(x_pos)

        x_conv = x_conv + x_pos

        f_0 = self.mha_f_convert(x_conv, x_conv, x_conv)
        f_0 = self.mha_f_stack(f_0, f_0, f_0)
        

        #f_0 = self.fc_f_encoderr

        f_0 = f_0.view(-1, self.frames * self.z_dim_seq)

        f_mean = self.fc_f_mean(f_0)

        f_logvar = self.fc_f_logvar(f_0)

        f = self.reparameterize(f_mean, f_logvar)

        #########################

        x_conv = self.encode_conv_z(x) + x_pos

        z_0 = self.mha_z_convert(x_conv, x_conv, x_conv)
        z_0 = self.mha_z_stack(z_0, z_0, z_0)

        z_0 = z_0.view(-1, self.z_dim_seq * self.frames)

        z_mean = self.fc_z_mean(z_0)
        z_logvar = self.fc_z_logvar(z_0)

        z = self.reparameterize(z_mean, z_logvar)

        return f_mean, f_logvar, f, z_mean, z_logvar, z

    def decode(self, f, z):

        z = self.fc_z_decoder(z)

        z = z.view(-1, self.frames, self.z_dim_seq)

        bs = z.shape[0]
        x_pos = Variable(torch.Tensor(np.array(range(self.frames)).repeat(bs)).long()).cuda()
        x_pos = x_pos.view(-1, bs)
        x_pos = x_pos.transpose(1, 0)
        x_pos = self.pos_enc_decoder(x_pos)

        z = z + x_pos

        z = self.mha_z_decoder(z, z, z)
        #z = self.fc_z_decoder(z)

        f = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f), dim=2)

        zf = self.fc_zf(zf)

        #print(zf.shape)

        x_recon = self.decode_conv(zf)

        #print(x_recon.shape)

        return x_recon

    def get_attn_maps(self, x):

        x = x.view(-1, self.in_channels, self.in_size, self.in_size)
        _, az = self.pix_attn_z(x)
        _, af = self.pix_attn_f(x)

        return af, az


    def forward(self, x, dropLast=False):

        if dropLast:
            x[:, -1] = 0


       
        f_mean, f_logvar, f, z_mean, z_logvar, z = self.encode(x)

        #print(z.shape, f.shape)

        #these get matched to prior

        x_recon = self.decode(f, z).view(-1, self.frames, self.in_channels, self.in_size, self.in_size)

        return f_mean, f_logvar, f, z_mean, z_logvar, z, x_recon



#generate latent codes
class Distr_Network(nn.Module):

    def __init__(self, f_dim, z_dim_seq, frames, p_dim=128, mha_stack_size=5):

        super(Distr_Network, self).__init__()

        self.f_dim = f_dim
        self.z_dim_seq = z_dim_seq
        self.frames = frames
        self.p_dim = p_dim

        self.fc_1 = LinearUnit(p_dim, z_dim_seq * frames)

        self.mha_stack_1 = mha_stack(z_dim_seq, stack_size=mha_stack_size, residual=True)

        self.pos_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.frames, self.z_dim_seq, padding_idx=0),
            freeze=True)

        self.fc_2 = LinearUnit(z_dim_seq, z_dim_seq, nonlinearity=nn.Sigmoid())

        self.fc_3 = LinearUnit(p_dim, f_dim, nonlinearity=nn.Sigmoid())

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)


    def forward(self, x):
        #input_shape: bs, p_dim

        f = self.fc_3(x)

        x = self.fc_1(x)

        x = x.view(-1, self.frames, self.z_dim_seq)

        bs = x.shape[0]
        x_pos = Variable(torch.Tensor(np.array(range(self.frames)).repeat(bs)).long(), requires_grad=False).cuda()
        x_pos = x_pos.view(-1, bs)
        x_pos = x_pos.transpose(1, 0)
        x_pos = self.pos_enc(x_pos)

        x = x + x_pos

        x = self.mha_stack_1(x, x, x)

        z = self.fc_2(x)

        return f, z

#encode real samples into latent codes
class Encoder_Network(nn.Module):

    def __init__(self, f_dim, z_dim_seq, frames, conv_dim=128, in_channels=3, step=64, in_size=64):

        super(Encoder_Network, self).__init__()

        self.f_dim = f_dim
        self.z_dim_seq = z_dim_seq
        self.frames = frames
        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.in_size = in_size
        self.step = step

        self.fc_z_mean = LinearUnit(self.z_dim_seq, self.z_dim_seq, nonlinearity=nn.Sigmoid())
        #self.fc_z_logvar = LinearUnit(self.z_dim_seq, self.z_dim_seq)

        #self.fc_z_decoder = LinearUnit(self.z_dim, self.z_dim_seq * self.frames)

        self.mha_z_convert = multihead_attention(conv_dim, z_dim_seq, residual=False)
        self.mha_z_stack = mha_stack(z_dim_seq, residual=True)

        self.mha_f_convert = multihead_attention(conv_dim, z_dim_seq, residual=False)
        self.mha_f_stack = mha_stack(z_dim_seq, residual=True)

        self.pos_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.frames, self.conv_dim, padding_idx=0),
            freeze=True)

        self.fc_f_mean = LinearUnit(self.z_dim_seq * self.frames, self.f_dim, nonlinearity=nn.Sigmoid())
        #self.fc_f_logvar = LinearUnit(self.z_dim_seq * self.frames, self.f_dim)

        self.final_conv_size = 32
        
        self.conv_f = nn.Sequential(
                ConvUnit(self.in_channels, step // 4, 3, 1, 1),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                ConvUnit(step // 4, step, 3, 2, 1)
                )
        self.conv_f_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.conv_z = nn.Sequential(
                ConvUnit(self.in_channels, step // 4, 3, 1, 1),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                #ConvUnit(step // 4, step // 4, 3, 1, 2),
                ConvUnit(step // 4, step, 3, 2, 1)
                )
        self.conv_z_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_conv_f(self, x):
        x = x.view(-1, self.in_channels, self.in_size, self.in_size)
        #x, _ = self.pix_attn_f(x)
        x = self.conv_f(x)
        #print(x.shape)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_f_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def encode_conv_z(self, x):
        x = x.view(-1, self.in_channels, self.in_size, self.in_size)
        #x, _ = self.pix_attn_z(x)
        x = self.conv_z(x)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_z_fc(x)
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def forward(self, x):

        x_conv = self.encode_conv_f(x)
        
        bs = x_conv.shape[0]
        x_pos = Variable(torch.Tensor(np.array(range(self.frames)).repeat(bs)).long(), requires_grad=False).cuda()
        x_pos = x_pos.view(-1, bs)
        x_pos = x_pos.transpose(1, 0)
        x_pos = self.pos_enc(x_pos)

        x_conv = x_conv + x_pos

        f_0 = self.mha_f_convert(x_conv, x_conv, x_conv)
        f_0 = self.mha_f_stack(f_0, f_0, f_0)
        

        #f_0 = self.fc_f_encoderr

        f_0 = f_0.view(-1, self.frames * self.z_dim_seq)

        f = self.fc_f_mean(f_0)

        #f_logvar = self.fc_f_logvar(f_0)

        #f = self.reparameterize(f_mean, f_logvar)

        #########################

        x_conv = self.encode_conv_z(x) + x_pos

        z_0 = self.mha_z_convert(x_conv, x_conv, x_conv)
        z_0 = self.mha_z_stack(z_0, z_0, z_0)

        #z_0 = z_0.view(-1, self.z_dim_seq * self.frames)

        z = self.fc_z_mean(z_0)
        #z_logvar = self.fc_z_logvar(z_0)

        #z = self.reparameterize(z_mean, z_logvar)

        return f, z



#decode latent codes into sequences of images
class Decoder_Network(nn.Module):

    def __init__(self, f_dim, z_dim_seq, frames, conv_dim=128, in_channels=3, step=64, in_size=64):

        super(Decoder_Network, self).__init__()

        self.f_dim = f_dim
        self.z_dim_seq = z_dim_seq
        self.frames = frames
        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.in_size = in_size
        self.step = step

        self.pos_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.frames, self.z_dim_seq, padding_idx=0),
            freeze=True)

        self.mha_z_decoder = mha_stack(z_dim_seq, stack_size=5, residual=True, self_attention=False, causality=True)

        self.fc_zf = LinearUnit(self.z_dim_seq + self.f_dim, self.z_dim_seq + self.f_dim)

        #self.fc_z_decoder = LinearUnit(self.z_dim, self.z_dim_seq * self.frames)

        self.final_conv_size = 32

        self.deconv_fc = nn.Sequential(LinearUnit(self.f_dim + self.z_dim_seq, self.conv_dim * 2, False),
                LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2), False))
        self.deconv = nn.Sequential(
                ConvUnitTranspose(step, step // 4, 3, 2, 1, 1),
                #ConvUnitTranspose(step // 4, step // 4, 3, 2, 2, 1),
                #ConvUnitTranspose(step // 4, step // 4, 3, 2, 2, 1),
                ConvUnitTranspose(step // 4, self.in_channels, 3, 1, 1, 0, nonlinearity=nn.Sigmoid()))

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)

    def decode_conv(self, zf):
        x = self.deconv_fc(zf)
        x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        x = self.deconv(x)
        #print(x.shape)
        return x.view(-1, self.frames, self.in_channels, self.in_size, self.in_size)

    def decode(self, f0, z0):

        #z = self.fc_z_decoder(z)

        z = z0.view(-1, self.frames, self.z_dim_seq)

        bs = z.shape[0]
        x_pos = Variable(torch.Tensor(np.array(range(self.frames)).repeat(bs)).long(), requires_grad=False).cuda()
        x_pos = x_pos.view(-1, bs)
        x_pos = x_pos.transpose(1, 0)
        x_pos = self.pos_enc(x_pos)

        z = z + x_pos

        z = self.mha_z_decoder(z, z, z)
        #z = self.fc_z_decoder(z)

        f = f0.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f), dim=2)

        zf = self.fc_zf(zf)

        #print(zf.shape)

        x_recon = self.decode_conv(zf)

        #print(x_recon.shape)

        return x_recon

    def forward(self, f, z):

        x_recon = self.decode(f, z).view(-1, self.frames, self.in_channels, self.in_size, self.in_size)

        return x_recon


#discriminate between encoded samples(0) and generated codes(1)
class Discr_Network(nn.Module):

    def __init__(self, f_dim, z_dim_seq, frames, p_dim=32, mha_stack_size=5):

        super(Discr_Network, self).__init__()

        self.f_dim = f_dim
        self.z_dim_seq = z_dim_seq
        self.frames = frames
        self.p_dim = p_dim

        self.mha_stack_1 = mha_stack(z_dim_seq, stack_size=mha_stack_size, residual=True)

        self.pos_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.frames, self.z_dim_seq, padding_idx=0),
            freeze=True)

        self.attn = attention_layer(frames, z_dim_seq)
        #self.fc_dummy = LinearUnit(frames * z_dim_seq, z_d)

        self.fc_1 = LinearUnit(z_dim_seq + f_dim, p_dim)


        self.fc_2 = LinearUnit(p_dim, 1, nonlinearity=nn.Tanh())

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)

    def forward(self, f0, z0):
        #for debugging
        #return torch.Tensor([0])

        #input_shape: bs, p_dim

        z = z0.view(-1, self.frames, self.z_dim_seq)

        bs = z.shape[0]
        x_pos = Variable(torch.Tensor(np.array(range(self.frames)).repeat(bs)).long(), requires_grad=False).cuda()
        x_pos = x_pos.view(-1, bs)
        x_pos = x_pos.transpose(1, 0)
        x_pos = self.pos_enc(x_pos)

        z = z + x_pos

        z = self.mha_stack_1(z, z, z)

        #z = self.attn(z)
        z = torch.mean(z, dim=(-2,))

        zf = torch.cat([z, f0], dim=-1)

        zf = self.fc_1(zf)

        zf = self.fc_2(zf)

        return zf



class DisentangledVAE_Adversarial(nn.Module):

    def __init__(self, f_dim=32, z_dim_seq=32, frames=10, p_dim=128, mha_stack_size=5, in_channels=3):

        super(DisentangledVAE_Adversarial, self).__init__()

        self.f_dim = f_dim
        self.z_dim_seq = z_dim_seq
        self.frames = frames
        self.p_dim = p_dim
        self.in_channels = in_channels

        self.distr = Distr_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames, p_dim=p_dim)

        self.enc = Encoder_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames, in_channels=in_channels)

        self.dec = Decoder_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames, in_channels=in_channels)

        self.discr = Discr_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames)

    def forward(self, x, get_score=True):

        f, z = self.enc(x)

        x_recon = self.dec(f, z)

        if get_score == False:
            return x_recon

        D_score = self.discr(f, z)

        return x_recon, D_score

    def forward_score(self, x):

        f, z = self.enc(x)

        D_score = self.discr(f, z)

        return D_score

    def gen_codes(self, batch_size=64):

        inputs_r = torch.randn((batch_size, self.p_dim)).cuda()

        f, z = self.distr(inputs_r)

        D_score = self.discr(f, z)

        return f, z, D_score

    def gen_seq(self, batch_size=64, get_score=False):


        inputs_r = torch.randn((batch_size, self.p_dim)).cuda()

        f, z = self.distr(inputs_r)

        x_gen = self.dec(f, z)


        if get_score == False:

            return x_gen

        D_score = self.discr(f, z)

        return x_gen, D_score

    def zero_grad_all(self):

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)

        self.distr.optimizer.zero_grad()
        self.discr.optimizer.zero_grad()
        self.enc.optimizer.zero_grad()
        self.dec.optimizer.zero_grad()

    def enc_dec_step(self):

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)

        self.enc.optimizer.step()
        self.dec.optimizer.step()

    def gen_codes_step(self):

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)

        self.discr.optimizer.step()
        self.discr.optimizer.step()

    def encode(self, x):

        return self.enc(x)

    def decode(self, f, z):

        return self.dec(f, z)



import matplotlib.pyplot as plt

def plot(x):

    x0 = x.detach().cpu()

    plt.scatter(x0 [:, 0], x0 [:, 1])
    plt.show()


if __name__ == '__main__':

    batch_size = 32
    frames = 10
    channels = 3

    p_dim = 128

    f_dim = 32
    z_dim_seq = 32

    inputs_r = torch.randn((batch_size, p_dim)).cuda()

    #plot(inputs_r)

    inputs_x = torch.randn((batch_size, frames, channels, 64, 64)).cuda()

    distr = Distr_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames, p_dim=p_dim)

    enc = Encoder_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames)

    o1_f, o1_z = distr(inputs_r)

    o2_f, o2_z = enc(inputs_x)

    print(o1_f.shape, o1_z.shape)

    print(o2_f.shape, o2_z.shape)

    discr = Discr_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames)

    o3 = discr(o1_f, o1_z)

    o4 = discr(o2_f, o2_z)

    print(o3)

    print(o4)

    dec = Decoder_Network(f_dim=f_dim, z_dim_seq=z_dim_seq, frames=frames)

    o5 = dec(o1_f, o1_z)

    print(o5.shape)

    #plot(o1_f)

    """
    

    model = DisentangledVAE_Attention(frames=frames)

    inputs = torch.randn((batch_size * frames, channels, 64, 64)).to(model.device) * 0.2 - 0.05
    inputs = torch.clamp(inputs, 0., 1.)

    print(inputs.min(), inputs.max(), inputs.mean(), inputs.std())

    _, _, _, _, _, _, out = model.forward(inputs)

    print(out.shape)

    print(out.min(), out.max(), out.mean(), out.std())
    """

    """
    conv_x = model.encode_frames(inputs)

    print(conv_x.shape)

    mha_8 = multihead_attention(2048, 8, residual=False)

    z = mha_8(conv_x, conv_x, conv_x)

    print(z.shape)
    """


    #x = torch.randn((frames, 3, 64, 64))

    #m = pixel_attention(3, 64)

    #x = m(x)


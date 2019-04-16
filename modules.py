import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# A block consisting of convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, batchnorm=False, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
                    nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

# A block consisting of a transposed convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnitTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, batchnorm=False, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding),
                    nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

# A block consisting of an affine layer, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=False, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.residual = (in_features == out_features)
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

        self.cuda()

    def forward(self, x):
        out = self.model(x)
        if self.residual:
            out = out + x
        return out


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.
        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features).cuda())
        self.beta = nn.Parameter(torch.zeros(features).cuda())

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta



def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class positional_encoding(nn.Module):

    def __init__(self, num_units, zeros_pad=True, scale=True):
        '''Sinusoidal Positional_Encoding.
        Args:
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
        '''
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs):
        # inputs: A 2d Tensor with shape of (N, T).
        N, T = inputs.size()[0: 2]

        # First part of the PE function: sin and cos argument
        position_ind = Variable(torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1).long()).cuda()
        position_enc = torch.Tensor([
            [pos / np.power(10000, 2. * i / self.num_units) for i in range(self.num_units)]
            for pos in range(T)]).cuda()

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a Variable
        lookup_table = Variable(position_enc)

        if self.zeros_pad:
            lookup_table = torch.cat((Variable(torch.zeros(1, self.num_units)),
                                     lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1


        outputs = self._backend.Embedding.apply(
            position_ind, lookup_table, padding_idx, None, 2, False, False)   # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * self.num_units ** 0.5

        return outputs


class multihead_attention(nn.Module):

    def __init__(self, input_size, num_units, residual=True, num_heads=8, dropout_rate=0, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.input_size = input_size
        self.num_units = num_units
        self.residual = residual
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.input_size, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.input_size, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.input_size, self.num_units), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

        self.normalization = layer_normalization(self.num_units)

        self.cuda()

    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]

        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        padding = Variable(torch.ones(*outputs.size()).cuda() * (-2 ** 32 + 1))
        condition = key_masks.eq(0.).float()
        outputs = padding * condition + outputs * (1. - condition)

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size()).cuda()  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            # print(tril)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)

            padding = Variable(torch.ones(*masks.size()).cuda() * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)

        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks

        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        if self.residual:
            outputs = outputs + queries

        # Normalize
        outputs = self.normalization(outputs)  # (N, T_q, C)

        return outputs


class mha_stack(nn.Module):
    def __init__(self, num_units, stack_size=3, residual=True, num_heads=4, dropout_rate=0, causality=False, self_attention=True):

        super(mha_stack, self).__init__()
        self.num_units = num_units
        self.residual = residual
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.stack_size = stack_size

        self.self_attention = self_attention

        self.mha_layers = [multihead_attention(num_units, num_units, residual, num_heads, dropout_rate, causality) for i in range(stack_size)]
        self.ff = [LinearUnit(num_units, num_units) for i in range(stack_size)]

        self.cuda()

    def forward(self, queries, keys, values):

        for i in range(self.stack_size):
            enc = self.mha_layers[i](queries, keys, values)
            enc = self.ff[i](enc)
            if self.self_attention:
                keys, values = enc, enc
            queries = enc

        return enc


class pixel_attention(nn.Module):

    def __init__(self, in_channels, in_side, hidden_channels, hidden_linear):
        super(pixel_attention, self).__init__()

        self.in_channels = in_channels

        self.conv = ConvUnit(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = ConvUnit(hidden_channels, 1, 1)

        self.lin = nn.Sequential(LinearUnit(in_channels * in_side * in_side, hidden_linear),
                                 LinearUnit(hidden_linear, in_side * in_side))

    def forward(self, x):

        a = self.conv(x)

        a = self.conv2(a)

        ash = a.shape

        al = self.lin(x.view(-1, self.in_channels * ash[-2] * ash[-1]))

        a = a.view(-1, ash[-2] * ash[-1]) + al

        a = torch.sigmoid(a)

        #a = a.softmax(dim=-1)

        a = a.view(ash)

        return x * a, a


class attention_layer(nn.Module):

    def __init__(self, seq_dim, input_size):

        super(attention_layer, self).__init__()
        self.seq_dim = seq_dim
        self.input_size = input_size

        self.W = LinearUnit(input_size, 1, nonlinearity=nn.Tanh())

        self.cuda()

    def forward(self, x):

        a = self.W(x)

        #a = a.view(-1, self.seq_dim)

        a = a.softmax(dim=-2)

        #print(a)

        y = x * a

        return torch.sum(y, dim=(1,))#y.sum(dim=(1,))